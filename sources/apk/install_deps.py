import subprocess
import sys
import os
import json

def install_and_import(package):
    """自動檢查並安裝缺少的 Python 套件"""
    try:
        __import__(package)
        print(f"✅ 套件 [{package}] 已存在，無需重複安裝。")
    except ImportError:
        print(f"📦 找不到套件 [{package}]，正在為您自動安裝...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"🎉 套件 [{package}] 安裝成功！")
        except Exception as e:
            print(f"❌ 套件 [{package}] 安裝失敗，請手動執行: pip install {package}")
            print(f"錯誤原因: {str(e)}")

# ==================== 1. 自動安裝依賴套件 ====================
print("--- 開始檢查環境依賴 ---")
install_and_import("msgpack")
install_and_import("lz4")
print("--- 環境檢查完成 ---\n")

# 成功安裝後匯入
import msgpack
import lz4.block

# ==================== 2. Unity 核心解密與解壓邏輯 ====================
def try_unity_uncompress_and_decode(raw_bytes):
    """
    針對 Unity 引擎產生的 MessagePack / Addressables 二進位檔案進行脫殼。
    支援：1. 未壓縮但有標頭位移  2. LZ4 區塊壓縮 (含動態長度標頭)
    """
    # 嘗試 A：如果檔案其實沒壓縮，只是前面有 Unity 的自訂標頭（如 Addressables 索引）
    # 暴力掃描前 64 個位元組，尋找合法的 MessagePack 進入點
    for offset in range(0, min(64, len(raw_bytes))):
        test_bytes = raw_bytes[offset:]
        try:
            unpacker = msgpack.Unpacker(raw=True, strict_map_key=False)
            unpacker.feed(test_bytes)
            next(unpacker)
            print(f"💡 偵測成功：在 Offset {offset} 處尋獲未壓縮的標準 MessagePack 資料。")
            return test_bytes
        except Exception:
            pass

    # 嘗試 B：處理 Unity 最常見的 LZ4 區塊壓縮
    # Unity 通常會在壓縮數據前方塞入 4~24 位元組的「解壓後原始長度」或「封包長度標記」
    print("正在嘗試對 Unity LZ4 壓縮層進行暴力破譯...")
    for offset in [4, 8, 12, 16, 20, 24]:
        if len(raw_bytes) > offset:
            try:
                # 嘗試從不同位移切入解壓
                uncompressed = lz4.block.decompress(raw_bytes[offset:])
                
                # 驗證解壓後的內容是否為合法的 MessagePack 結構
                unpacker = msgpack.Unpacker(raw=True, strict_map_key=False)
                unpacker.feed(uncompressed)
                next(unpacker)
                
                print(f"🎉 破譯成功！在 Offset {offset} 處成功完成 LZ4 解壓縮並識別出資料結構。")
                return uncompressed
            except Exception:
                continue

    # 嘗試 C：直接對全檔進行無標頭的標準 LZ4 解壓
    try:
        uncompressed = lz4.block.decompress(raw_bytes)
        print("🎉 破譯成功！全檔案無標頭直接完成 LZ4 解壓縮。")
        return uncompressed
    except Exception:
        pass

    print("⚠️ 警告：此檔案未通過 Unity 常規解壓與特徵校驗，將嘗試以原始二進位強制解析...")
    return raw_bytes

def decode_bytes(data):
    """遞迴走訪結構：將 bytes 轉為 UTF-8 字串，若為非文字之二進位殘留則轉為 Hex 確保不崩潰"""
    if isinstance(data, bytes):
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            return f"__hex__{data.hex()}"
    elif isinstance(data, dict):
        return {str(decode_bytes(k)): decode_bytes(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [decode_bytes(item) for item in data]
    return data

# ==================== 3. 主轉換程序 ====================
def convert_bin_to_json(bin_path, json_path):
    print(f"正在讀取原始檔案: {bin_path}...")
    
    if not os.path.exists(bin_path):
        print(f"錯誤：在當前目錄找不到檔案 [{bin_path}]")
        return

    try:
        with open(bin_path, 'rb') as f:
            bin_data = f.read()

        # 執行 Unity 專用解壓脫殼
        processed_bytes = try_unity_uncompress_and_decode(bin_data)

        # 讀取 MessagePack 串流
        unpacker = msgpack.Unpacker(raw=True, strict_map_key=False)
        unpacker.feed(processed_bytes)
        
        extracted_data = []
        for unpacked_object in unpacker:
            extracted_data.append(unpacked_object)
        
        if not extracted_data:
            print("❌ 錯誤：解開後的區塊中未包含任何有效的 MessagePack 數據。")
            return

        # 結構優化：若只有單一主結構則直接解開，多個結構則打包成 Array
        final_data = extracted_data[0] if len(extracted_data) == 1 else extracted_data

        print("正在清洗編碼型態並產出 JSON 檔案...")
        cleaned_data = decode_bytes(final_data)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

        print(f"\n✨ 任務完成！")
        print(f"💾 純淨 JSON 檔已儲存至: {json_path}")
        print(f"📊 產出檔案大小: {os.path.getsize(json_path) / 1024:.2f} KB")

    except Exception as e:
        print(f"❌ 轉換失敗。後端錯誤原因: {str(e)}")

if __name__ == "__main__":
    # 設定目標檔案名稱 (請確保 refs.bin 與此腳本放同一個資料夾)
    input_file = "refs.bin"
    output_file = "refs_pure.json"
    
    convert_bin_to_json(input_file, output_file)