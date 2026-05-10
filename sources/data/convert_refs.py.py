import msgpack
import json
import os

def convert_bin_to_json(bin_path, json_path):
    print(f"正在讀取原始檔案: {bin_path}...")
    
    if not os.path.exists(bin_path):
        print(f"錯誤：找不到檔案 {bin_path}")
        return

    try:
        # 1. 以二進位模式讀取 refs.bin
        with open(bin_path, 'rb') as f:
            bin_data = f.read()

        # 2. 使用 msgpack 進行解包 (Unpack)
        # raw=False 會將字串自動轉為 utf-8，確保 JSON 相容性
        # strict_map_key=False 增加對非標準 Key 的相容性
        data = msgpack.unpackb(bin_data, raw=False, strict_map_key=False)

        # 3. 轉換為純 JSON 並儲存
        # indent=2 方便你閱讀結構
        # ensure_ascii=False 確保如果裡面有原始非英文編碼能正確顯示
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"轉換成功！純淨 JSON 已儲存至: {json_path}")
        print(f"檔案大小: {os.path.getsize(json_path) / 1024:.2f} KB")

    except Exception as e:
        print(f"轉換失敗。錯誤原因: {str(e)}")

if __name__ == "__main__":
    # 設定你的檔案名稱
    input_file = "refs.bin"
    output_file = "refs_pure.json"
    
    convert_bin_to_json(input_file, output_file)