# convert_refs_unity_v3.py
# Stdlib-only converter for this Unity game's refs.bin custom binary table.
# Outputs only the parsed refs data.

import argparse
import base64
import json
import os
import struct
from typing import Any


class RefsBinParser:
    def __init__(self, data: bytes):
        self.data = data
        self.n = len(data)
        self.counts: dict[int, int] = {}

    def read_uvar(self, pos: int) -> tuple[int, int]:
        value = 0
        shift = 0
        while True:
            if pos >= self.n:
                raise EOFError("unexpected EOF while reading varint")
            byte = self.data[pos]
            pos += 1
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                return value, pos
            shift += 7
            if shift > 70:
                raise ValueError("varint is too long")

    def read_string(self, pos: int) -> tuple[str | dict[str, str], int]:
        length, pos = self.read_uvar(pos)
        raw = self.data[pos:pos + length]
        if len(raw) != length:
            raise EOFError("unexpected EOF while reading string")
        pos += length
        try:
            return raw.decode("utf-8"), pos
        except UnicodeDecodeError:
            return {"__bytes_base64__": base64.b64encode(raw).decode("ascii")}, pos

    def parse_value(self, pos: int) -> tuple[Any, int]:
        if pos >= self.n:
            raise EOFError("unexpected EOF while reading value")

        tag = self.data[pos]
        pos += 1
        self.counts[tag] = self.counts.get(tag, 0) + 1

        # 0x00: null / empty marker
        if tag == 0x00:
            return None, pos

        # 0x03: boolean, followed by 0x00 or 0x01
        if tag == 0x03:
            if pos >= self.n:
                raise EOFError("unexpected EOF while reading bool")
            value = bool(self.data[pos])
            return value, pos + 1

        # 0x0e: little-endian double
        if tag == 0x0E:
            if pos + 8 > self.n:
                raise EOFError("unexpected EOF while reading double")
            return struct.unpack("<d", self.data[pos:pos + 8])[0], pos + 8

        # 0x12: UTF-8 string with varint length
        if tag == 0x12:
            return self.read_string(pos)

        # 0x15: object/map, varint pair count, then key/value typed values
        if tag == 0x15:
            count, pos = self.read_uvar(pos)
            obj: dict[str, Any] = {}
            for _ in range(count):
                key, pos = self.parse_value(pos)
                value, pos = self.parse_value(pos)
                key = self.key_to_json_key(key)

                # Preserve duplicate keys instead of silently overwriting them.
                if key in obj:
                    if not isinstance(obj[key], list) or not (
                        len(obj[key]) == 2 and isinstance(obj[key][0], dict) and obj[key][0].get("__duplicate_key__") is True
                    ):
                        obj[key] = [{"__duplicate_key__": True}, obj[key]]
                    obj[key].append(value)
                else:
                    obj[key] = value
            return obj, pos

        # 0x16: array/list, varint length, then typed values
        if tag == 0x16:
            count, pos = self.read_uvar(pos)
            arr = []
            for _ in range(count):
                value, pos = self.parse_value(pos)
                arr.append(value)
            return arr, pos

        # 0x1e: compact reference token used by this file.
        # The referenced dictionary is not embedded as plain text here, so preserve it losslessly.
        if tag == 0x1E:
            ref_id, pos = self.read_uvar(pos)
            return {"__ref__": ref_id}, pos

        # 0x20: positive varint numeric value.
        if tag == 0x20:
            value, pos = self.read_uvar(pos)
            return value, pos

        # 0x21: negative varint numeric value.
        # This file uses the tag as the sign marker: stored magnitude N -> value -N.
        if tag == 0x21:
            value, pos = self.read_uvar(pos)
            return -value, pos

        raise ValueError(f"unknown tag 0x{tag:02x} at byte offset {pos - 1}")

    @staticmethod
    def key_to_json_key(key: Any) -> str:
        if isinstance(key, str):
            return key
        if isinstance(key, dict) and set(key.keys()) == {"__ref__"}:
            return f"@ref:{key['__ref__']}"
        return json.dumps(key, ensure_ascii=False, sort_keys=True)


def parse_refs_bin(data: bytes, start_offset: int = 2) -> Any:
    parser = RefsBinParser(data)
    root, _end = parser.parse_value(start_offset)
    return root


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert this Unity game's refs.bin custom binary table to JSON.")
    ap.add_argument("input", nargs="?", default="refs.bin", help="input refs.bin path")
    ap.add_argument("output", nargs="?", default="refs_pure.json", help="output json path")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"找不到檔案: {args.input}")

    print(f"正在讀取原始檔案: {args.input}...")
    with open(args.input, "rb") as f:
        raw = f.read()

    parsed = parse_refs_bin(raw, start_offset=2)

    print("解析成功")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    print(f"輸出檔案: {args.output}")
    print(f"輸出大小: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
