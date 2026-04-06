import json
import re
from pathlib import Path

import cv2
import easyocr
from rapidocr_onnxruntime import RapidOCR


ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "docs" / "sources" / "monster_1.jpg"
CELLS_PATH = ROOT / ".work" / "ocr" / "monster_1_cells.json"
OUTPUT_JSON = ROOT / "docs" / "dataset" / "monster.json"
OUTPUT_CONFLICTS = ROOT / ".work" / "ocr" / "monster_1_conflicts.json"

FIELDS = [
    "ID",
    "Health",
    "Armor (Est.)",
    "Damage Reduction (%)",
    "Physical ATK",
    "Elemental ATK",
    "Special Mechanic(s) Summary",
    "Common Location(s)",
]


def clean_text(text: str) -> str:
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("，", ",").replace("；", ";").replace("：", ":")
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ;")


def normalize_text(text: str) -> str:
    text = clean_text(text)
    text = text.replace("±", "x").replace("+", "x").replace("*", "x")
    text = text.replace("簣", "x").replace("Ёг", "x")
    text = text.lower()
    text = re.sub(r"[^a-z0-9(),.;:/%<>\- x]", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def preprocess(image, scale: int):
    enlarged = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = enlarged if len(enlarged.shape) == 2 else cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )


def easy_text(reader, crop) -> str:
    results = reader.readtext(crop, detail=0, paragraph=False)
    return clean_text(" ".join(results))


def rapid_text(engine, crop) -> str:
    results, _ = engine(crop)
    if not results:
        return ""
    return clean_text(" ".join(item[1] for item in results))


def main() -> int:
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")

    payload = json.loads(CELLS_PATH.read_text(encoding="utf-8"))
    xs = payload["vertical_lines"]
    ys = payload["horizontal_lines"]

    reader = easyocr.Reader(["en"], gpu=False)
    rapid = RapidOCR()

    rows = []
    conflicts = []
    total_rows = 132

    for row_index in range(total_rows):
        top = 49 if row_index == 0 else ys[row_index]
        bottom = ys[row_index + 1]
        row = {"row": row_index + 1}
        row_conflicts = []

        for col_index, field in enumerate(FIELDS):
            x1 = xs[col_index]
            x2 = xs[col_index + 1]
            crop = image[top:bottom, x1:x2]
            if crop.size == 0:
                continue

            h, w = crop.shape[:2]
            inset_x = min(4, max(1, w // 30))
            inset_y = min(3, max(1, h // 20))
            crop = crop[
                inset_y : max(inset_y + 1, h - inset_y),
                inset_x : max(inset_x + 1, w - inset_x),
            ]

            easy = easy_text(reader, preprocess(crop, 4))
            rapid_value = rapid_text(rapid, preprocess(crop, 3))
            easy_norm = normalize_text(easy)
            rapid_norm = normalize_text(rapid_value)

            if easy_norm and easy_norm == rapid_norm:
                row[field] = easy if len(easy) >= len(rapid_value) else rapid_value
            elif easy_norm or rapid_norm:
                row_conflicts.append(
                    {
                        "field": field,
                        "easyocr": easy,
                        "rapidocr": rapid_value,
                    }
                )

        rows.append(row)
        if row_conflicts:
            conflicts.append(
                {
                    "row": row_index + 1,
                    "id": row.get("ID", ""),
                    "conflicts": row_conflicts,
                }
            )

        if (row_index + 1) % 20 == 0:
            print(f"progress {row_index + 1}/{total_rows}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_CONFLICTS.write_text(json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT_JSON)
    print(OUTPUT_CONFLICTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
