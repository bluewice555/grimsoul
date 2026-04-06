import json
import re
from pathlib import Path

import cv2
import easyocr
from rapidocr_onnxruntime import RapidOCR


ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "docs" / "sources" / "monster_1.jpg"
CELLS_PATH = ROOT / ".work" / "ocr" / "monster_1_cells.json"
OUTPUT_JSON = ROOT / "docs" / "dataset" / "monster_overlap.json"
OUTPUT_CONFLICTS = ROOT / ".work" / "ocr" / "monster_1_overlap_conflicts.json"

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
    text = text.replace("|", "I")
    text = text.replace("’", "'").replace("`", "'")
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ;")


def normalize_text(text: str) -> str:
    text = clean_text(text)
    text = text.replace("+", "x").replace("*", "x")
    text = text.lower()
    text = re.sub(r"[^a-z0-9(),.;:/%<>\- x]", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def preprocess_a(crop):
    big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )


def preprocess_b(crop):
    big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 5, 35, 35)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def preprocess_c(crop):
    big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    denoise = cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)
    sharp = cv2.addWeighted(
        denoise, 1.5, cv2.GaussianBlur(denoise, (0, 0), 1.0), -0.5, 0
    )
    _, th = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


METHODS = {
    "a_adapt": preprocess_a,
    "b_bilat_otsu": preprocess_b,
    "c_denoise_otsu": preprocess_c,
}


def easy_text(reader, image) -> str:
    return clean_text(" ".join(reader.readtext(image, detail=0, paragraph=False)))


def rapid_text(engine, image) -> str:
    results, _ = engine(image)
    if not results:
        return ""
    return clean_text(" ".join(item[1] for item in results))


def pick_overlap(candidates):
    overlaps = []
    easy_values = {}
    rapid_values = {}

    for c in candidates:
        if c["easy_norm"]:
            easy_values.setdefault(c["easy_norm"], []).append(c["easy"])
        if c["rapid_norm"]:
            rapid_values.setdefault(c["rapid_norm"], []).append(c["rapid"])
        if c["easy_norm"] and c["easy_norm"] == c["rapid_norm"]:
            overlaps.extend([c["easy"], c["rapid"]])

    if overlaps:
        return max(overlaps, key=len)

    cross = set(easy_values.keys()) & set(rapid_values.keys())
    if cross:
        best_key = max(cross, key=len)
        pool = easy_values[best_key] + rapid_values[best_key]
        return max(pool, key=len)

    consensus_pool = {}
    for c in candidates:
        if c["easy_norm"]:
            consensus_pool.setdefault(c["easy_norm"], []).append(c["easy"])
        if c["rapid_norm"]:
            consensus_pool.setdefault(c["rapid_norm"], []).append(c["rapid"])
    consensus = [texts for _, texts in consensus_pool.items() if len(texts) >= 2]
    if consensus:
        merged = [t for group in consensus for t in group]
        return max(merged, key=len)

    return None


def main() -> int:
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")

    payload = json.loads(CELLS_PATH.read_text(encoding="utf-8"))
    xs = payload["vertical_lines"]
    ys = payload["horizontal_lines"]
    total_rows = 132

    reader = easyocr.Reader(["en"], gpu=True)
    rapid = RapidOCR(
        det_use_cuda=True,
        det_model_path=None,
        cls_use_cuda=True,
        cls_model_path=None,
        rec_use_cuda=True,
        rec_model_path=None,
    )
    print("EasyOCR device: GPU (forced)")
    print("RapidOCR device: GPU (forced)")

    rows = []
    conflicts = []

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

            method_results = []
            for name, fn in METHODS.items():
                processed = fn(crop)
                easy_value = easy_text(reader, processed)
                rapid_value = rapid_text(rapid, processed)
                method_results.append(
                    {
                        "method": name,
                        "easy": easy_value,
                        "rapid": rapid_value,
                        "easy_norm": normalize_text(easy_value),
                        "rapid_norm": normalize_text(rapid_value),
                    }
                )

            if all((m["easy"] == "" and m["rapid"] == "") for m in method_results):
                continue

            selected = pick_overlap(method_results)
            if selected is not None:
                row[field] = selected
            else:
                row_conflicts.append({"field": field, "methods": method_results})

        rows.append(row)
        if row_conflicts:
            conflicts.append(
                {"row": row_index + 1, "id": row.get("ID", ""), "conflicts": row_conflicts}
            )

        if (row_index + 1) % 20 == 0:
            print(f"progress {row_index + 1}/{total_rows}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONFLICTS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_CONFLICTS.write_text(
        json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(OUTPUT_JSON)
    print(OUTPUT_CONFLICTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
