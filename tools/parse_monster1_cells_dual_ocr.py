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
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def preprocess_b(crop):
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 5, 35, 35)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def preprocess_c(crop):
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


METHODS = {
    "a_lanczos_otsu": preprocess_a,
    "b_bilat_otsu": preprocess_b,
    "c_clahe_otsu": preprocess_c,
}


def easy_text(reader, image) -> str:
    return clean_text(" ".join(reader.readtext(image, detail=0, paragraph=False)))


def rapid_text(engine, image) -> str:
    results, _ = engine(image)
    if not results:
        return ""
    return clean_text(" ".join(item[1] for item in results))


def try_stage(stage_name, image, reader, rapid):
    easy_value = easy_text(reader, image)
    rapid_value = rapid_text(rapid, image)
    return {
        "method": stage_name,
        "easy": easy_value,
        "rapid": rapid_value,
        "easy_norm": normalize_text(easy_value),
        "rapid_norm": normalize_text(rapid_value),
    }


def decide_with_rules(method_results):
    # Rule 1: same-stage easy == rapid and both non-empty => absolute accept
    for m in method_results:
        if m["easy_norm"] and m["easy_norm"] == m["rapid_norm"]:
            chosen = m["easy"] if len(m["easy"]) >= len(m["rapid"]) else m["rapid"]
            return chosen, "rule1_same_stage_agree", m["method"], m["easy_norm"]

    # Rule 2: cross-engine matching across preprocessing stages
    # (easy_any_stage vs rapid_any_stage), explicitly no self-engine comparison.
    easy_map = {}
    rapid_map = {}
    for m in method_results:
        if m["easy_norm"]:
            easy_map.setdefault(m["easy_norm"], []).append(m["easy"])
        if m["rapid_norm"]:
            rapid_map.setdefault(m["rapid_norm"], []).append(m["rapid"])
    cross = [k for k in easy_map.keys() if k in rapid_map]
    if cross:
        cross.sort(key=len, reverse=True)
        k = cross[0]
        pool = easy_map[k] + rapid_map[k]
        return max(pool, key=len), "rule2_cross_stage_cross_engine_agree", "cross", k

    # Rule 3: one engine has data, the other engine has none => accept non-empty side
    any_easy = [m["easy"] for m in method_results if m["easy_norm"]]
    any_rapid = [m["rapid"] for m in method_results if m["rapid_norm"]]
    if any_easy and not any_rapid:
        return max(any_easy, key=len), "rule3_easy_only", "cross", normalize_text(max(any_easy, key=len))
    if any_rapid and not any_easy:
        return max(any_rapid, key=len), "rule3_rapid_only", "cross", normalize_text(max(any_rapid, key=len))

    return None, "conflict", "none", ""


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
            stages = [("raw", crop)] + [(name, fn(crop)) for name, fn in METHODS.items()]
            for stage_name, stage_img in stages:
                method_results.append(try_stage(stage_name, stage_img, reader, rapid))

            if all((m["easy"] == "" and m["rapid"] == "") for m in method_results):
                continue

            selected, reason, used_stage, selected_norm = decide_with_rules(method_results)
            unique_norms = len(
                set([m["easy_norm"] for m in method_results if m["easy_norm"]])
                | set([m["rapid_norm"] for m in method_results if m["rapid_norm"]])
            )
            if selected is not None:
                row[field] = selected
                print(
                    f"cell r{row_index + 1:03d} c{col_index + 1:02d} {field}: "
                    f"accepted ({reason}, stage={used_stage}, unique_norms={unique_norms}, norm={selected_norm})"
                )
            else:
                row_conflicts.append({"field": field, "methods": method_results})
                non_empty = sum(
                    1
                    for m in method_results
                    for v in (m["easy_norm"], m["rapid_norm"])
                    if v
                )
                print(
                    f"cell r{row_index + 1:03d} c{col_index + 1:02d} {field}: "
                    f"conflict ({reason}, outputs={non_empty}, unique_norms={unique_norms})"
                )

        rows.append(row)
        if row_conflicts:
            conflicts.append({"row": row_index + 1, "id": row.get("ID", ""), "conflicts": row_conflicts})

        print(
            f"row progress {row_index + 1}/{total_rows} "
            f"(row_conflicts={len(row_conflicts)}, total_conflicts={len(conflicts)})"
        )

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONFLICTS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_CONFLICTS.write_text(json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT_JSON)
    print(OUTPUT_CONFLICTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
