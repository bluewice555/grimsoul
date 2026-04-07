import argparse
import json
import re
from pathlib import Path

import cv2
import easyocr
from rapidocr_onnxruntime import RapidOCR


ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "docs" / "sources" / "weapons.jpg"
BOXES_PATH = ROOT / ".work" / "ocr" / "weapon_cells_boxes.json"
OUT_RAW = ROOT / ".work" / "ocr" / "weapon_raw.json"
OUT_CONFLICTS = ROOT / ".work" / "ocr" / "weapon_conflicts.json"


def clean_text(text: str) -> str:
    text = text.replace("|", "I")
    text = text.replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text: str) -> str:
    text = clean_text(text)
    text = text.replace("+", "x").replace("*", "x")
    text = text.lower()
    text = re.sub(r"[^a-z0-9(),.;:/%<>\- x]", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def preprocess_1(crop):
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def preprocess_2(crop):
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 5, 35, 35)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def preprocess_3(crop):
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


METHODS = {
    "raw": lambda x: x,
    "p1_lanczos_otsu": preprocess_1,
    "p2_bilat_otsu": preprocess_2,
    "p3_clahe_otsu": preprocess_3,
}


def easy_text(reader, image) -> str:
    return clean_text(" ".join(reader.readtext(image, detail=0, paragraph=False)))


def rapid_text(engine, image) -> str:
    results, _ = engine(image)
    if not results:
        return ""
    return clean_text(" ".join(item[1] for item in results))


def decide(method_results):
    # Rule 1: same stage, easy==rapid, and both non-empty => absolute accept
    for m in method_results:
        if m["easy_norm"] and m["easy_norm"] == m["rapid_norm"]:
            chosen = m["easy"] if len(m["easy"]) >= len(m["rapid"]) else m["rapid"]
            return chosen, "rule1_same_stage_agree", m["method"], m["easy_norm"]

    # Rule 2: easy(any stage) vs rapid(any stage), cross-engine only
    easy_map = {}
    rapid_map = {}
    for m in method_results:
        if m["easy_norm"]:
            easy_map.setdefault(m["easy_norm"], []).append(m["easy"])
        if m["rapid_norm"]:
            rapid_map.setdefault(m["rapid_norm"], []).append(m["rapid"])
    cross = [k for k in easy_map if k in rapid_map]
    if cross:
        cross.sort(key=len, reverse=True)
        k = cross[0]
        pool = easy_map[k] + rapid_map[k]
        return max(pool, key=len), "rule2_cross_stage_cross_engine_agree", "cross", k

    # Rule 3: one engine all empty, the other has data => accept non-empty side
    any_easy = [m["easy"] for m in method_results if m["easy_norm"]]
    any_rapid = [m["rapid"] for m in method_results if m["rapid_norm"]]
    if any_easy and not any_rapid:
        best = max(any_easy, key=len)
        return best, "rule3_easy_only", "cross", normalize_text(best)
    if any_rapid and not any_easy:
        best = max(any_rapid, key=len)
        return best, "rule3_rapid_only", "cross", normalize_text(best)

    return None, "conflict", "none", ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows")
    args = parser.parse_args()

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")

    boxes = json.loads(BOXES_PATH.read_text(encoding="utf-8"))
    if not boxes:
        raise RuntimeError(f"No boxes found in {BOXES_PATH}")

    max_row = max(b["row"] for b in boxes)
    max_col = max(b["col"] for b in boxes)
    row_limit = max_row if args.max_rows <= 0 else min(args.max_rows, max_row)

    table = {(b["row"], b["col"]): b for b in boxes if b["row"] <= row_limit}

    reader = easyocr.Reader(["en"], gpu=True)
    rapid = RapidOCR(
        det_use_cuda=True,
        det_model_path="",
        cls_use_cuda=True,
        cls_model_path="",
        rec_use_cuda=True,
        rec_model_path="",
    )
    print("EasyOCR device: GPU (forced)")
    print("RapidOCR device: GPU (forced)")
    print(f"rows={row_limit}, cols={max_col}")

    rows = []
    conflicts = []
    total_conflicts = 0

    for r in range(1, row_limit + 1):
        row_out = {"row": r, "cells": {}}
        row_conflicts = []
        for c in range(1, max_col + 1):
            box = table.get((r, c))
            if not box:
                continue
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            crop = image[y1:y2, x1:x2]
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
            for method_name, fn in METHODS.items():
                stage_img = fn(crop)
                easy_v = easy_text(reader, stage_img)
                rapid_v = rapid_text(rapid, stage_img)
                method_results.append(
                    {
                        "method": method_name,
                        "easy": easy_v,
                        "rapid": rapid_v,
                        "easy_norm": normalize_text(easy_v),
                        "rapid_norm": normalize_text(rapid_v),
                    }
                )

            # Only ""+"" across all stages is true empty. "0" is non-empty by design.
            if all((m["easy"] == "" and m["rapid"] == "") for m in method_results):
                print(f"cell r{r:03d} c{c:02d}: empty (both engines all stages)")
                continue

            selected, reason, used_stage, selected_norm = decide(method_results)
            unique_norms = len(
                set([m["easy_norm"] for m in method_results if m["easy_norm"]])
                | set([m["rapid_norm"] for m in method_results if m["rapid_norm"]])
            )

            if selected is not None:
                row_out["cells"][str(c)] = selected
                print(
                    f"cell r{r:03d} c{c:02d}: accepted "
                    f"({reason}, stage={used_stage}, unique_norms={unique_norms}, norm={selected_norm})"
                )
            else:
                row_conflicts.append(
                    {
                        "col": c,
                        "box": box,
                        "methods": method_results,
                    }
                )
                total_conflicts += 1
                non_empty = sum(
                    1 for m in method_results for v in (m["easy_norm"], m["rapid_norm"]) if v
                )
                print(
                    f"cell r{r:03d} c{c:02d}: conflict "
                    f"(outputs={non_empty}, unique_norms={unique_norms}, total_conflicts={total_conflicts})"
                )

        rows.append(row_out)
        if row_conflicts:
            conflicts.append({"row": r, "conflicts": row_conflicts})
        print(
            f"row progress {r}/{row_limit} "
            f"(row_conflicts={len(row_conflicts)}, total_conflicts={total_conflicts})"
        )

    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    OUT_CONFLICTS.parent.mkdir(parents=True, exist_ok=True)
    OUT_RAW.write_text(
        json.dumps(
            {
                "image": str(IMAGE_PATH),
                "rows": row_limit,
                "cols": max_col,
                "data": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    OUT_CONFLICTS.write_text(json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_RAW)
    print(OUT_CONFLICTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
