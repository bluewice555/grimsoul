import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import easyocr
from rapidocr_onnxruntime import RapidOCR


ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "docs" / "sources" / "monster_1.jpg"
CELLS_PATH = ROOT / ".work" / "ocr" / "monster_1_cells.json"
REPORT_PATH = ROOT / ".work" / "ocr" / "monster_1_id_preprocess_report.json"


def norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def crop_id_cell(image, xs, ys, row_index: int):
    top = 49 if row_index == 0 else ys[row_index]
    bottom = ys[row_index + 1]
    x1 = xs[0]
    x2 = xs[1]
    crop = image[top:bottom, x1:x2]
    h, w = crop.shape[:2]
    inset_x = min(4, max(1, w // 30))
    inset_y = min(3, max(1, h // 20))
    return crop[
        inset_y : max(inset_y + 1, h - inset_y),
        inset_x : max(inset_x + 1, w - inset_x),
    ]


def p0(crop):
    big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)


def p1(crop):
    big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)


def p2(crop):
    big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 5, 35, 35)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def p3(crop):
    big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    den = cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)
    sharp = cv2.addWeighted(den, 1.5, cv2.GaussianBlur(den, (0, 0), 1.0), -0.5, 0)
    _, th = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def p4(crop):
    big = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    th = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10)
    return 255 - th


PIPELINES = {
    "p0_basic_adapt": p0,
    "p1_clahe_adapt": p1,
    "p2_bilateral_otsu": p2,
    "p3_denoise_sharp_otsu": p3,
    "p4_invert_mean_adapt": p4,
}


@dataclass
class Score:
    name: str
    both_nonempty: int
    exact_norm_match: int
    row_details: list


def main() -> int:
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")
    payload = json.loads(CELLS_PATH.read_text(encoding="utf-8"))
    xs = payload["vertical_lines"]
    ys = payload["horizontal_lines"]

    reader = easyocr.Reader(["en"], gpu=True)
    rapid = RapidOCR(
        det_use_cuda=True,
        det_model_path=None,
        cls_use_cuda=True,
        cls_model_path=None,
        rec_use_cuda=True,
        rec_model_path=None,
    )

    scores = []
    for name, fn in PIPELINES.items():
        both_nonempty = 0
        exact_norm_match = 0
        details = []
        for row_index in range(20):
            crop = crop_id_cell(image, xs, ys, row_index)
            proc = fn(crop)
            e = " ".join(reader.readtext(proc, detail=0, paragraph=False)).strip()
            r_res, _ = rapid(proc)
            r = "" if not r_res else " ".join(item[1] for item in r_res).strip()

            en = norm(e)
            rn = norm(r)
            if en and rn:
                both_nonempty += 1
            if en and rn and en == rn:
                exact_norm_match += 1
            details.append({"row": row_index + 1, "easyocr": e, "rapidocr": r, "en": en, "rn": rn})
        score = Score(name=name, both_nonempty=both_nonempty, exact_norm_match=exact_norm_match, row_details=details)
        scores.append(score)
        print(f"{name}: both_nonempty={both_nonempty}/20 exact_norm_match={exact_norm_match}/20")

    scores.sort(key=lambda s: (s.exact_norm_match, s.both_nonempty), reverse=True)
    best = scores[0]
    report = {
        "best": {
            "name": best.name,
            "both_nonempty": best.both_nonempty,
            "exact_norm_match": best.exact_norm_match,
        },
        "ranking": [
            {
                "name": s.name,
                "both_nonempty": s.both_nonempty,
                "exact_norm_match": s.exact_norm_match,
            }
            for s in scores
        ],
        "best_row_details": best.row_details,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
