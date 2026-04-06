import json
import re
from pathlib import Path

import cv2
import easyocr
import numpy as np
from rapidocr_onnxruntime import RapidOCR


ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "docs" / "sources" / "monster_2.jpg"
BOXES_PATH = ROOT / ".work" / "ocr" / "monster_2_cells_boxes.json"
OUT_JSON = ROOT / "docs" / "dataset" / "monster_2.json"
OUT_CONFLICTS = ROOT / ".work" / "ocr" / "monster_2_conflicts.json"
OUT_LOG = ROOT / ".work" / "ocr" / "monster_2_ocr_log.txt"


def clean_text(text: str) -> str:
    text = text.replace("|", "I").replace("’", "'").replace("`", "'")
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ;")


def normalize_text(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def normalize_mark(text: str):
    t = text.strip().lower()
    if not t:
        return None
    check_tokens = ["✓", "✔", "☑", "v", "vi", "yes", "true"]
    cross_tokens = ["✗", "✘", "x", "no", "false"]
    if any(tok in t for tok in check_tokens):
        return "V"
    if any(tok in t for tok in cross_tokens):
        return "X"
    if t in {"1", "l"}:
        return "V"
    return None


def detect_mark_shape(image):
    # OpenCV-only mark detection (no OCR): tries to classify checkbox-like strokes as V/X.
    if image is None or image.size == 0:
        return None
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(big, (3, 3), 0)
    bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    h, w = bw.shape[:2]
    mx = max(2, w // 12)
    my = max(2, h // 12)
    roi = bw[my : h - my, mx : w - mx]
    if roi.size == 0:
        return None

    ink_ratio = float((roi > 0).mean())
    if ink_ratio < 0.01 or ink_ratio > 0.40:
        return None

    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180,
        threshold=18,
        minLineLength=max(10, min(roi.shape) // 4),
        maxLineGap=6,
    )
    if lines is None:
        return None

    pos_len = 0.0
    neg_len = 0.0
    for ln in lines[:, 0, :]:
        x1, y1, x2, y2 = ln
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dx) < 1e-6:
            continue
        length = float((dx * dx + dy * dy) ** 0.5)
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        # keep only diagonal-ish segments
        if angle < 18 or angle > 165:
            continue
        if dy * dx > 0:
            pos_len += length
        else:
            neg_len += length

    total = pos_len + neg_len
    if total < 20:
        return None
    if pos_len > 10 and neg_len > 10:
        ratio = min(pos_len, neg_len) / max(pos_len, neg_len)
        return "X" if ratio >= 0.68 else "V"
    if max(pos_len, neg_len) > 18:
        return "V"
    return None


def preprocess(crop):
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 5, 35, 35)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


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


def easy_text(reader, image):
    return clean_text(" ".join(reader.readtext(image, detail=0, paragraph=False)))


def rapid_text(engine, image):
    results, _ = engine(image)
    if not results:
        return ""
    return clean_text(" ".join(item[1] for item in results))


def decide_value(easy: str, rapid: str):
    em = normalize_mark(easy)
    rm = normalize_mark(rapid)
    if em and rm and em == rm:
        return em, "mark_agree"
    if em and not rapid:
        return em, "mark_easy_only"
    if rm and not easy:
        return rm, "mark_rapid_only"

    en = normalize_text(easy)
    rn = normalize_text(rapid)
    if not en and not rn:
        return None, "empty"
    if en and rn and en == rn:
        return (easy if len(easy) >= len(rapid) else rapid), "agree"
    if en and not rn:
        return easy, "easy_only"
    if rn and not en:
        return rapid, "rapid_only"
    return None, "conflict"


def pick_overlap(candidates):
    pool = {}
    for c in candidates:
        if c["easy_norm"]:
            pool.setdefault(c["easy_norm"], []).append(c["easy"])
        if c["rapid_norm"]:
            pool.setdefault(c["rapid_norm"], []).append(c["rapid"])
    agree = [(k, vals) for k, vals in pool.items() if len(vals) >= 2]
    if agree:
        agree.sort(key=lambda kv: (len(kv[1]), len(kv[0])), reverse=True)
        return max(agree[0][1], key=len), agree[0][0], len(agree[0][1])
    return None, "", 0


def run_dual_ensemble(reader, rapid, crop):
    candidates = []
    stages = [("raw", crop)] + [(name, fn(crop)) for name, fn in METHODS.items()]

    # First pass: OpenCV shape vote for check/cross marks.
    shape_votes = []
    for stage_name, img in stages:
        mk = detect_mark_shape(img)
        if mk:
            shape_votes.append((stage_name, mk))
            same = [v for _, v in shape_votes if v == mk]
            if len(same) >= 2:
                return mk, "mark_shape_vote", stage_name, mk, mk

    for stage_name, img in stages:
        e = easy_text(reader, img)
        r = rapid_text(rapid, img)
        em = normalize_mark(e)
        rm = normalize_mark(r)
        if em and rm and em == rm:
            return em, "mark_agree", stage_name, e, r
        candidates.append(
            {
                "stage": stage_name,
                "easy": e,
                "rapid": r,
                "easy_norm": normalize_text(e),
                "rapid_norm": normalize_text(r),
            }
        )
        chosen, chosen_norm, support = pick_overlap(candidates)
        if chosen is not None:
            return chosen, f"agree_support_{support}", stage_name, e, r
    # fallback to last single-stage decision
    if not candidates:
        return None, "empty", "none", "", ""
    last = candidates[-1]
    v, reason = decide_value(last["easy"], last["rapid"])
    return v, reason, last["stage"], last["easy"], last["rapid"]


def find_cell(cells, row, col):
    for c in cells:
        if c["row_start"] <= row <= c["row_end"] and c["col_start"] <= col <= c["col_end"]:
            return c
    return None


def crop_cell(image, cell):
    x1, x2 = int(cell["x1"]), int(cell["x2"])
    y1, y2 = int(cell["y1"]), int(cell["y2"])
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    ix = min(4, max(1, w // 30))
    iy = min(3, max(1, h // 20))
    return crop[iy : max(iy + 1, h - iy), ix : max(ix + 1, w - ix)]


def main() -> int:
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")

    cells = json.loads(BOXES_PATH.read_text(encoding="utf-8"))
    max_row = max(c["row_end"] for c in cells)
    max_col = max(c["col_end"] for c in cells)

    reader = easyocr.Reader(["en"], gpu=True)
    rapid = RapidOCR(
        det_use_cuda=True,
        det_model_path=None,
        cls_use_cuda=True,
        cls_model_path=None,
        rec_use_cuda=True,
        rec_model_path=None,
    )

    logs = []
    conflicts = []

    # Build column schema from row1/row2
    schema = {}
    for col in range(1, max_col + 1):
        h1 = find_cell(cells, 1, col)
        h2 = find_cell(cells, 2, col)
        parent = ""
        child = ""
        if h1:
            c1 = crop_cell(image, h1)
            parent, _, _, _, _ = run_dual_ensemble(reader, rapid, c1)
            parent = parent or ""
        if h2 and h2["row_start"] == 2 and h2["row_end"] == 2:
            c2 = crop_cell(image, h2)
            child, _, _, _, _ = run_dual_ensemble(reader, rapid, c2)
            child = child or ""
        schema[col] = {"parent": parent, "child": child}

    rows_out = []
    for row in range(3, max_row + 1):
        row_obj = {"row": row}
        for col in range(1, max_col + 1):
            cell = find_cell(cells, row, col)
            if not cell:
                continue
            crop = crop_cell(image, cell)
            if crop.size == 0:
                continue
            decided, reason, used_stage, easy, rapid_v = run_dual_ensemble(reader, rapid, crop)

            key_parent = schema[col]["parent"] or f"col_{col}"
            key_child = schema[col]["child"]
            log_line = f"r{row} c{col} -> {reason} stage={used_stage} | easy='{easy}' rapid='{rapid_v}'"
            logs.append(log_line)
            print(log_line)

            if reason == "conflict":
                conflicts.append(
                    {
                        "row": row,
                        "col": col,
                        "parent": key_parent,
                        "child": key_child,
                        "easyocr": easy,
                        "rapidocr": rapid_v,
                    }
                )
                continue
            if decided is None:
                continue

            if key_child:
                row_obj.setdefault(key_parent, {})
                row_obj[key_parent][key_child] = decided
            else:
                row_obj[key_parent] = decided

        rows_out.append(row_obj)
        print(f"row progress {row}/{max_row} conflicts={len(conflicts)}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_CONFLICTS.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows_out, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_CONFLICTS.write_text(json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_LOG.write_text("\n".join(logs), encoding="utf-8")
    print(OUT_JSON)
    print(OUT_CONFLICTS)
    print(OUT_LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
