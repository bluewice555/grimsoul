import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "docs" / "sources" / "monster_2.jpg"
LINES_PATH = ROOT / ".work" / "ocr" / "monster_2_cells.json"
OUT_CELLS = ROOT / ".work" / "ocr" / "monster_2_cells_boxes.json"
OUT_PREVIEW = ROOT / ".work" / "ocr" / "monster_2_boxes_merged_preview.png"


def line_strength(hline_img: np.ndarray, y: int, x1: int, x2: int, band: int = 2) -> float:
    y1 = max(0, y - band)
    y2 = min(hline_img.shape[0], y + band + 1)
    x1 = max(0, x1)
    x2 = min(hline_img.shape[1], x2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = hline_img[y1:y2, x1:x2]
    return float(np.mean(roi > 0))


def vline_strength(vline_img: np.ndarray, x: int, y1: int, y2: int, band: int = 2) -> float:
    x1 = max(0, x - band)
    x2 = min(vline_img.shape[1], x + band + 1)
    y1 = max(0, y1)
    y2 = min(vline_img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = vline_img[y1:y2, x1:x2]
    return float(np.mean(roi > 0))


def vline_grad_strength(grad_x: np.ndarray, x: int, y1: int, y2: int, band: int = 2) -> float:
    x1 = max(0, x - band)
    x2 = min(grad_x.shape[1], x + band + 1)
    y1 = max(0, y1)
    y2 = min(grad_x.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = grad_x[y1:y2, x1:x2]
    return float(np.mean(roi))


class DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        pa = self.find(a)
        pb = self.find(b)
        if pa == pb:
            return
        if self.r[pa] < self.r[pb]:
            pa, pb = pb, pa
        self.p[pb] = pa
        if self.r[pa] == self.r[pb]:
            self.r[pa] += 1


def main() -> int:
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")

    payload = json.loads(LINES_PATH.read_text(encoding="utf-8"))
    xs_raw = payload["vertical_lines"]
    ys = payload["horizontal_lines"]
    xs = [xs_raw[0]]
    for i in range(1, len(xs_raw)):
        if xs_raw[i] - xs[-1] >= 80 or i == len(xs_raw) - 1:
            xs.append(xs_raw[i])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (3, 3), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, image.shape[1] // 45), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, image.shape[0] // 45)))
    hline = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vline = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=1)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_x = np.abs(grad_x)

    n_rows = len(ys) - 1
    n_cols = len(xs) - 1
    n = n_rows * n_cols
    dsu = DSU(n)

    def idx(r: int, c: int) -> int:
        return r * n_cols + c

    # Merge vertically if horizontal boundary is weak in this column span.
    # Header zone (first two rows) uses a different threshold because of dark background.
    for r in range(n_rows - 1):
        # r1-r2 merge is disabled; header merge starts from r2-r3.
        if r == 0 or r > 2:
            continue
        y = ys[r + 1]
        for c in range(n_cols):
            s = line_strength(hline, y, xs[c] + 2, xs[c + 1] - 2)
            th = 0.38
            if s < th:
                dsu.union(idx(r, c), idx(r + 1, c))

    # Horizontal merges are enabled only in header zone to avoid chain-merging body rows.
    for r in range(n_rows):
        # Do not horizontal-merge row 1.
        if r == 0 or r > 2:
            continue
        strengths = []
        for b in range(1, n_cols):
            # Prefer gradient-based strength for gray separator lines.
            strengths.append(vline_grad_strength(grad_x, xs[b], ys[r] + 2, ys[r + 1] - 2))
        weak_th = 10.0
        strong_th = 20.0
        for c in range(n_cols - 1):
            s = strengths[c]
            left_ok = True if c == 0 else strengths[c - 1] >= strong_th
            right_ok = True if c == n_cols - 2 else strengths[c + 1] >= strong_th
            # Only merge on isolated weak boundary, avoiding chain merges across whole header rows.
            if s < weak_th and left_ok and right_ok:
                dsu.union(idx(r, c), idx(r, c + 1))

    groups = {}
    for r in range(n_rows):
        for c in range(n_cols):
            root = dsu.find(idx(r, c))
            groups.setdefault(root, []).append((r, c))

    cells = []
    for members in groups.values():
        rs = [m[0] for m in members]
        cs = [m[1] for m in members]
        r0 = min(rs)
        r1 = max(rs)
        c0 = min(cs)
        c1 = max(cs)
        row_span = r1 - r0 + 1
        col_span = c1 - c0 + 1
        # Guardrail: if a chain merge is too wide in body rows, keep vertical merges but split by columns.
        # Header rows (top 3 index range) are allowed to keep horizontal merges.
        if r0 > 2 and row_span <= 3 and col_span > 3:
            for cc in range(c0, c1 + 1):
                cells.append(
                    {
                        "row_start": r0 + 1,
                        "row_end": r1 + 1,
                        "col_start": cc + 1,
                        "col_end": cc + 1,
                        "x1": int(xs[cc]),
                        "x2": int(xs[cc + 1]),
                        "y1": int(ys[r0]),
                        "y2": int(ys[r1 + 1]),
                        "cell_count": row_span,
                    }
                )
        elif row_span > 3 or len(members) > 6:
            for rr, cc in members:
                cells.append(
                    {
                        "row_start": rr + 1,
                        "row_end": rr + 1,
                        "col_start": cc + 1,
                        "col_end": cc + 1,
                        "x1": int(xs[cc]),
                        "x2": int(xs[cc + 1]),
                        "y1": int(ys[rr]),
                        "y2": int(ys[rr + 1]),
                        "cell_count": 1,
                    }
                )
        else:
            cells.append(
                {
                    "row_start": r0 + 1,
                    "row_end": r1 + 1,
                    "col_start": c0 + 1,
                    "col_end": c1 + 1,
                    "x1": int(xs[c0]),
                    "x2": int(xs[c1 + 1]),
                    "y1": int(ys[r0]),
                    "y2": int(ys[r1 + 1]),
                    "cell_count": len(members),
                }
            )

    cells.sort(key=lambda x: (x["row_start"], x["col_start"], x["row_end"], x["col_end"]))

    OUT_CELLS.parent.mkdir(parents=True, exist_ok=True)
    OUT_CELLS.write_text(json.dumps(cells, ensure_ascii=False, indent=2), encoding="utf-8")

    preview = image.copy()
    for cell in cells:
        x1, y1, x2, y2 = cell["x1"], cell["y1"], cell["x2"], cell["y2"]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 1)
        if cell["row_end"] > cell["row_start"] or cell["col_end"] > cell["col_start"]:
            cv2.putText(
                preview,
                f"r{cell['row_start']}-{cell['row_end']} c{cell['col_start']}-{cell['col_end']}",
                (x1 + 3, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(OUT_PREVIEW), preview)
    merged = [c for c in cells if c["cell_count"] > 1]
    print(f"boxes={len(cells)} merged_boxes={len(merged)}")
    print(OUT_CELLS)
    print(OUT_PREVIEW)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
