import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "docs" / "sources" / "weapons.jpg"
OUTPUT_JSON = ROOT / ".work" / "ocr" / "weapon_cells.json"
OUTPUT_PREVIEW = ROOT / ".work" / "ocr" / "weapon_cells_preview.png"


def cluster_indices(indices: np.ndarray, gap: int = 12) -> list[int]:
    if len(indices) == 0:
        return []
    clusters = []
    start = int(indices[0])
    prev = int(indices[0])
    for x in indices[1:]:
        x = int(x)
        if x - prev > gap:
            clusters.append((start, prev))
            start = x
        prev = x
    clusters.append((start, prev))
    return [int(round((a + b) / 2)) for a, b in clusters]


def detect_lines(binary_inv: np.ndarray, axis: int) -> list[int]:
    proj = np.sum(binary_inv > 0, axis=axis)
    threshold = max(10, int(proj.max() * 0.35))
    idx = np.where(proj >= threshold)[0]
    return cluster_indices(idx, gap=10)


def main() -> int:
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")
    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 45)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 45), 1))

    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    xs = detect_lines(vertical, axis=0)
    ys = detect_lines(horizontal, axis=1)

    if not xs or xs[0] > 5:
        xs = [0] + xs
    if not ys or ys[0] > 5:
        ys = [0] + ys
    if xs[-1] < w - 5:
        xs.append(w)
    if ys[-1] < h - 5:
        ys.append(h)

    payload = {
        "image": str(IMAGE_PATH),
        "width": w,
        "height": h,
        "vertical_lines": xs,
        "horizontal_lines": ys,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    preview = image.copy()
    for x in xs:
        cv2.line(preview, (x, 0), (x, h - 1), (0, 255, 0), 2)
    for y in ys:
        cv2.line(preview, (0, y), (w - 1, y), (0, 200, 255), 2)
    cv2.imwrite(str(OUTPUT_PREVIEW), preview)

    print(f"vertical_lines={len(xs)}")
    print(f"horizontal_lines={len(ys)}")
    print(OUTPUT_JSON)
    print(OUTPUT_PREVIEW)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

