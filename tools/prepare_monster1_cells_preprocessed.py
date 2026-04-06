import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "docs" / "sources" / "monster_1.jpg"
CELLS_PATH = ROOT / ".work" / "ocr" / "monster_1_cells.json"
PREPROC_DIR = ROOT / ".work" / "ocr" / "monster_1_cells_preprocessed"
PREPROC_MANIFEST = ROOT / ".work" / "ocr" / "monster_1_cells_preprocessed.json"

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


def preprocess_cell_for_ocr(crop):
    enlarged = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = enlarged if len(enlarged.shape) == 2 else cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    denoise = cv2.fastNlMeansDenoising(contrast, None, 10, 7, 21)
    sharp = cv2.addWeighted(denoise, 1.35, cv2.GaussianBlur(denoise, (0, 0), 1.2), -0.35, 0)
    return cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
    )


def main() -> int:
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")

    payload = json.loads(CELLS_PATH.read_text(encoding="utf-8"))
    xs = payload["vertical_lines"]
    ys = payload["horizontal_lines"]
    total_rows = 132

    PREPROC_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for row_index in range(total_rows):
        top = 49 if row_index == 0 else ys[row_index]
        bottom = ys[row_index + 1]
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

            processed = preprocess_cell_for_ocr(crop)
            filename = f"r{row_index + 1:03d}_c{col_index + 1:02d}.png"
            path = PREPROC_DIR / filename
            cv2.imwrite(str(path), processed)
            manifest.append(
                {
                    "row": row_index + 1,
                    "field": field,
                    "col": col_index + 1,
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                }
            )

        if (row_index + 1) % 20 == 0:
            print(f"preprocess progress {row_index + 1}/{total_rows}")

    PREPROC_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(PREPROC_MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
