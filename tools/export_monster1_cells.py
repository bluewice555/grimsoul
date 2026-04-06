import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "docs" / "sources" / "monster_1.jpg"
CELLS_PATH = ROOT / ".work" / "ocr" / "monster_1_cells.json"
OUT_DIR = ROOT / ".work" / "ocr" / "monster_1_cells_raw"
OUT_MANIFEST = ROOT / ".work" / "ocr" / "monster_1_cells_raw.json"

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


def main() -> int:
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")

    payload = json.loads(CELLS_PATH.read_text(encoding="utf-8"))
    xs = payload["vertical_lines"]
    ys = payload["horizontal_lines"]
    total_rows = 132

    OUT_DIR.mkdir(parents=True, exist_ok=True)
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

            filename = f"r{row_index + 1:03d}_c{col_index + 1:02d}.png"
            path = OUT_DIR / filename
            cv2.imwrite(str(path), crop)

            manifest.append(
                {
                    "row": row_index + 1,
                    "col": col_index + 1,
                    "field": field,
                    "x1": int(x1),
                    "x2": int(x2),
                    "y1": int(top),
                    "y2": int(bottom),
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                }
            )

        if (row_index + 1) % 20 == 0:
            print(f"export progress {row_index + 1}/{total_rows}")

    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_DIR)
    print(OUT_MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
