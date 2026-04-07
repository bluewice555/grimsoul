import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "docs" / "sources" / "weapons.jpg"
LINES_PATH = ROOT / ".work" / "ocr" / "weapon_cells.json"
OUT_BOXES = ROOT / ".work" / "ocr" / "weapon_cells_boxes.json"
OUT_PREVIEW = ROOT / ".work" / "ocr" / "weapon_cells_boxes_preview.png"


def main() -> int:
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")

    payload = json.loads(LINES_PATH.read_text(encoding="utf-8-sig"))
    xs = payload["vertical_lines"]
    ys = payload["horizontal_lines"]

    boxes = []
    for r in range(len(ys) - 1):
        y1 = int(ys[r])
        y2 = int(ys[r + 1])
        for c in range(len(xs) - 1):
            x1 = int(xs[c])
            x2 = int(xs[c + 1])
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            boxes.append(
                {
                    "row": r + 1,
                    "col": c + 1,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "w": w,
                    "h": h,
                    "is_header": r < 2,
                    "is_id_col": c == 0,
                }
            )

    OUT_BOXES.parent.mkdir(parents=True, exist_ok=True)
    OUT_BOXES.write_text(json.dumps(boxes, ensure_ascii=False, indent=2), encoding="utf-8")

    preview = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imwrite(str(OUT_PREVIEW), preview)

    print(f"rows={len(ys) - 1} cols={len(xs) - 1} boxes={len(boxes)}")
    print(OUT_BOXES)
    print(OUT_PREVIEW)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

