import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "docs" / "sources" / "monster_2.jpg"
CELLS_PATH = ROOT / ".work" / "ocr" / "monster_2_cells.json"
OUT_PATH = ROOT / ".work" / "ocr" / "monster_2_boxes_preview.png"


def main() -> int:
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")

    payload = json.loads(CELLS_PATH.read_text(encoding="utf-8"))
    xs = payload["vertical_lines"]
    ys = payload["horizontal_lines"]

    preview = image.copy()
    for r in range(len(ys) - 1):
        y1 = int(ys[r])
        y2 = int(ys[r + 1])
        for c in range(len(xs) - 1):
            x1 = int(xs[c])
            x2 = int(xs[c + 1])
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 1)
            if r < 8 and c < 8:
                cv2.putText(
                    preview,
                    f"r{r+1}c{c+1}",
                    (x1 + 3, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PATH), preview)
    print(OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
