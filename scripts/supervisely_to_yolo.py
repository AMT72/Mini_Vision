"""
Task 1: Supervisely JSON -> YOLO instance segmentation labels.

Reads one JSON per image from the Supervisely `ann/` dir, keeps only the 8
target damage classes, normalizes polygon points by image width/height, and
writes one `.txt` per image into an output labels dir. Images with no damage
objects produce an empty `.txt` (kept as background per YOLO best practice).
"""
from __future__ import annotations

import json
from pathlib import Path

# Project root = parent of this scripts/ dir
ROOT = Path(__file__).resolve().parent.parent

ANN_DIR = ROOT / "data" / "Car damages dataset" / "File1" / "ann"
IMG_DIR = ROOT / "data" / "Car damages dataset" / "File1" / "img"
OUT_LABELS_DIR = ROOT / "data" / "damages_only_dataset" / "_staging" / "labels"
OUT_IMAGES_DIR = ROOT / "data" / "damages_only_dataset" / "_staging" / "images"

# Class title -> YOLO class id (0..7). Exact match against Supervisely classTitle.
CLASS_MAP: dict[str, int] = {
    "Missing part": 0,
    "Broken part": 1,
    "Scratch":     2,
    "Cracked":     3,
    "Dent":        4,
    "Flaking":     5,
    "Paint chip":  6,
    "Corrosion":   7,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_image_for_json(json_path: Path) -> Path | None:
    """Supervisely names JSONs as `<image_filename>.json` (e.g. `foo.png.json`)."""
    stem = json_path.name[:-5] if json_path.name.endswith(".json") else json_path.stem
    candidate = IMG_DIR / stem
    if candidate.exists():
        return candidate
    # Fallback: try swapping/adding common extensions if the exact name is missing.
    base = Path(stem).stem
    for ext in IMG_EXTS:
        p = IMG_DIR / f"{base}{ext}"
        if p.exists():
            return p
    return None


def convert_one(json_path: Path) -> tuple[int, int]:
    """Returns (n_damage_objects_written, 1_if_image_paired_else_0)."""
    img_path = find_image_for_json(json_path)
    if img_path is None:
        return (-1, 0)  # sentinel: missing image

    data = json.loads(json_path.read_text(encoding="utf-8"))
    size = data.get("size") or {}
    W, H = size.get("width"), size.get("height")
    if not W or not H:
        return (-1, 0)

    lines: list[str] = []
    for obj in data.get("objects", []):
        title = obj.get("classTitle")
        cls_id = CLASS_MAP.get(title)
        if cls_id is None:
            continue  # ignore non-damage classes (car parts, etc.)
        if obj.get("geometryType") != "polygon":
            continue
        exterior = (obj.get("points") or {}).get("exterior") or []
        if len(exterior) < 3:
            continue  # YOLO seg needs >=3 points to form a polygon

        # Normalize: x/W, y/H -> [0,1]; clip to guard against off-by-one labels.
        coords: list[str] = []
        for x, y in exterior:
            nx = min(max(x / W, 0.0), 1.0)
            ny = min(max(y / H, 0.0), 1.0)
            coords.append(f"{nx:.6f} {ny:.6f}")
        lines.append(f"{cls_id} " + " ".join(coords))

    # Mirror the image filename's stem for the label file
    out_label = OUT_LABELS_DIR / (img_path.stem + ".txt")
    out_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # Stage the image alongside (copy so original dataset stays intact)
    out_img = OUT_IMAGES_DIR / img_path.name
    if not out_img.exists():
        out_img.write_bytes(img_path.read_bytes())

    return (len(lines), 1)


def main() -> None:
    OUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(ANN_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON annotation files in {ANN_DIR}")

    processed = 0
    empty_bg = 0
    skipped = 0
    total_objs = 0

    for jp in json_files:
        n_objs, paired = convert_one(jp)
        if paired == 0:
            skipped += 1
            print(f"  [skip] {jp.name}: no matching image or invalid size")
            continue
        processed += 1
        total_objs += n_objs
        if n_objs == 0:
            empty_bg += 1

    print("-" * 60)
    print(f"Processed: {processed} images")
    print(f"  - with damage labels:   {processed - empty_bg}")
    print(f"  - empty background:     {empty_bg}")
    print(f"Skipped (no image/size):  {skipped}")
    print(f"Total damage polygons written: {total_objs}")
    print(f"Labels -> {OUT_LABELS_DIR}")
    print(f"Images -> {OUT_IMAGES_DIR}")


if __name__ == "__main__":
    main()
