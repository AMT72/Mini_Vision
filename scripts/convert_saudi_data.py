"""
Convert our Saudi annotated data (COCO segmentation from Roboflow export)
to YOLO instance-seg format, matching CarDD's 6-class taxonomy exactly.

Fixes applied:
  - Strips the dummy "Mini-Vision" category (id=0, 0 annotations).
  - Normalises class names to lowercase to match CarDD:
        Dent -> dent, Scratch -> scratch
  - Converts RLE masks to polygons (via cv2.findContours on the decoded bitmap).
  - Drops tiny annotations (<100 px area) that are annotation noise.
  - Splits 80/10/10 into train/val/test (the Roboflow export dumped everything
    into train/).

Output:  data/saudi_dataset/{images,labels}/{train,val,test}/
         data/saudi_dataset/saudi_dataset.yaml
"""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data" / "OUR_ANNOTATIE_DATA" / "_extracted" / "train"
OUT_DIR = ROOT / "data" / "saudi_dataset"

SEED = 42
SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}
MIN_AREA = 100  # drop annotations smaller than this (px^2)

# CarDD's canonical class names (lowercase, 0-indexed).
FINAL_NAMES: dict[int, str] = {
    0: "dent",
    1: "scratch",
    2: "crack",
    3: "glass shatter",
    4: "lamp broken",
    5: "tire flat",
}

# Map from our Roboflow category names -> final YOLO class id.
# Case-insensitive lookup built at runtime.
NAME_TO_ID: dict[str, int] = {name: cid for cid, name in FINAL_NAMES.items()}


def _rle_to_polygon(rle: dict, h: int, w: int) -> list[list[float]] | None:
    """Decode a COCO RLE mask to the largest polygon contour.
    Returns list of [x, y] pairs or None if decoding fails."""
    counts = rle["counts"]
    if isinstance(counts, str):
        # Compressed RLE — decode byte-by-byte
        import pycocotools.mask as mask_util  # type: ignore
        binary = mask_util.decode(rle)
    else:
        # Uncompressed RLE (list of ints)
        binary = np.zeros(h * w, dtype=np.uint8)
        pos = 0
        for i, cnt in enumerate(counts):
            if i % 2 == 1:  # odd indices are "filled" runs
                binary[pos : pos + cnt] = 1
            pos += cnt
        binary = binary.reshape((h, w), order="F")  # COCO uses column-major

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Take the largest contour
    biggest = max(contours, key=cv2.contourArea)
    if len(biggest) < 3:
        return None
    # Squeeze to (N, 2)
    pts = biggest.squeeze().tolist()
    if isinstance(pts[0], int):
        # Only 1 point somehow
        return None
    return pts


def load_coco(ann_path: Path) -> tuple[dict, list[dict], list[dict]]:
    """Load COCO JSON, return (categories_by_id, images, annotations)."""
    with open(ann_path) as f:
        data = json.load(f)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    return cats, data["images"], data["annotations"]


def convert() -> None:
    ann_path = SRC_DIR / "_annotations.coco.json"
    if not ann_path.exists():
        raise SystemExit(
            f"Annotation file not found: {ann_path}\n"
            "Run: unzip Mini-Vision.coco-segmentation.zip -d _extracted/"
        )

    cats, images, annotations = load_coco(ann_path)

    # Build category id -> final YOLO id mapping
    cat_remap: dict[int, int | None] = {}
    for cat_id, cat_name in cats.items():
        # Case-insensitive match
        final_id = NAME_TO_ID.get(cat_name.lower())
        cat_remap[cat_id] = final_id
        status = f"-> {final_id} ({FINAL_NAMES[final_id]})" if final_id is not None else "-> SKIP"
        print(f"  Category {cat_id} '{cat_name}' {status}")

    # Index: image_id -> image info
    img_lookup: dict[int, dict] = {img["id"]: img for img in images}

    # Group annotations by image
    anns_by_image: dict[int, list[dict]] = {}
    for a in annotations:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    # ── Fresh output dirs ──
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    # ── Convert each image ──
    converted: list[tuple[Path, str]] = []  # (image_path, yolo_label_text)
    stats = Counter()

    for img_info in images:
        img_id = img_info["id"]
        fname = img_info["file_name"]
        W, H = img_info["width"], img_info["height"]
        img_path = SRC_DIR / fname

        if not img_path.exists():
            stats["img_missing"] += 1
            continue

        img_anns = anns_by_image.get(img_id, [])
        label_lines: list[str] = []

        for a in img_anns:
            final_id = cat_remap.get(a["category_id"])
            if final_id is None:
                stats["skipped_class"] += 1
                continue

            # Skip tiny annotations
            if a.get("area", 0) < MIN_AREA:
                stats["skipped_tiny"] += 1
                continue

            # Get polygon points
            seg = a["segmentation"]
            if isinstance(seg, dict):
                # RLE -> polygon
                pts = _rle_to_polygon(seg, H, W)
                if pts is None:
                    stats["rle_fail"] += 1
                    continue
                stats["rle_converted"] += 1
                # pts is [[x, y], ...]
                flat_coords = []
                for x, y in pts:
                    flat_coords.extend([x / W, y / H])
            elif isinstance(seg, list) and len(seg) > 0:
                # Standard COCO polygon: [[x1,y1,x2,y2,...], ...]
                # Take the largest polygon if multi-part
                best_poly = max(seg, key=len)
                if len(best_poly) < 6:  # need at least 3 points
                    stats["skipped_small_poly"] += 1
                    continue
                # Normalise to [0, 1]
                flat_coords = []
                for i in range(0, len(best_poly), 2):
                    flat_coords.append(best_poly[i] / W)
                    flat_coords.append(best_poly[i + 1] / H)
                stats["polygon_ok"] += 1
            else:
                stats["skipped_empty_seg"] += 1
                continue

            # Clamp to [0, 1]
            flat_coords = [max(0.0, min(1.0, c)) for c in flat_coords]

            coords_str = " ".join(f"{c:.6f}" for c in flat_coords)
            label_lines.append(f"{final_id} {coords_str}")

        if label_lines:
            converted.append((img_path, "\n".join(label_lines) + "\n"))
            stats["images_with_labels"] += 1
        else:
            # Keep image with empty label (background) only if it had annotations
            # that were all filtered out
            if img_anns:
                converted.append((img_path, ""))
                stats["images_all_filtered"] += 1

    print(f"\n  Conversion stats: {dict(stats)}")
    print(f"  Total images to split: {len(converted)}")

    # ── Split 80/10/10 ──
    rng = random.Random(SEED)
    rng.shuffle(converted)
    n = len(converted)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])

    split_data = {
        "train": converted[:n_train],
        "val": converted[n_train : n_train + n_val],
        "test": converted[n_train + n_val :],
    }

    for split_name, items in split_data.items():
        img_dir = OUT_DIR / "images" / split_name
        lbl_dir = OUT_DIR / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, label_text in items:
            # Copy image
            shutil.copy2(img_path, img_dir / img_path.name)
            # Write label
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            lbl_path.write_text(label_text, encoding="utf-8")

        print(f"  {split_name}: {len(items)} images")

    # ── Write YAML ──
    yaml_path = OUT_DIR / "saudi_dataset.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in FINAL_NAMES.items())
    yaml_path.write_text(
        "# Saudi annotated damage dataset — YOLO seg format (auto-generated)\n"
        f"path: {OUT_DIR.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"nc: {len(FINAL_NAMES)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    print(f"\n  YAML: {yaml_path}")

    # ── Per-class histogram ──
    print("\n  Per-split class counts:")
    for split_name in ["train", "val", "test"]:
        lbl_dir = OUT_DIR / "labels" / split_name
        class_counts: Counter = Counter()
        for lbl in lbl_dir.glob("*.txt"):
            for ln in lbl.read_text().splitlines():
                if ln.strip():
                    class_counts[int(ln.split()[0])] += 1
        pretty = ", ".join(f"{FINAL_NAMES[i]}:{class_counts.get(i,0)}" for i in range(6))
        print(f"    {split_name}: {pretty}")


if __name__ == "__main__":
    print("Converting Saudi COCO annotations -> YOLO seg format\n")
    convert()
    print("\nDone.")
