"""
Build a combined YOLO instance-seg dataset:
  - CarDD (COCO)  -> 6 classes preserved (ids 0..5)
  - Kaggle damages_only_dataset (YOLO) -> contributes class 6 (missing part)

Final taxonomy (0..6):
  0 dent  1 scratch  2 crack  3 glass shatter  4 lamp broken  5 tire flat
  6 missing part

CarDD converted via `ultralytics.data.converter.convert_coco`.
Kaggle images that contain any unmapped damage class are dropped entirely.
"""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

from ultralytics.data.converter import convert_coco

ROOT = Path(__file__).resolve().parent.parent

# Sources
CARDD_DIR  = ROOT / "data" / "CarDD_release" / "CarDD_COCO"
CARDD_ANN  = CARDD_DIR / "annotations"
KAGGLE_DIR = ROOT / "data" / "damages_only_dataset"

# Outputs
OUT       = ROOT / "data" / "combined_dataset"
TMP_CARDD = ROOT / "data" / "_tmp_cardd_yolo"

SPLITS = ["train", "val", "test"]
CARDD_SPLIT_DIR = {"train": "train2017", "val": "val2017", "test": "test2017"}

# Kaggle id -> final id. None means "unmapped" (drop the entire image).
KAGGLE_REMAP: dict[int, int | None] = {
    0: 6,    # Missing part -> missing part
    1: None, # Broken part  -> drop image
    2: 1,    # Scratch      -> scratch
    3: 2,    # Cracked      -> crack
    4: 0,    # Dent         -> dent
    5: None, # Flaking      -> drop image
    6: None, # Paint chip   -> drop image
    7: None, # Corrosion    -> drop image
}

FINAL_NAMES = {
    0: "dent", 1: "scratch", 2: "crack",
    3: "glass shatter", 4: "lamp broken", 5: "tire flat",
    6: "missing part",
}


def fresh_dirs() -> None:
    """Create empty output dirs (wipes prior runs to keep this idempotent)."""
    if OUT.exists():
        shutil.rmtree(OUT)
    if TMP_CARDD.exists():
        shutil.rmtree(TMP_CARDD)
    for split in SPLITS:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)


def convert_cardd() -> None:
    """Run Ultralytics' COCO->YOLO converter on all three CarDD splits."""
    print(f"[CarDD] convert_coco -> {TMP_CARDD}")
    convert_coco(
        labels_dir=str(CARDD_ANN),
        save_dir=str(TMP_CARDD),
        use_segments=True,
        cls91to80=False,
    )


def import_cardd_split(split: str) -> tuple[int, int]:
    """Move converted labels and copy images for one CarDD split.
    Returns (n_images, n_polygons)."""
    # convert_coco names the output dirs after the JSON's basename
    # (without the `instances_` prefix), e.g. `labels/train2017/`.
    src_lbl_dir = TMP_CARDD / "labels" / CARDD_SPLIT_DIR[split]
    src_img_dir = CARDD_DIR / CARDD_SPLIT_DIR[split]
    dst_lbl_dir = OUT / "labels" / split
    dst_img_dir = OUT / "images" / split

    if not src_lbl_dir.exists():
        raise SystemExit(f"convert_coco output missing: {src_lbl_dir}")

    n_imgs = 0
    n_polys = 0
    for lbl in sorted(src_lbl_dir.glob("*.txt")):
        # Match by stem to the image (CarDD images are flat, .jpg).
        stem = lbl.stem
        img = src_img_dir / f"{stem}.jpg"
        if not img.exists():
            # Some COCO entries may not have an image on disk; skip.
            continue
        # Copy label as-is (already 0..5)
        shutil.copy2(lbl, dst_lbl_dir / lbl.name)
        shutil.copy2(img, dst_img_dir / img.name)
        n_imgs += 1
        n_polys += sum(1 for _ in lbl.read_text().splitlines() if _.strip())
    return n_imgs, n_polys


def import_kaggle_split(split: str) -> tuple[int, int, int, int]:
    """Filter + remap Kaggle for one split (hybrid rule).
    Returns (n_kept_images, n_dropped_images, n_polygons, n_soft_kept)."""
    src_lbl_dir = KAGGLE_DIR / "labels" / split
    src_img_dir = KAGGLE_DIR / "images" / split
    dst_lbl_dir = OUT / "labels" / split
    dst_img_dir = OUT / "images" / split

    if not src_lbl_dir.exists():
        print(f"  [warn] Kaggle split {split} has no labels dir; skipping")
        return 0, 0, 0

    kept = dropped = polys = 0
    soft_kept = 0  # images kept under the hybrid relaxation (had Missing part)
    for lbl in sorted(src_lbl_dir.glob("*.txt")):
        lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
        ids = [int(ln.split()[0]) for ln in lines]
        has_unmapped = any(KAGGLE_REMAP.get(c) is None for c in ids)
        has_missing  = 0 in ids  # Kaggle id 0 = Missing part

        # HYBRID rule:
        #   - if the image has unmapped polygons AND no Missing part -> drop.
        #     (We don't accept unlabeled-damage noise just to keep common
        #      classes that CarDD already covers well.)
        #   - if the image has Missing part -> keep it; we'll filter out the
        #     unmapped polygons line-by-line below. Accepts some unlabeled
        #     damage in exchange for the rare class we actually need.
        if has_unmapped and not has_missing:
            dropped += 1
            continue
        if has_unmapped and has_missing:
            soft_kept += 1
        # Lines whose class is unmapped get filtered when building new_lines.

        # Find matching image (try same stem with common image extensions).
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            candidate = src_img_dir / f"{lbl.stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            dropped += 1
            continue

        # Remap and write. Skip any polygon whose class is unmapped
        # (only happens on hybrid-kept images that contain Missing part).
        new_lines: list[str] = []
        for ln in lines:
            parts = ln.split()
            old_id = int(parts[0])
            new_id = KAGGLE_REMAP[old_id]
            if new_id is None:
                continue
            new_lines.append(f"{new_id} " + " ".join(parts[1:]))
        if not new_lines:
            # All polygons were unmapped (shouldn't happen with our rule, but guard)
            dropped += 1
            continue
        out_label = dst_lbl_dir / f"kaggle_{lbl.stem}.txt"
        out_label.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        out_image = dst_img_dir / f"kaggle_{img_path.name}"
        shutil.copy2(img_path, out_image)
        kept += 1
        polys += len(new_lines)
    return kept, dropped, polys, soft_kept


def write_yaml() -> Path:
    yaml_path = OUT / "combined_dataset.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in FINAL_NAMES.items())
    yaml_path.write_text(
        "# Combined CarDD + Kaggle damage seg dataset (auto-generated)\n"
        f"path: {OUT.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"nc: {len(FINAL_NAMES)}\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    return yaml_path


def class_histogram(split: str) -> Counter:
    c: Counter = Counter()
    for lbl in (OUT / "labels" / split).glob("*.txt"):
        for ln in lbl.read_text().splitlines():
            if ln.strip():
                c[int(ln.split()[0])] += 1
    return c


def main() -> None:
    fresh_dirs()
    convert_cardd()

    print("\n[Import CarDD splits]")
    cardd_stats = {}
    for s in SPLITS:
        cardd_stats[s] = import_cardd_split(s)
        print(f"  {s}: imgs={cardd_stats[s][0]}, polygons={cardd_stats[s][1]}")

    print("\n[Import Kaggle splits (hybrid filter + remap)]")
    kaggle_stats = {}
    for s in SPLITS:
        kaggle_stats[s] = import_kaggle_split(s)
        kept, drop, p, soft = kaggle_stats[s]
        print(f"  {s}: kept={kept} (soft={soft}), dropped={drop}, polygons={p}")

    yaml_path = write_yaml()

    print("\n[Final per-split class histograms]")
    for s in SPLITS:
        hist = class_histogram(s)
        pretty = ", ".join(f"{i}:{hist.get(i, 0)}" for i in range(7))
        n_imgs = sum(1 for _ in (OUT / "images" / s).iterdir())
        print(f"  {s}: images={n_imgs}  classes -> {pretty}")

    # Cleanup tmp
    if TMP_CARDD.exists():
        shutil.rmtree(TMP_CARDD)

    print("\n" + "-" * 60)
    print(f"Combined dataset:  {OUT}")
    print(f"YAML:              {yaml_path}")


if __name__ == "__main__":
    main()
