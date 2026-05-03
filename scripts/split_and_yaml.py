"""
Task 2 + 3: Split staged images/labels 80/10/10 into the standard YOLO
folder layout, then emit `damage_dataset.yaml`.

Run AFTER `supervisely_to_yolo.py`. Image+label files always move together.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data" / "damages_only_dataset"
STAGE_IMG = DATASET_DIR / "_staging" / "images"
STAGE_LBL = DATASET_DIR / "_staging" / "labels"

SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}
SEED = 42

CLASS_NAMES = {
    0: "Missing part",
    1: "Broken part",
    2: "Scratch",
    3: "Cracked",
    4: "Dent",
    5: "Flaking",
    6: "Paint chip",
    7: "Corrosion",
}


def collect_pairs() -> list[tuple[Path, Path]]:
    """Return [(image_path, label_path), ...] for every image with a matching label."""
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(STAGE_IMG.iterdir()):
        if not img.is_file():
            continue
        lbl = STAGE_LBL / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def split_pairs(pairs: list[tuple[Path, Path]]) -> dict[str, list[tuple[Path, Path]]]:
    rng = random.Random(SEED)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])
    # remainder goes to test so we use every sample
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }


def materialize(splits: dict[str, list[tuple[Path, Path]]]) -> None:
    for split, items in splits.items():
        img_out = DATASET_DIR / "images" / split
        lbl_out = DATASET_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.copy2(img, img_out / img.name)
            shutil.copy2(lbl, lbl_out / lbl.name)


def write_yaml() -> Path:
    yaml_path = DATASET_DIR / "damage_dataset.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in CLASS_NAMES.items())
    content = (
        f"# YOLO instance segmentation dataset config (auto-generated)\n"
        f"path: {DATASET_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_block}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main() -> None:
    if not STAGE_IMG.exists() or not STAGE_LBL.exists():
        raise SystemExit(f"Staging dirs missing. Run supervisely_to_yolo.py first.")

    pairs = collect_pairs()
    print(f"Collected {len(pairs)} image+label pairs from staging.")
    if not pairs:
        raise SystemExit("No pairs found; nothing to split.")

    splits = split_pairs(pairs)
    for k, v in splits.items():
        print(f"  {k}: {len(v)}")

    materialize(splits)
    yaml_path = write_yaml()

    print("-" * 60)
    print(f"Dataset written to: {DATASET_DIR}")
    print(f"YAML config:        {yaml_path}")
    print("Tip: you can now delete the _staging/ folder if you want.")


if __name__ == "__main__":
    main()
