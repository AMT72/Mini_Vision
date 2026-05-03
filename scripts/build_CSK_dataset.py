"""
Build the CSK (CarDD + Saudi + Kaggle) merged dataset for YOLO seg training.

Sources
-------
CarDD  : data/CarDD_release/CarDD_COCO/  — COCO format, pre-split train2017/val2017/test2017
Saudi  : data/OUR_ANNOTATIE_DATA/saudi_dataset/ — YOLO seg, pre-split train/val/test
Kaggle : data/damages_only_dataset/ — YOLO seg, pre-split train/val/test

Final 6-class taxonomy (matches CarDD exactly)
----------------------------------------------
  0: dent          CarDD(0) + Saudi(0) + Kaggle(id4→0)
  1: scratch       CarDD(1) + Saudi(1) + Kaggle(id2→1)
  2: crack         CarDD(2) + Saudi(2) + Kaggle(id3→2)
  3: glass shatter CarDD(3) + Saudi(3)
  4: lamp broken   CarDD(4) + Saudi(4)
  5: tire flat     CarDD(5) + Saudi(5)

Kaggle filter (STRICT — no hybrid this time)
--------------------------------------------
  Keep image only if ALL its annotation classes are in {2, 3, 4}.
  Any image touching 0(Missing), 1(Broken), 5(Flaking), 6(Paint chip),
  7(Corrosion) is dropped entirely. No partial polygon deletion.

Splits: merge split-to-split to avoid leakage.
  CSK train ← CarDD train2017 + Saudi train + Kaggle train (filtered)
  CSK val   ← CarDD val2017   + Saudi val   + Kaggle val   (filtered)
  CSK test  ← CarDD test2017  + Saudi test  + Kaggle test  (filtered)

Output: data/CSK_Dataset/
"""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

from ultralytics.data.converter import convert_coco

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "data" / "CSK_Dataset"
TMP  = ROOT / "data" / "_tmp_cardd_yolo"

# ── Source paths ──────────────────────────────────────────────────────────────
CARDD_ANN   = ROOT / "data" / "CarDD_release" / "CarDD_COCO" / "annotations"
CARDD_IMGS  = ROOT / "data" / "CarDD_release" / "CarDD_COCO"
CARDD_SPLIT = {"train": "train2017", "val": "val2017", "test": "test2017"}

SAUDI_DIR   = ROOT / "data" / "OUR_ANNOTATIE_DATA" / "saudi_dataset"
KAGGLE_DIR  = ROOT / "data" / "damages_only_dataset"

SPLITS = ["train", "val", "test"]

FINAL_NAMES = {
    0: "dent", 1: "scratch", 2: "crack",
    3: "glass shatter", 4: "lamp broken", 5: "tire flat",
}

# Kaggle id → final id. None = unmapped (drop entire image if present).
KAGGLE_REMAP: dict[int, int | None] = {
    0: None,  # Missing part  → drop
    1: None,  # Broken part   → drop
    2: 1,     # Scratch       → scratch
    3: 2,     # Cracked       → crack
    4: 0,     # Dent          → dent
    5: None,  # Flaking       → drop
    6: None,  # Paint chip    → drop
    7: None,  # Corrosion     → drop
}
KAGGLE_ALLOWED = {k for k, v in KAGGLE_REMAP.items() if v is not None}  # {2,3,4}


# ── Helpers ───────────────────────────────────────────────────────────────────

def fresh_dirs() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    if TMP.exists():
        shutil.rmtree(TMP)
    for split in SPLITS:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)


def class_histogram(split: str) -> Counter:
    c: Counter = Counter()
    for lbl in (OUT / "labels" / split).glob("*.txt"):
        for ln in lbl.read_text().splitlines():
            if ln.strip():
                c[int(ln.split()[0])] += 1
    return c


# ── CarDD ─────────────────────────────────────────────────────────────────────

def import_cardd() -> None:
    print("[CarDD] Converting COCO → YOLO …")
    convert_coco(
        labels_dir=str(CARDD_ANN),
        save_dir=str(TMP),
        use_segments=True,
        cls91to80=False,
    )
    for split in SPLITS:
        src_lbl = TMP / "labels" / CARDD_SPLIT[split]
        src_img = CARDD_IMGS / CARDD_SPLIT[split]
        dst_lbl = OUT / "labels" / split
        dst_img = OUT / "images" / split

        if not src_lbl.exists():
            print(f"  [warn] CarDD converted labels missing for {split}, skipping")
            continue

        n = 0
        for lbl in sorted(src_lbl.glob("*.txt")):
            img = src_img / f"{lbl.stem}.jpg"
            if not img.exists():
                continue
            shutil.copy2(lbl, dst_lbl / lbl.name)
            shutil.copy2(img, dst_img / img.name)
            n += 1
        print(f"  {split}: {n} images")

    shutil.rmtree(TMP, ignore_errors=True)


# ── Saudi ─────────────────────────────────────────────────────────────────────

def import_saudi() -> None:
    print("[Saudi] Copying pre-converted YOLO labels …")
    for split in SPLITS:
        src_lbl = SAUDI_DIR / "labels" / split
        src_img = SAUDI_DIR / "images" / split
        dst_lbl = OUT / "labels" / split
        dst_img = OUT / "images" / split

        if not src_lbl.exists():
            print(f"  [warn] Saudi split '{split}' missing, skipping")
            continue

        n = 0
        for lbl in sorted(src_lbl.glob("*.txt")):
            # Find matching image (mixed extensions in Saudi)
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                candidate = src_img / f"{lbl.stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue
            # Prefix with "saudi_" to avoid any stem collisions with CarDD
            shutil.copy2(lbl,      dst_lbl / f"saudi_{lbl.name}")
            shutil.copy2(img_path, dst_img / f"saudi_{img_path.name}")
            n += 1
        print(f"  {split}: {n} images")


# ── Kaggle ────────────────────────────────────────────────────────────────────

def import_kaggle() -> None:
    print("[Kaggle] Strict-filtering and remapping …")
    for split in SPLITS:
        src_lbl = KAGGLE_DIR / "labels" / split
        src_img = KAGGLE_DIR / "images" / split
        dst_lbl = OUT / "labels" / split
        dst_img = OUT / "images" / split

        if not src_lbl.exists():
            print(f"  [warn] Kaggle split '{split}' missing, skipping")
            continue

        kept = dropped = 0
        for lbl in sorted(src_lbl.glob("*.txt")):
            lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
            ids = {int(ln.split()[0]) for ln in lines}

            # Strict: drop if ANY id is unmapped
            if not ids.issubset(KAGGLE_ALLOWED):
                dropped += 1
                continue

            # Find image
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                candidate = src_img / f"{lbl.stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                dropped += 1
                continue

            # Remap class ids
            new_lines = []
            for ln in lines:
                parts = ln.split()
                new_id = KAGGLE_REMAP[int(parts[0])]
                new_lines.append(f"{new_id} " + " ".join(parts[1:]))

            out_lbl = dst_lbl / f"kaggle_{lbl.name}"
            out_lbl.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            shutil.copy2(img_path, dst_img / f"kaggle_{img_path.name}")
            kept += 1

        print(f"  {split}: kept={kept}, dropped={dropped}")


# ── YAML ──────────────────────────────────────────────────────────────────────

def write_yaml() -> Path:
    yaml_path = OUT / "CSK_dataset.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in FINAL_NAMES.items())
    yaml_path.write_text(
        "# CSK Dataset: CarDD + Saudi + Kaggle (auto-generated)\n"
        f"path: {OUT.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "test:  images/test\n\n"
        f"nc: {len(FINAL_NAMES)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    return yaml_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Building CSK_Dataset → {OUT}\n")
    fresh_dirs()
    import_cardd()
    print()
    import_saudi()
    print()
    import_kaggle()
    print()

    yaml_path = write_yaml()

    print("─" * 60)
    print("Final image counts:")
    for split in SPLITS:
        n = sum(1 for _ in (OUT / "images" / split).iterdir())
        print(f"  {split}: {n}")

    print("\nPer-split class histograms:")
    for split in SPLITS:
        hist = class_histogram(split)
        row = ", ".join(f"{FINAL_NAMES[i]}:{hist.get(i,0)}" for i in range(6))
        print(f"  {split}: {row}")

    total_anns = sum(class_histogram(s)[c] for s in SPLITS for c in range(6))
    print(f"\nTotal annotations across all splits: {total_anns}")
    print(f"YAML written to: {yaml_path}")
    print("\nDone ✓")


if __name__ == "__main__":
    main()
