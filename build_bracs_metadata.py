from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build metadata from the official BRACS ROI split tree")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="projects/bracs_metadata.csv")
    parser.add_argument("--source-magnification", type=float, default=40.0)
    args = parser.parse_args()
    root = Path(args.root)
    expected = {"train", "val", "test"}
    actual = {path.name for path in root.iterdir() if path.is_dir()}
    if not expected.issubset(actual):
        raise RuntimeError(f"Official train/val/test directories are required, got {sorted(actual)}")
    rows = []
    for split in ("train", "val", "test"):
        for class_dir in sorted((root / split).iterdir()):
            if not class_dir.is_dir() or "_" not in class_dir.name:
                continue
            class_index, label = class_dir.name.split("_", 1)
            for path in sorted(class_dir.glob("*.png")):
                with Image.open(path) as image:
                    width, height = image.size
                rows.append(
                    {
                        "sample_id": path.stem,
                        "image_path": str(path.resolve()),
                        "split": split,
                        "class_index": int(class_index),
                        "label": label,
                        "magnification": args.source_magnification,
                        "width": width,
                        "height": height,
                    }
                )
    if len(rows) != 4539:
        raise RuntimeError(f"Expected 4,539 BRACS ROIs, found {len(rows)}")
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {split: sum(row["split"] == split for row in rows) for split in expected}
    print(f"Wrote {len(rows)} rows to {target}: {counts}")


if __name__ == "__main__":
    main()
