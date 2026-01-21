# -*- coding: utf-8 -*-
"""
VSCode-friendly CLEVR single-shape statistics.

What it does:
1) Count images whose objects all share the same shape (cube/sphere/cylinder)
2) Output distribution by number of objects: 1->X, 2->Y, ...
3) Output distribution by shape category

How to run (VSCode):
- Open this file
- Click "Run Python File" (top-right triangle)
No command-line args needed.

Optional:
- Set ROOT_DIR below if your CLEVR_v1.0 is elsewhere.
- Set SAVE_CSV_PATH to export CSV.

Expected CLEVR structure:
CLEVR_v1.0/
  scenes/CLEVR_train_scenes.json (and/or val/test)
"""


# [INFO] Using CLEVR root: D:\github file\REVAE\CLEVR_v1.0
# ========================================================================
# Split: train
# Total images (scenes): 70000
# Single-shape images:   1458
# Ratio:                2.08%
# ------------------------------------------------------------------------
# Distribution by number of objects (only single-shape images):
#    3 objects: 982
#    4 objects: 278
#    5 objects: 129
#    6 objects: 45
#    7 objects: 19
#    8 objects: 3
#    9 objects: 1
#   10 objects: 1
# ------------------------------------------------------------------------
# Distribution by shape category (only single-shape images):
#     sphere: 507
#   cylinder: 479
#       cube: 472
# ========================================================================

# ========================================================================
# Split: val
# Total images (scenes): 15000
# Single-shape images:   317
# Ratio:                2.11%
# ------------------------------------------------------------------------
# Distribution by number of objects (only single-shape images):
#    3 objects: 212
#    4 objects: 70
#    5 objects: 23
#    6 objects: 8
#    7 objects: 3
#    8 objects: 1
# ------------------------------------------------------------------------
# Distribution by shape category (only single-shape images):
#   cylinder: 109
#       cube: 105
#     sphere: 103
# ========================================================================

from __future__ import annotations
import json
import os
from collections import Counter
from typing import Dict, List, Tuple, Optional

# =========================
# User settings (edit here)
# =========================

# If None, the script will auto-detect CLEVR_v1.0 near this .py file.
ROOT_DIR: Optional[str] = r"D:\github file\REVAE\CLEVR_v1.0"

# Which splits to analyze. Use ["train","val","test"] or subset.
# If None, analyze all splits found under scenes/.
SPLITS: Optional[List[str]] = None

# If not None, export distribution by num_objects to CSV.
# Example: "single_shape_distribution.csv"
SAVE_CSV_PATH: Optional[str] = None

# If True, treat empty-object scenes as "single-shape" (usually False).
COUNT_EMPTY_SCENES_AS_SINGLE_SHAPE: bool = False


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _auto_find_clevr_root() -> str:
    """
    Auto-detect CLEVR_v1.0 folder by searching in:
    1) script directory
    2) parent directory
    3) grandparent directory
    """
    candidates = []
    base = _script_dir()
    candidates.append(os.path.join(base, "CLEVR_v1.0"))
    candidates.append(os.path.join(os.path.dirname(base), "CLEVR_v1.0"))
    candidates.append(os.path.join(os.path.dirname(os.path.dirname(base)), "CLEVR_v1.0"))

    for c in candidates:
        scenes_dir = os.path.join(c, "scenes")
        if os.path.isdir(scenes_dir):
            return c

    # If not found, give a helpful error
    raise FileNotFoundError(
        "Cannot auto-detect 'CLEVR_v1.0'.\n"
        "Fix options:\n"
        "  1) Put this .py file in the same directory that contains CLEVR_v1.0/\n"
        "  2) Or set ROOT_DIR at the top to your absolute path, e.g.:\n"
        r"     ROOT_DIR = r'D:\datasets\CLEVR_v1.0'\n"
    )


def find_scene_jsons(root: str) -> Dict[str, str]:
    """
    Find CLEVR_*_scenes.json under <root>/scenes.
    Returns mapping split -> filepath.
    """
    scenes_dir = os.path.join(root, "scenes")
    if not os.path.isdir(scenes_dir):
        raise FileNotFoundError(
            f"Cannot find scenes directory: {scenes_dir}\n"
            f"Please ensure ROOT_DIR points to CLEVR_v1.0 folder."
        )

    mapping: Dict[str, str] = {}
    for fname in os.listdir(scenes_dir):
        if fname.startswith("CLEVR_") and fname.endswith("_scenes.json"):
            # e.g., CLEVR_train_scenes.json
            parts = fname.split("_")
            if len(parts) >= 3:
                split = parts[1].lower()
                mapping[split] = os.path.join(scenes_dir, fname)

    if not mapping:
        raise FileNotFoundError(f"No CLEVR_*_scenes.json found under: {scenes_dir}")
    return mapping


def load_scenes(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "scenes" not in data or not isinstance(data["scenes"], list):
        raise ValueError(f"Invalid scenes json format: {path} (missing 'scenes' list)")
    return data["scenes"]


def analyze_split(scenes: List[dict]) -> Tuple[int, int, Counter, Counter]:
    """
    Returns:
      total_images: int
      single_shape_images: int
      dist_by_num_objects (Counter): num_objects -> count (only for single-shape images)
      dist_by_shape (Counter): shape -> count images (only for single-shape images)
    """
    total_images = 0
    single_shape_images = 0
    dist_by_num_objects: Counter = Counter()
    dist_by_shape: Counter = Counter()

    for scene in scenes:
        total_images += 1
        objs = scene.get("objects", [])

        if not objs:
            if COUNT_EMPTY_SCENES_AS_SINGLE_SHAPE:
                single_shape_images += 1
                dist_by_num_objects[0] += 1
                dist_by_shape["(empty)"] += 1
            continue

        shapes = [o.get("shape", None) for o in objs]
        shapes = [s for s in shapes if s is not None]
        if not shapes:
            continue

        unique_shapes = set(shapes)
        if len(unique_shapes) == 1:
            single_shape_images += 1
            nobj = len(objs)
            dist_by_num_objects[nobj] += 1
            the_shape = next(iter(unique_shapes))
            dist_by_shape[the_shape] += 1

    return total_images, single_shape_images, dist_by_num_objects, dist_by_shape


def print_report(split: str, total_images: int, single_shape_images: int,
                 dist_by_num_objects: Counter, dist_by_shape: Counter) -> None:
    print("=" * 72)
    print(f"Split: {split}")
    print(f"Total images (scenes): {total_images}")
    print(f"Single-shape images:   {single_shape_images}")
    if total_images > 0:
        ratio = single_shape_images / total_images * 100.0
        print(f"Ratio:                {ratio:.2f}%")
    print("-" * 72)

    if dist_by_num_objects:
        print("Distribution by number of objects (only single-shape images):")
        for nobj in sorted(dist_by_num_objects.keys()):
            print(f"  {nobj:>2} objects: {dist_by_num_objects[nobj]}")
    else:
        print("No single-shape images found (or scenes contain no objects).")

    print("-" * 72)

    if dist_by_shape:
        print("Distribution by shape category (only single-shape images):")
        for shape, cnt in dist_by_shape.most_common():
            print(f"  {shape:>8}: {cnt}")

    print("=" * 72)
    print()


def save_csv(path: str, rows: List[Tuple[str, int, int]]) -> None:
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "num_objects", "count_images"])
        for r in rows:
            w.writerow(list(r))


def main():
    # 1) Resolve root
    root = ROOT_DIR
    if root is None:
        root = _auto_find_clevr_root()

    print(f"[INFO] Using CLEVR root: {os.path.abspath(root)}")

    # 2) Find scene jsons
    scene_map = find_scene_jsons(root)

    # 3) Determine splits
    splits = SPLITS
    if splits is None:
        splits = sorted(scene_map.keys())
    else:
        splits = [s.lower() for s in splits]
        missing = [s for s in splits if s not in scene_map]
        if missing:
            raise ValueError(f"Requested splits not found: {missing}. Available: {sorted(scene_map.keys())}")

    csv_rows: List[Tuple[str, int, int]] = []

    # 4) Analyze
    for sp in splits:
        scenes = load_scenes(scene_map[sp])
        total_images, single_shape_images, dist_by_num_objects, dist_by_shape = analyze_split(scenes)
        print_report(sp, total_images, single_shape_images, dist_by_num_objects, dist_by_shape)

        for nobj in sorted(dist_by_num_objects.keys()):
            csv_rows.append((sp, nobj, dist_by_num_objects[nobj]))

    # 5) Optional save
    if SAVE_CSV_PATH:
        save_csv(SAVE_CSV_PATH, csv_rows)
        print(f"[Saved] CSV written to: {os.path.abspath(SAVE_CSV_PATH)}")


if __name__ == "__main__":
    main()
