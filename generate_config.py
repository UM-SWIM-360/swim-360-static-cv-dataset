#!/usr/bin/env python3
"""
generate_split_config.py
=========================
One-time (or occasional) bootstrap: computes a participant-disjoint
train/valid/test split over the swimmers in the registry and writes
dataset_config.yaml -- a small, hand-editable file mapping each swimmer ID
to a split.

This is deliberately a SEPARATE tool from compile_coco.py. compile_coco.py's
only job is to read dataset_config.yaml and merge JSONs; it no longer
computes or changes any swimmer's split assignment. Once dataset_config.yaml
exists, moving a swimmer between splits is a one-line hand-edit + a
compile_coco.py re-run -- no need to touch this script again.

Safety: refuses to overwrite an existing dataset_config.yaml unless --force
is passed, since manual edits (e.g. "move P002 to train") are meant to
persist across compiles and shouldn't be silently wiped out by a re-run.

Split method
------------
Swimmers are assigned with a greedy load-balancing pass: largest
swimmer-by-frame-count first, each assigned to whichever split is currently
furthest below its target ratio (ties broken toward the split with the
larger target ratio). This keeps achieved image-count ratios close to the
requested ones despite uneven per-swimmer frame counts, while guaranteeing
no swimmer's frames ever cross a split boundary. --seed only affects
tie-breaking among swimmers with exactly equal frame counts.

Usage
-----
    python generate_split_config.py \\
        --registry_root swimmer-pose-repo/ \\
        --output dataset_config.yaml \\
        --train_ratio 0.7 --valid_ratio 0.15 --test_ratio 0.15
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:
    print(
        "This script needs PyYAML.\nInstall it with:  python -m pip install pyyaml --break-system-packages",
        file=sys.stderr,
    )
    raise

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "valid", "test")


def iter_registry_frames(registry_root: Path) -> list[Path]:
    return sorted((registry_root / "data").glob("*/*/*/annotations/*/frame_*.json"))


def swimmer_from_frame_path(frame_path: Path, registry_root: Path) -> str:
    rel = frame_path.relative_to(registry_root / "data")
    return rel.parts[0]


def assign_swimmers_to_splits(
    swimmer_frame_counts: dict[str, int],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Greedy load-balanced participant-disjoint assignment (see module docstring)."""
    active_splits = [s for s in SPLIT_NAMES if ratios.get(s, 0) > 0]
    if not active_splits:
        log.error("All split ratios are 0; nothing to assign.")
        sys.exit(1)

    rng = random.Random(seed)
    swimmers = list(swimmer_frame_counts.keys())
    rng.shuffle(swimmers)
    swimmers.sort(key=lambda s: swimmer_frame_counts[s], reverse=True)

    running_counts = {s: 0 for s in active_splits}
    assignment: dict[str, str] = {}

    for swimmer in swimmers:
        target = min(active_splits, key=lambda s: (running_counts[s] / ratios[s], -ratios[s]))
        assignment[swimmer] = target
        running_counts[target] += swimmer_frame_counts[swimmer]

    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute a participant-disjoint train/valid/test split and write dataset_config.yaml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--registry_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dataset_config.yaml"))
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite --output even if it already exists.")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        parser.error(f"--train_ratio + --valid_ratio + --test_ratio must sum to 1.0 (got {ratio_sum}).")

    if args.output.exists() and not args.force:
        log.error(
            "%s already exists. Refusing to overwrite it (it may contain hand-edits) -- "
            "pass --force if you really want to regenerate it from scratch.",
            args.output,
        )
        sys.exit(1)

    if not (args.registry_root / "data").is_dir():
        log.error("No data/ directory found under %s -- did build_dataset.py run?", args.registry_root)
        sys.exit(1)

    frame_paths = iter_registry_frames(args.registry_root)
    if not frame_paths:
        log.error("No frame_*.json files found under %s/data.", args.registry_root)
        sys.exit(1)

    frames_by_swimmer: dict[str, list[Path]] = defaultdict(list)
    for frame_path in frame_paths:
        swimmer = swimmer_from_frame_path(frame_path, args.registry_root)
        frames_by_swimmer[swimmer].append(frame_path)
    swimmer_frame_counts = {s: len(paths) for s, paths in frames_by_swimmer.items()}

    log.info("Found %d frame(s) across %d swimmer(s).", len(frame_paths), len(swimmer_frame_counts))

    ratios = {"train": args.train_ratio, "valid": args.valid_ratio, "test": args.test_ratio}
    assignment = assign_swimmers_to_splits(swimmer_frame_counts, ratios, args.seed)

    achieved_counts = defaultdict(int)
    for swimmer, split_name in assignment.items():
        achieved_counts[split_name] += swimmer_frame_counts[swimmer]
    total = sum(achieved_counts.values())

    config: dict[str, Any] = {"splits": {s: assignment[s] for s in sorted(assignment)}}

    header = (
        "# dataset_config.yaml\n"
        "# Maps each swimmer ID to a split (train / valid / test).\n"
        "# Hand-edit a value below + re-run compile_coco.py to move that swimmer\n"
        "# between splits -- no folders need to be moved manually.\n"
        "# Generated by generate_split_config.py; re-running that script will NOT\n"
        "# overwrite this file unless --force is passed, so edits here are safe.\n\n"
    )
    with args.output.open("w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)

    print()
    print("=" * 60)
    print("SPLIT CONFIG GENERATED")
    print("=" * 60)
    for split_name in SPLIT_NAMES:
        swimmers_in_split = sorted(s for s, sp in assignment.items() if sp == split_name)
        pct = achieved_counts[split_name] / total * 100 if total else 0.0
        print(f"{split_name:5s}: {len(swimmers_in_split):3d} swimmers  {achieved_counts[split_name]:6d} frames "
              f"(target {ratios[split_name]:.0%}, achieved {pct:.1f}%)")
        print(f"       {', '.join(swimmers_in_split)}")
    print()
    print(f"Written to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()