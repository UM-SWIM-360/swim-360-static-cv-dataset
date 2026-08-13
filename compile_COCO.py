#!/usr/bin/env python3
"""
reads dataset_config.yaml (which swimmer goes to which split)
and merges each swimmer's micro-JSON annotation registry into a standard,
trainable COCO keypoints dataset -- Roboflow-style layout.

This script does NOT decide who goes where -- it only follows
dataset_config.yaml. To move a swimmer between splits (e.g. P002 from valid
to train), hand-edit that one line in dataset_config.yaml and re-run this
script; no folders need to be moved manually. To generate
dataset_config.yaml in the first place (or regenerate it from scratch), use
the separate generate_split_config.py.

Registry read
-------------
    <registry_root>/data/<swimmer>/<camera_angle>/<stroke>/annotations/<video_stem>/frame_*.json

(same layout build_dataset.py writes and blur_faces.py reads).

dataset_config.yaml read
-------------------------
    splits:
      P001: train
      P002: valid
      P003: test
      ...

Any swimmer present in the registry but missing from dataset_config.yaml is
excluded from every split (not silently guessed into one) and reported
clearly at the end, so nothing new gets compiled in without a deliberate
decision. Any swimmer listed in the config but not found in the registry is
also reported, non-fatally.

Output written
--------------
<output_dir>/
├── train/
│   ├── images/                 <- every training frame, pooled flat
│   └── _annotations.coco.json  <- each image tagged with stroke (== stroke_type)
│                                   metadata, so a master pool or per-stroke
│                                   subsets can both be pulled from the same file
├── valid/
│   ├── images/
│   └── _annotations.coco.json
├── test/
│   ├── images/
│   └── _annotations.coco.json
└── dataset.yaml                <- categories, flip_idx, resulting per-split stats
                                    (output/provenance -- NOT the same file as
                                    dataset_config.yaml, which is the input)

Each split is fully self-contained: images/ + _annotations.coco.json sit
together. Source images are copied in from --images_root (wherever
blur_faces.py or your raw frame extraction wrote them); pass --link to
hardlink instead of copy (same drive/volume only).

Since images/ is flat (no per-video subfolder -- Roboflow's format doesn't
support that), each pooled filename is prefixed with its video_stem to stay
collision-safe: a bare "frame_000000.png" would otherwise collide between
e.g. a SWIM1 clip and a TURN clip that share a swimmer/angle/stroke.
"<video_stem>_frame_000000.png" instead.

The shared keypoint "categories" schema is read once, using whichever value
the MAJORITY of metadata.json files agree on (not just whichever file
happens to sort first -- that's fragile if the first one is the outlier).

Usage
-----
    python compile_coco.py \\
        --registry_root . \\
        --dataset_config dataset_config.yaml \\
        --output_dir dataset/ \\
        --images_root "[name_of_hf_directory]\blurred_images"

--images_root defaults to "blurred_images" (the output of blur_faces.py),
since that's the anonymised set you'd actually want to train/ship on. Pass
--images_root local_images/ explicitly for raw (non-blurred) frames instead.
Image lookup tries both blur_faces.py's nested layout and build_dataset.py's
flat --extract_frames layout, so either --images_root convention works.

Images currently fetched from Hugging Face / annotations from GitHub note
//////
This script only deals with what's on local disk right now. dataset.yaml has
placeholder "huggingface_repo" / "github_repo" fields (null for now) to fill
in later once that split hosting exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:
    print(
        "This script needs PyYAML to write dataset.yaml.\n"
        "Install it with:  pip install pyyaml --break-system-packages",
        file=sys.stderr,
    )
    raise

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "valid", "test")
ANNOTATIONS_FILENAME = "_annotations.coco.json"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Registry walking
# ---------------------------------------------------------------------------

def iter_registry_frames(registry_root: Path) -> list[Path]:
    """All micro-JSONs under data/<swimmer>/<camera_angle>/<stroke>/annotations/<video>/frame_*.json."""
    return sorted((registry_root / "data").glob("*/*/*/annotations/*/frame_*.json"))


def swimmer_from_frame_path(frame_path: Path, registry_root: Path) -> str:
    """Swimmer is the first path component under data/ -- read from the path
    itself (not the JSON contents) so grouping frames by swimmer doesn't
    require opening every file up front."""
    rel = frame_path.relative_to(registry_root / "data")
    return rel.parts[0]


def describe_category_diff(majority: list[dict], other: list[dict]) -> tuple[bool, list[str]]:
    """Compare two 'categories' values field by field. Returns (structural, lines).

    structural=True means keypoint COUNT or NAME/ORDER actually differs --
    the numeric keypoints arrays in files using this schema are indexed
    differently than the majority schema now labeling them, so those frames'
    keypoints would be silently misaligned once merged (index 3 might mean
    "Left Ear" under the majority schema but something else under theirs).

    structural=False means only cosmetic fields (name/supercategory wording,
    or skeleton edge order -- skeleton is just for visualization, it doesn't
    index into the keypoints array) differ; safe to ignore.
    """
    lines: list[str] = []
    structural = False

    by_id_majority = {c.get("id"): c for c in majority}
    by_id_other = {c.get("id"): c for c in other}

    for cat_id in sorted(set(by_id_majority) | set(by_id_other)):
        maj = by_id_majority.get(cat_id)
        oth = by_id_other.get(cat_id)
        if maj is None or oth is None:
            structural = True
            present_in = "this file" if maj is None else "majority"
            entry = oth if maj is None else maj
            lines.append(
                f"    category id {cat_id}: only in {present_in} -- "
                f"name='{entry.get('name')}', keypoints={len(entry.get('keypoints', []))}"
            )
            continue

        if maj.get("name") != oth.get("name") or maj.get("supercategory") != oth.get("supercategory"):
            lines.append(
                f"    cosmetic: name/supercategory '{maj.get('name')}'/'{maj.get('supercategory')}' "
                f"vs '{oth.get('name')}'/'{oth.get('supercategory')}'"
            )

        maj_kp = maj.get("keypoints", [])
        oth_kp = oth.get("keypoints", [])
        if len(maj_kp) != len(oth_kp):
            structural = True
            lines.append(f"    STRUCTURAL: keypoint count differs -- majority has {len(maj_kp)}, this file has {len(oth_kp)}")
        else:
            mismatches = [(i, m, o) for i, (m, o) in enumerate(zip(maj_kp, oth_kp)) if m != o]
            if mismatches:
                structural = True
                shown = ", ".join(f"index {i}: '{m}' vs '{o}'" for i, m, o in mismatches[:5])
                more = f" (+{len(mismatches) - 5} more)" if len(mismatches) > 5 else ""
                lines.append(f"    STRUCTURAL: keypoint name/order differs -- {shown}{more}")

        if maj.get("skeleton") != oth.get("skeleton"):
            lines.append("    cosmetic: skeleton edge list differs (visualization only, doesn't affect keypoint indexing)")

    return structural, lines


def compare_named_categories(canonical: list[dict], other: list[dict]) -> list[str]:
    """For categories sharing the same NAME between the two schemas (even
    under different id numbers), verify their keypoint lists actually match.
    This is the real safety check for whether a name-based id remap is sound
    -- comparing by raw id number (as describe_category_diff does) compares
    mismatched categories if the id numbering itself differs between files."""
    canonical_by_name = {c.get("name"): c for c in canonical}
    other_by_name = {c.get("name"): c for c in other}
    lines: list[str] = []

    for name in sorted(set(canonical_by_name) & set(other_by_name)):
        c, o = canonical_by_name[name], other_by_name[name]
        if c.get("id") == o.get("id"):
            continue  # same id already covered by describe_category_diff
        if c.get("keypoints", []) == o.get("keypoints", []):
            lines.append(
                f"    NAME MATCH: '{name}' has IDENTICAL keypoints under different ids "
                f"(canonical id {c.get('id')} vs this file's id {o.get('id')}) -- safe to remap {o.get('id')} -> {c.get('id')}"
            )
        else:
            lines.append(
                f"    NAME MISMATCH: '{name}' keypoints differ even though the name matches "
                f"(canonical id {c.get('id')} vs this file's id {o.get('id')}) -- NOT safe to auto-remap"
            )

    for name in sorted(set(other_by_name) - set(canonical_by_name)):
        lines.append(f"    '{name}' (id {other_by_name[name].get('id')}) has no name match in canonical -- cannot remap")

    return lines


def build_category_remap(
    categories_by_path: dict[Path, list],
    canonical: list[dict],
    registry_root: Path,
) -> tuple[dict[tuple[str, str, str], dict[int, int]], set[str]]:
    """Build a per-(swimmer, camera_angle, stroke) local category_id ->
    canonical category_id map, matched by category NAME rather than trusting
    raw id numbers -- so e.g. 'Swimmer' always becomes the canonical pose
    category id regardless of which number a given file's CVAT export
    happened to assign it. Returns (remap_by_scope, names_with_no_canonical_match)."""
    canonical_name_to_id = {c.get("name"): c.get("id") for c in canonical}
    remap_by_scope: dict[tuple[str, str, str], dict[int, int]] = {}
    unmappable_names: set[str] = set()

    for meta_path, cats in categories_by_path.items():
        rel = meta_path.relative_to(registry_root / "data")
        scope = (rel.parts[0], rel.parts[1], rel.parts[2])  # swimmer, camera_angle, stroke
        local_remap: dict[int, int] = {}
        for c in cats:
            name, local_id = c.get("name"), c.get("id")
            canonical_id = canonical_name_to_id.get(name)
            if canonical_id is not None:
                local_remap[local_id] = canonical_id
            else:
                unmappable_names.add(name)
        remap_by_scope[scope] = local_remap

    return remap_by_scope, unmappable_names


def load_categories(registry_root: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[int, int]]]:
    """Read the shared keypoint schema, using whichever 'categories' value the
    MAJORITY of metadata.json files actually agree on -- not just whichever
    file happens to glob-sort first. Picking "first found" is fragile: if
    that file happens to be the outlier (e.g. from an early/different CVAT
    export template), every other (correct, majority) file would incorrectly
    get flagged as "wrong" instead. Divergent files are reported once per
    distinct schema, not once per file, to keep the log readable across a
    large registry -- but each divergent schema gets a real field-by-field
    diff against the majority (both by-id and by-name) so a cosmetic
    difference (naming/wording, or just a different id NUMBERING for the same
    named category) can be told apart from a genuine keypoint-order mismatch.

    Also returns a per-(swimmer, camera_angle, stroke) category_id remap
    (built by matching category NAMES, not raw id numbers) so annotations
    from files that used a different id numbering for the same categories
    get correctly relabeled instead of excluded."""
    meta_paths = sorted((registry_root / "data").glob("*/*/*/metadata.json"))
    if not meta_paths:
        log.error("No metadata.json files found under %s/data -- did build_dataset.py run?", registry_root)
        sys.exit(1)

    categories_by_path: dict[Path, list] = {}
    for meta_path in meta_paths:
        cats = load_json(meta_path).get("categories")
        if cats:
            categories_by_path[meta_path] = cats

    if not categories_by_path:
        log.error("None of the %d metadata.json files have a 'categories' field.", len(meta_paths))
        sys.exit(1)

    serialized_counts: Counter[str] = Counter()
    serialized_to_value: dict[str, list] = {}
    serialized_to_paths: dict[str, list[Path]] = defaultdict(list)
    for path, cats in categories_by_path.items():
        key = json.dumps(cats, sort_keys=True)
        serialized_counts[key] += 1
        serialized_to_value[key] = cats
        serialized_to_paths[key].append(path)

    total = len(categories_by_path)
    canonical_key, canonical_count = serialized_counts.most_common(1)[0]
    categories = serialized_to_value[canonical_key]

    if len(serialized_counts) > 1:
        log.warning(
            "Found %d distinct 'categories' schemas across %d metadata.json files. "
            "Using the majority schema (%d/%d files agree, e.g. %s).",
            len(serialized_counts), total, canonical_count, total, serialized_to_paths[canonical_key][0],
        )
        for key, count in serialized_counts.most_common():
            if key == canonical_key:
                continue
            examples = serialized_to_paths[key]
            more = f" (+{count - 1} more)" if count > 1 else ""
            structural, diff_lines = describe_category_diff(categories, serialized_to_value[key])
            severity = "STRUCTURAL -- keypoints likely misaligned" if structural else "cosmetic only, safe"
            log.warning("  %d file(s) use a DIFFERENT schema [%s], e.g. %s%s", count, severity, examples[0], more)
            for line in diff_lines:
                log.warning(line)
            for line in compare_named_categories(categories, serialized_to_value[key]):
                log.warning(line)
            log.warning("    affected files: %s", ", ".join(str(p) for p in examples))
    else:
        log.info("Categories schema consistent across all %d metadata.json files.", total)

    remap_by_scope, unmappable_names = build_category_remap(categories_by_path, categories, registry_root)
    if unmappable_names:
        log.warning(
            "%d category name(s) found with no match in the canonical schema (will be excluded, not remapped): %s",
            len(unmappable_names), ", ".join(sorted(unmappable_names)),
        )

    return categories, remap_by_scope


def compute_flip_idx(keypoint_names: list[str]) -> list[int]:
    """Left/right keypoint swap indices for flip augmentation, matched
    case-insensitively so naming quirks (e.g. a lowercase 'right eye' among
    otherwise Title Case names) don't break the pairing."""
    name_to_idx = {name.strip().lower(): i for i, name in enumerate(keypoint_names)}
    flip_idx: list[int] = []
    for name in keypoint_names:
        lname = name.strip().lower()
        if lname.startswith("left "):
            pair = "right " + lname[len("left "):]
        elif lname.startswith("right "):
            pair = "left " + lname[len("right "):]
        else:
            pair = lname
        flip_idx.append(name_to_idx.get(pair, name_to_idx[lname]))
    return flip_idx


# ---------------------------------------------------------------------------
# dataset_config.yaml
# ---------------------------------------------------------------------------

def load_split_assignment(config_path: Path) -> dict[str, str]:
    """Read {swimmer: split} from dataset_config.yaml. This is the ONLY
    source of split assignment -- compile_coco.py never computes or changes
    it. Invalid split values fail loudly rather than silently creating an
    unexpected split bucket (e.g. a typo like "vald")."""
    if not config_path.exists():
        log.error(
            "%s not found. Generate it first with generate_split_config.py, e.g.:\n"
            "  python generate_split_config.py --registry_root <registry> --output %s",
            config_path, config_path,
        )
        sys.exit(1)

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    splits = (raw or {}).get("splits")
    if not splits:
        log.error("%s has no top-level 'splits:' mapping.", config_path)
        sys.exit(1)

    assignment: dict[str, str] = {}
    for swimmer, split_name in splits.items():
        if split_name not in SPLIT_NAMES:
            log.error(
                "%s: swimmer %s has invalid split '%s' (must be one of %s).",
                config_path, swimmer, split_name, SPLIT_NAMES,
            )
            sys.exit(1)
        assignment[str(swimmer)] = split_name

    log.info("Loaded split assignment for %d swimmer(s) from %s", len(assignment), config_path)
    return assignment


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_split(
    split_name: str,
    frame_paths: list[Path],
    categories: list[dict[str, Any]],
    category_remap_by_scope: dict[tuple[str, str, str], dict[int, int]],
    images_root: Path,
    split_dir: Path,
    link_images: bool,
    overwrite_images: bool,
) -> dict[str, Any]:
    """Build one split: copy/link its images into <split_dir>/images/, write
    <split_dir>/_annotations.coco.json, and return a stats dict for dataset.yaml."""
    images_out_dir = split_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id = 1
    ann_id = 1

    swimmers_seen: set[str] = set()
    by_camera_angle: Counter[str] = Counter()
    by_stroke: Counter[str] = Counter()
    by_phase: Counter[str] = Counter()
    missing_source_images: list[str] = []
    invalid_category_frames: list[str] = []
    partial_drops: list[str] = []
    images_copied = 0
    images_already_present = 0

    for frame_path in frame_paths:
        record = load_json(frame_path)

        swimmer = record.get("swimmer")
        camera_angle = record.get("camera_angle")
        stroke = record.get("stroke")
        phase = record.get("phase")
        source_video = record.get("source_video")
        frame_index = record.get("frame_index")

        rel_file_name = record["image"]["file_name"]  # "<video_stem>/frame_NNNNNN.png"

        # Remap each annotation's category_id via this frame's scope
        # (by NAME, built in build_category_remap) rather than trusting the
        # raw id number -- different CVAT export sessions can number the same
        # named category differently. Only an annotation whose category name
        # has no match anywhere in the canonical schema gets dropped; the
        # frame itself is only excluded if NONE of its annotations remap.
        scope_remap = category_remap_by_scope.get((swimmer, camera_angle, stroke), {})
        frame_annotations = record.get("annotations", [])
        remapped_annotations: list[dict[str, Any]] = []
        dropped_ids: list[int] = []
        for ann in frame_annotations:
            canonical_id = scope_remap.get(ann.get("category_id"))
            if canonical_id is None:
                dropped_ids.append(ann.get("category_id"))
            else:
                remapped_annotations.append({**ann, "category_id": canonical_id})

        if dropped_ids and remapped_annotations:
            partial_drops.append(f"{rel_file_name}: dropped unmappable category_id(s) {sorted(set(dropped_ids))}, kept {len(remapped_annotations)} other annotation(s)")
        if frame_annotations and not remapped_annotations:
            invalid_category_frames.append(f"{rel_file_name} (all annotation category_id(s) {sorted(set(dropped_ids))} unmappable)")
            continue

        # Two known conventions exist across the pipeline's own scripts:
        #  - blur_faces.py writes  <images_root>/<swimmer>/<camera_angle>/<stroke>/<video_stem>/frame_NNNNNN.png
        #  - build_dataset.py's --extract_frames writes flat: <images_root>/<video_stem>/frame_NNNNNN.png
        # Try both rather than assuming one, so this works regardless of which
        # --images_root convention is actually on disk.
        candidate_paths = [
            images_root / swimmer / camera_angle / stroke / rel_file_name,
            images_root / rel_file_name,
        ]
        source_image_path = next((p for p in candidate_paths if p.exists()), None)

        if source_image_path is None:
            missing_source_images.append(rel_file_name)
            continue  # skip this image (and its annotations) -- keep the split self-consistent

        pooled_name = f"{source_video}_{Path(rel_file_name).name}"
        dest_image_path = images_out_dir / pooled_name

        if dest_image_path.exists() and not overwrite_images:
            images_already_present += 1
        else:
            if link_images:
                if dest_image_path.exists():
                    dest_image_path.unlink()
                os.link(source_image_path, dest_image_path)
            else:
                shutil.copy2(source_image_path, dest_image_path)
            images_copied += 1

        swimmers_seen.add(swimmer)
        by_camera_angle[camera_angle] += 1
        by_stroke[stroke] += 1
        by_phase[phase] += 1

        image_entry = {
            **record["image"],
            "id": image_id,
            "file_name": pooled_name,
            "swimmer": swimmer,
            "camera_angle": camera_angle,
            "stroke": stroke,
            "phase": phase,
            "source_video": source_video,
            "frame_index": frame_index,
        }
        images.append(image_entry)

        for ann in remapped_annotations:
            annotations.append({**ann, "id": ann_id, "image_id": image_id})
            ann_id += 1

        image_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    save_json(coco, split_dir / ANNOTATIONS_FILENAME)

    if partial_drops:
        log.warning(
            "  [%s] %d frame(s) had at least one unmappable annotation dropped but were otherwise kept "
            "(showing up to 10):", split_name, len(partial_drops),
        )
        for name in partial_drops[:10]:
            log.warning("    • %s", name)

    if invalid_category_frames:
        log.warning(
            "  [%s] %d frame(s) excluded -- annotation category_id not declared in output categories "
            "(showing up to 10):",
            split_name, len(invalid_category_frames),
        )
        for name in invalid_category_frames[:10]:
            log.warning("    • %s", name)

    if missing_source_images:
        log.warning(
            "  [%s] %d source image(s) not found under %s -- excluded from this split (showing up to 10):",
            split_name, len(missing_source_images), images_root,
        )
        for name in missing_source_images[:10]:
            log.warning("    • %s", name)

    return {
        "dir": split_name,
        "annotations_file": f"{split_name}/{ANNOTATIONS_FILENAME}",
        "images_dir": f"{split_name}/images",
        "num_swimmers": len(swimmers_seen),
        "swimmers": sorted(swimmers_seen),
        "num_images": len(images),
        "num_annotations": len(annotations),
        "images_copied_or_linked": images_copied,
        "images_already_present": images_already_present,
        "missing_source_images": len(missing_source_images),
        "invalid_category_frames": len(invalid_category_frames),
        "partial_category_drops": len(partial_drops),
        "by_camera_angle": dict(sorted(by_camera_angle.items())),
        "by_stroke": dict(sorted(by_stroke.items())),
        "by_phase": dict(sorted(by_phase.items())),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile the SWIM-360/DIVE micro-JSON registry into a Roboflow-style "
                    "train/valid/test COCO dataset + dataset.yaml, using dataset_config.yaml "
                    "as the source of truth for which swimmer goes where.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--registry_root", type=Path, required=True,
                        help="Repo root containing data/<swimmer>/<camera_angle>/<stroke>/annotations/...")
    parser.add_argument("--dataset_config", type=Path, default=Path("dataset_config.yaml"),
                        help="Maps swimmer -> split. Generate with generate_split_config.py if it doesn't exist yet.")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Where train/, valid/, test/, and dataset.yaml are written.")
    parser.add_argument("--images_root", type=Path, default=Path("blurred_images"),
                        help="Where source frame images currently live (e.g. blur_faces.py's --output_dir).")
    parser.add_argument("--link", action="store_true",
                        help="Hardlink images into place instead of copying (saves disk space; "
                             "--images_root and --output_dir must be on the same drive/volume).")
    parser.add_argument("--overwrite_images", action="store_true",
                        help="Re-copy/re-link images even if already present in the split's images/ folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (args.registry_root / "data").is_dir():
        log.error("No data/ directory found under %s -- did build_dataset.py run?", args.registry_root)
        sys.exit(1)
    if not args.images_root.is_dir():
        log.error("--images_root %s does not exist.", args.images_root)
        sys.exit(1)

    assignment = load_split_assignment(args.dataset_config)

    frame_paths = iter_registry_frames(args.registry_root)
    if not frame_paths:
        log.error("No frame_*.json files found under %s/data.", args.registry_root)
        sys.exit(1)
    log.info("Found %d frame(s) across the registry.", len(frame_paths))

    categories, category_remap_by_scope = load_categories(args.registry_root)
    keypoint_names = categories[0].get("keypoints", []) if categories else []
    flip_idx = compute_flip_idx(keypoint_names) if keypoint_names else []

    # Group frame paths by swimmer (path-derived, no JSON parsing needed yet).
    frames_by_swimmer: dict[str, list[Path]] = defaultdict(list)
    for frame_path in frame_paths:
        swimmer = swimmer_from_frame_path(frame_path, args.registry_root)
        frames_by_swimmer[swimmer].append(frame_path)

    registry_swimmers = set(frames_by_swimmer)
    config_swimmers = set(assignment)

    # Swimmers with annotated frames but no entry in dataset_config.yaml are
    # excluded from every split (never guessed into one) -- reported clearly
    # so a newly-annotated swimmer doesn't silently vanish from the compiled
    # dataset just because nobody added them to the config yet.
    unconfigured = sorted(registry_swimmers - config_swimmers, key=lambda s: -len(frames_by_swimmer[s]))
    if unconfigured:
        log.warning(
            "%d swimmer(s) have annotated frames but no entry in %s -- EXCLUDED from all splits: %s",
            len(unconfigured), args.dataset_config,
            ", ".join(f"{s} ({len(frames_by_swimmer[s])} frames)" for s in unconfigured),
        )

    # Swimmers listed in the config but with no frames in the registry (typo,
    # or just not annotated yet) -- non-fatal, just informational.
    unmatched_config = sorted(config_swimmers - registry_swimmers)
    if unmatched_config:
        log.warning(
            "%d swimmer(s) listed in %s have no frames in the registry (typo, or not annotated yet): %s",
            len(unmatched_config), args.dataset_config, ", ".join(unmatched_config),
        )

    frames_by_split: dict[str, list[Path]] = defaultdict(list)
    for swimmer, split_name in assignment.items():
        frames_by_split[split_name].extend(frames_by_swimmer.get(swimmer, []))
    for split_name in frames_by_split:
        frames_by_split[split_name].sort()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.link:
        log.info("Hardlinking images into place (--link).")

    all_stats: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        split_frames = frames_by_split.get(split_name, [])
        split_dir = args.output_dir / split_name
        if not split_frames:
            log.warning("Split '%s' has 0 swimmers/frames per %s; writing an empty split.",
                        split_name, args.dataset_config)
        stats = compile_split(
            split_name, split_frames, categories, category_remap_by_scope, args.images_root, split_dir,
            link_images=args.link, overwrite_images=args.overwrite_images,
        )
        all_stats[split_name] = stats
        log.info(
            "  [%s] %d swimmer(s), %d image(s), %d annotation(s) -> %s/",
            split_name, stats["num_swimmers"], stats["num_images"], stats["num_annotations"], split_name,
        )

    total_images = sum(s["num_images"] for s in all_stats.values())
    achieved_ratios = {
        name: round(s["num_images"] / total_images, 4) if total_images else 0.0
        for name, s in all_stats.items()
    }

    dataset_yaml = {
        "dataset": {
            "name": "SWIM-360 / DIVE Pose Dataset",
            # Filled in later once images/annotations are actually hosted; local-only for now.
            "huggingface_repo": None,
            "github_repo": None,
        },
        "categories": categories,
        "num_keypoints": len(keypoint_names),
        "flip_idx": flip_idx,
        "split": {
            "method": "participant-disjoint (assignment read from dataset_config.yaml, not computed here)",
            "config_file": str(args.dataset_config),
            "achieved_ratios": achieved_ratios,
            "unconfigured_swimmers_excluded": unconfigured,
            "config_swimmers_not_in_registry": unmatched_config,
        },
        "splits": all_stats,
    }
    yaml_path = args.output_dir / "dataset.yaml"
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(dataset_yaml, fh, sort_keys=False, allow_unicode=True)

    print()
   
    print("COMPILE COCO SUMMARY")
    
    for split_name in SPLIT_NAMES:
        s = all_stats[split_name]
        print(f"{split_name:5s}: {s['num_swimmers']:3d} swimmers  {s['num_images']:6d} images  "
              f"{s['num_annotations']:6d} annotations  ({achieved_ratios[split_name]:.1%} of compiled total)")
        if s["missing_source_images"]:
            print(f"       {s['missing_source_images']} image(s) missing from --images_root, excluded")
        if s["invalid_category_frames"]:
            print(f"       {s['invalid_category_frames']} frame(s) excluded -- no annotation could be remapped (see warnings above)")
        if s["partial_category_drops"]:
            print(f"       {s['partial_category_drops']} frame(s) kept but had an unmappable annotation dropped (see warnings above)")
    if unconfigured:
        print()
        print(f"EXCLUDED (not in {args.dataset_config}): {', '.join(unconfigured)}")
        print(f"  Add them to {args.dataset_config} and re-run to include them.")
    print()
    print(f"Categories        : {len(keypoint_names)} keypoints, flip_idx computed")
    print(f"Output directory  : {args.output_dir}")
    print(f"  train/images + train/{ANNOTATIONS_FILENAME}")
    print(f"  valid/images + valid/{ANNOTATIONS_FILENAME}")
    print(f"  test/images  + test/{ANNOTATIONS_FILENAME}")
    print(f"  dataset.yaml")
    


if __name__ == "__main__":
    main()