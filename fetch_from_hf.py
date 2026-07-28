#!/usr/bin/env python3
"""
-------------------------------------------------------------
Fetch SWIM-360/DIVE files from a Hugging Face dataset repo
-------------------------------------------------------------
Downloads files from a Hugging Face Hub dataset repository into a local
folder, with the option to pull everything or just a slice of it (one
swimmer, one camera angle, one stroke, images only, JSON only, etc).

Filtering
---------
Either build a filter from the registry's own dimensions:
    --swimmer swimmer_003 --camera_angle Above_Water --stroke freestyle
(any of these left unset match everything at that level), scoped under
a top-level folder with --prefix (default: 'blurred_images'), OR pass
raw glob patterns directly for anything outside that layout:
    --allow_patterns "data/**/*.json"
If both are given, --allow_patterns/--ignore_patterns are used as-is
and the --swimmer/--camera_angle/--stroke/--prefix convenience filter
is ignored -- edit PATTERN below to match your actual repo layout if
it differs from <prefix>/<swimmer>/<camera_angle>/<stroke>/...

Downloads are resumable: files already present and up to date under
--local_dir are skipped on re-run.

Usage
-----
    # Everything
    python fetch_from_hf.py --repo_id liamkazzopard/swim360-dive --local_dir [name of local folder]

    # Just one swimmer's blurred images
    python fetch_from_hf.py --repo_id liamkazzopard/swim360-dive \\
        --swimmer P003 --local_dir [name of local folder]

    # Just one swimmer, one camera angle
    python fetch_from_hf.py --repo_id liamkazzopardi/swim360-dive \\
        --swimmer P003 --camera_angle Above_Water --local_dir [name of local folder]

    # Raw pattern: only the registry JSONs, no images
    python fetch_from_hf.py --repo_id liamkazzopardi/swim360-dive \\
        --allow_patterns "data/**/*.json" --local_dir [name of local folder]

Now Requires: pip install -U huggingface_hub --break-system-packages
Auth (only needed for private repos): `hf auth login`, HF_TOKEN, or --token.
-------------------------------------------------------------
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface_hub is not installed. Run:\n"
          "  pip install -U huggingface_hub --break-system-packages",
          file=sys.stderr)
    sys.exit(1)


def build_convenience_pattern(prefix: str, swimmer: str | None,
                              camera_angle: str | None, stroke: str | None) -> str:
    parts = [prefix, swimmer or "*", camera_angle or "*", stroke or "*", "**"]
    return "/".join(parts)


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def summarize(local_dir: Path) -> tuple[int, int]:
    count, total = 0, 0
    for f in local_dir.rglob("*"):
        if f.is_file() and ".cache/huggingface" not in f.as_posix():
            count += 1
            total += f.stat().st_size
    return count, total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download files from a Hugging Face dataset repo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo_id", required=True,
                        help="Source dataset repo, e.g. 'yourname/swim360-dive'.")
    parser.add_argument("--local_dir", type=Path, default=Path("./hf_data"),
                        help="Where files are written, mirroring the repo layout.")
    parser.add_argument("--revision", default="main",
                        help="Branch, tag, or commit hash to download from.")

    # Convenience filter matching the <swimmer>/<camera_angle>/<stroke> layout
    parser.add_argument("--prefix", default="blurred_images",
                        help="Top-level repo folder to scope the convenience "
                             "filter to (ignored if --allow_patterns is set).")
    parser.add_argument("--swimmer", default=None)
    parser.add_argument("--camera_angle", default=None,
                        help="e.g. Above_Water / Under_Water")
    parser.add_argument("--stroke", default=None)

    # Raw pattern override
    parser.add_argument("--allow_patterns", nargs="+", default=None,
                        help="Raw glob pattern(s); overrides the "
                             "--swimmer/--camera_angle/--stroke/--prefix filter.")
    parser.add_argument("--ignore_patterns", nargs="+", default=None,
                        help="Glob pattern(s) to exclude.")

    parser.add_argument("--token", default=None,
                        help="HF token. Defaults to HF_TOKEN env var or cached "
                             "`hf auth login` credentials. Only needed for "
                             "private repos.")
    args = parser.parse_args()

    if args.allow_patterns:
        allow_patterns = args.allow_patterns
    elif args.swimmer or args.camera_angle or args.stroke:
        allow_patterns = [build_convenience_pattern(
            args.prefix, args.swimmer, args.camera_angle, args.stroke)]
    else:
        allow_patterns = None  # everything

    print("=" * 60)
    print(f"HF Dataset Fetch <- {args.repo_id}")
    print("=" * 60)
    print(f"Local dir       : {args.local_dir}")
    print(f"Revision        : {args.revision}")
    print(f"Allow patterns  : {allow_patterns or '(all files)'}")
    print(f"Ignore patterns : {args.ignore_patterns or '(none)'}")
    print()

    args.local_dir.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(args.local_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=args.ignore_patterns,
        token=args.token,
    )

    count, total = summarize(Path(path))
    print("=" * 60)
    print("Finished.")
    print(f"Files on disk under {path}: {count} ({human_size(total)})")
    print("=" * 60)


if __name__ == "__main__":
    main()