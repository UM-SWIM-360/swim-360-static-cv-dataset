#!/usr/bin/env python3
"""
run_pipeline.py
================
Single entry point that runs the pipeline scripts in-process via importlib,
loading each one by its EXACT file path (next to this script) rather than
relying on `python <name>.py` from the command line each time. This sidesteps
the class of bug we kept hitting: a stale/renamed/divergent copy of a script
getting run by accident, since everything here always loads from one fixed
location.


Usage
-----
python run_pipeline.py
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_step(script_name: str, argv: list[str]) -> None:
    script_path = HERE / script_name
    if not script_path.exists():
        print(f"ERROR: {script_path} not found -- is it actually in this folder?", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(f"RUNNING: {script_name} {' '.join(argv)}")
    print("=" * 60)

    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # defines everything in the file; does NOT run main()

    old_argv = sys.argv
    sys.argv = [str(script_path)] + argv
    try:
        module.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code != 0:
            print(f"\nPIPELINE STOPPED: {script_name} exited with code {code}.", file=sys.stderr)
            sys.exit(code)
    finally:
        sys.argv = old_argv

    print()


if __name__ == "__main__":
    # Edit these to match your real paths/flags.
    run_step("generate_config.py", [
        "--registry_root", ".",
        "--output", "dataset_config.yaml",
        "--train_ratio", "0.6", "--valid_ratio", "0.2", "--test_ratio", "0.2",
        "--force",  # overwrite any existing config
    ])

    run_step("compile_COCO.py", [
        "--registry_root", ".",
        "--dataset_config", "dataset_config.yaml",
        "--output_dir", "dataset",
        "--images_root", "blurred_images",
    ])

    
    print("PIPELINE COMPLETE")
    