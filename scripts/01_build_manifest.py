"""Thin wrapper: python scripts/01_build_manifest.py --config configs/pilot.yaml [...] == clove manifest."""
import sys

from clove.cli import main

if __name__ == "__main__":
    sys.exit(main(["manifest", *sys.argv[1:]]))
