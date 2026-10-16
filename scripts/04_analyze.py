"""Thin wrapper: python scripts/04_analyze.py --run-dir artifacts/runs/<RUN_ID> == clove analyze."""
import sys

from clove.cli import main

if __name__ == "__main__":
    sys.exit(main(["analyze", *sys.argv[1:]]))
