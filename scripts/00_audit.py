"""Thin wrapper: python scripts/00_audit.py --config configs/pilot.yaml [...] == clove audit."""
import sys

from clove.cli import main

if __name__ == "__main__":
    sys.exit(main(["audit", *sys.argv[1:]]))
