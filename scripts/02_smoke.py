"""Thin wrapper: python scripts/02_smoke.py --config configs/pilot.yaml [...] == clove smoke."""
import sys

from clove.cli import main

if __name__ == "__main__":
    sys.exit(main(["smoke", *sys.argv[1:]]))
