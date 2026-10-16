"""Thin wrapper: python scripts/03_run_pilot.py --config configs/pilot.yaml [...] == clove run."""
import sys

from clove.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
