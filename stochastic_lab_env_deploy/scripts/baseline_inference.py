"""CLI wrapper for the stochastic lab heuristic baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stochastic_lab_env.baseline import DEFAULT_SEEDS, run_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=DEFAULT_SEEDS,
        help="Deterministic seeds used for baseline evaluation.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "outputs"
        / "evals"
        / "baseline_scores.json",
        help="Where to write the JSON summary.",
    )
    args = parser.parse_args()

    summary = run_baseline(args.seeds, save_path=args.save_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
