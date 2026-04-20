from __future__ import annotations

import argparse
from pathlib import Path

from fold_consistency_checks import build_context, run_all_checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run fold-consistency checks on one dataset configuration."
    )
    parser.add_argument("dataset_config", type=Path, help="Path to the dataset YAML configuration.")
    parser.add_argument(
        "--skip-prepare-data",
        action="store_true",
        help="Do not call prepare_data(); only inspect the existing cache.",
    )
    args = parser.parse_args()

    ctx = build_context(args.dataset_config, build_cache=not args.skip_prepare_data)
    results = run_all_checks(ctx)

    print(f"Dataset config: {args.dataset_config}")
    print(f"Cache dir: {ctx.cache_dir}")
    print("Checks:")
    for name, verdict in results.items():
        print(f"  [{verdict}] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
