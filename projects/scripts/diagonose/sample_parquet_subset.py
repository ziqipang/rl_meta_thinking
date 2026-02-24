#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path


def _load_parquet(path: Path):
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: datasets. Install dependencies (e.g. `pip install -r requirements.txt`)."
        ) from e
    return load_dataset("parquet", data_files=str(path))["train"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample a random subset of a train parquet to match a val parquet size.")
    parser.add_argument("--train", type=Path, required=True, help="Path to train parquet.")
    parser.add_argument("--val", type=Path, required=True, help="Path to val/test parquet (used to match size).")
    parser.add_argument("--out", type=Path, required=True, help="Output parquet path for sampled subset.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Override number of samples (default: len(val)).",
    )
    args = parser.parse_args()

    if not args.train.exists():
        raise SystemExit(f"Train parquet not found: {args.train}")
    if not args.val.exists():
        raise SystemExit(f"Val parquet not found: {args.val}")

    train_ds = _load_parquet(args.train)
    val_ds = _load_parquet(args.val)

    target_n = int(args.n) if args.n is not None else len(val_ds)
    if target_n <= 0:
        raise SystemExit(f"Invalid target sample count: {target_n}")

    train_n = len(train_ds)
    sample_n = min(target_n, train_n)

    rng = random.Random(int(args.seed))
    if sample_n == train_n:
        indices = list(range(train_n))
    else:
        indices = rng.sample(range(train_n), k=sample_n)

    subset = train_ds.select(indices)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(str(args.out))

    print(f"Wrote: {args.out}")
    print(f"Subset size: {sample_n} (train={train_n}, val={len(val_ds)}, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
