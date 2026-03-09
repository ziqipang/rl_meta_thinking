from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = ArgumentParser(description="Sample a fractional subset from a parquet file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, required=True, help="Fraction of rows to keep, in (0, 1].")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle after sampling.")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input parquet not found: {args.input}")
    if not (0.0 < args.fraction <= 1.0):
        raise SystemExit(f"Invalid fraction {args.fraction}; expected 0 < fraction <= 1.")

    df = pd.read_parquet(args.input)
    if len(df) == 0:
        subset_df = df
    else:
        sample_n = max(1, int(len(df) * args.fraction))
        sample_n = min(sample_n, len(df))
        subset_df = df.sample(n=sample_n, random_state=args.seed)
        if args.shuffle:
            subset_df = subset_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        else:
            subset_df = subset_df.reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_parquet(args.output, index=False)

    print(f"Wrote {len(subset_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
