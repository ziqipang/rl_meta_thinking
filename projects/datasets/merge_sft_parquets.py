from __future__ import annotations

from argparse import ArgumentParser
import os
from typing import Iterable

import pandas as pd


def _load_concat(parquet_files: Iterable[str]) -> pd.DataFrame:
    dataframes: list[pd.DataFrame] = []
    for path in parquet_files:
        dataframes.append(pd.read_parquet(path))
    if not dataframes:
        raise ValueError("No parquet files provided.")
    return pd.concat(dataframes, ignore_index=True)


def main() -> None:
    parser = ArgumentParser(description="Merge multiple SFT parquet files (prompt/response) into a single train/test pair.")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--train_files", type=str, nargs="+", required=True)
    parser.add_argument("--val_files", type=str, nargs="+", required=True)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle rows after concatenation.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_df = _load_concat(args.train_files)
    val_df = _load_concat(args.val_files)

    if args.shuffle:
        train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        val_df = val_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    train_path = os.path.join(args.out_dir, "train.parquet")
    val_path = os.path.join(args.out_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Wrote {len(train_df)} rows to {train_path}")
    print(f"Wrote {len(val_df)} rows to {val_path}")


if __name__ == "__main__":
    main()

