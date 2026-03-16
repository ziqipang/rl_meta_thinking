#!/usr/bin/env python3
"""Downsample bijective/non-bijective parquets to create 3:1 and 1:3 ratio splits.

Does NOT overwrite original train.parquet files.  Creates new files under
  data/numerical_task_v3_mix_code_in_prompt/multi_response_with_none/
    bijective_v3/train_downsampled_16667.parquet
    nonbijective_v2_balanced/train_downsampled_16667.parquet
"""

import pandas as pd
import pathlib

DATA_ROOT = pathlib.Path("data/numerical_task_v3_mix_code_in_prompt/multi_response_with_none")

DOWNSAMPLE_N = 16667  # 50000 / 3 ≈ 16667

def main():
    bij_path = DATA_ROOT / "bijective_v3" / "train.parquet"
    nb_path = DATA_ROOT / "nonbijective_v2_balanced" / "train.parquet"

    bij = pd.read_parquet(bij_path)
    nb = pd.read_parquet(nb_path)
    print(f"Original bijective: {len(bij)}")
    print(f"Original non-bijective: {len(nb)}")

    # Downsample bijective (for 1:3 bij:nb experiment)
    bij_down = bij.sample(n=DOWNSAMPLE_N, random_state=42).reset_index(drop=True)
    bij_out = DATA_ROOT / "bijective_v3" / f"train_downsampled_{DOWNSAMPLE_N}.parquet"
    bij_down.to_parquet(bij_out)
    print(f"Wrote {len(bij_down)} rows -> {bij_out}")

    # Downsample non-bijective (for 3:1 bij:nb experiment)
    nb_down = nb.sample(n=DOWNSAMPLE_N, random_state=42).reset_index(drop=True)
    nb_out = DATA_ROOT / "nonbijective_v2_balanced" / f"train_downsampled_{DOWNSAMPLE_N}.parquet"
    nb_down.to_parquet(nb_out)
    print(f"Wrote {len(nb_down)} rows -> {nb_out}")

    # Verify ratios
    print(f"\n3:1 bij:nb experiment -> {len(bij)} bij + {len(nb_down)} nb = {len(bij)+len(nb_down)} total")
    print(f"  ratio = {len(bij)/len(nb_down):.2f}:1")
    print(f"1:3 bij:nb experiment -> {len(bij_down)} bij + {len(nb)} nb = {len(bij_down)+len(nb)} total")
    print(f"  ratio = 1:{len(nb)/len(bij_down):.2f}")

if __name__ == "__main__":
    main()
