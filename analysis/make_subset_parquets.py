"""Sample 8 questions per (model, eval_set) and write subset parquets for HTML generation."""
import pandas as pd
import os
import random

random.seed(42)

models = {
    "both_model": "analysis/rollouts/both_model",
    "nb_only_model": "analysis/rollouts/nb_only_model",
}

eval_sets = [
    "nonbijective_test",
    "bijective_test",
    "nb_more_candidates",
    "nb_long_outputs",
]

N = 8
out_dir = "analysis/view_subset"

for model_name, model_dir in models.items():
    for eval_set in eval_sets:
        parquet_path = os.path.join(model_dir, f"{eval_set}.parquet")
        if not os.path.exists(parquet_path):
            print(f"SKIP (not found): {parquet_path}")
            continue

        df = pd.read_parquet(parquet_path)
        n_samples = min(N, len(df))
        indices = sorted(random.sample(range(len(df)), n_samples))
        subset = df.iloc[indices].reset_index(drop=True)

        out_path = os.path.join(out_dir, f"{model_name}_{eval_set}_subset.parquet")
        subset.to_parquet(out_path)
        print(f"Wrote {n_samples} samples: {out_path}")

# For consistent comparison, use the SAME 8 questions for both models on each eval set
# Re-do with shared indices
random.seed(42)
shared_indices = {}
for eval_set in eval_sets:
    # Use the both_model file to determine total count
    path = os.path.join("analysis/rollouts/both_model", f"{eval_set}.parquet")
    if not os.path.exists(path):
        continue
    df = pd.read_parquet(path)
    indices = sorted(random.sample(range(len(df)), min(N, len(df))))
    shared_indices[eval_set] = indices

print("\n--- Re-generating with shared indices for fair comparison ---")
for model_name, model_dir in models.items():
    for eval_set in eval_sets:
        if eval_set not in shared_indices:
            continue
        parquet_path = os.path.join(model_dir, f"{eval_set}.parquet")
        if not os.path.exists(parquet_path):
            print(f"SKIP (not found): {parquet_path}")
            continue

        df = pd.read_parquet(parquet_path)
        indices = shared_indices[eval_set]
        subset = df.iloc[indices].reset_index(drop=True)

        out_path = os.path.join(out_dir, f"{model_name}_{eval_set}_subset.parquet")
        subset.to_parquet(out_path)
        print(f"Wrote {len(indices)} samples: {out_path}")
