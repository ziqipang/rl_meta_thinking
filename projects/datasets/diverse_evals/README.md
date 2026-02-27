# Diverse eval generators (non-bijective inversion)

This folder contains *evaluation-only* dataset generators derived from:
- `projects/datasets/create_string_data.py`
- `projects/datasets/create_string_data_multi.py`

All scripts default to the `nonbijective_v2` **balanced** function split.

## 1) Longer outputs (y length 6–10)

```bash
bash projects/datasets/diverse_evals/make_eval_long_outputs.sh
```

## 2) More candidates (k in 3–6)

```bash
bash projects/datasets/diverse_evals/make_eval_more_candidates.sh
```

## 3) Long outputs + more candidates

```bash
bash projects/datasets/diverse_evals/make_eval_long_outputs_more_candidates.sh
```

## 4) Novel length-changing non-bijective functions (k in 3–5, y length 6–10)

```bash
bash projects/datasets/diverse_evals/make_eval_lenchange_functions.sh
```

