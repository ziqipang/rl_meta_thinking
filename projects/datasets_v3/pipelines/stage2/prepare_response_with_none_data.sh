#!/usr/bin/env bash
set -euo pipefail
set -x

bash projects/datasets_v3/pipelines/stage2/response_with_none_create_bijective_problems.sh
bash projects/datasets_v3/pipelines/stage2/response_with_none_create_nonbijective_problems.sh
bash projects/datasets_v3/pipelines/stage2/inverse_create_problems.sh
bash projects/datasets_v3/diverse_evals/make_eval_more_candidates.sh
bash projects/datasets_v3/diverse_evals/make_eval_long_outputs.sh
bash projects/datasets_v3/diverse_evals/make_eval_long_outputs_more_candidates.sh
bash projects/datasets_v3/diverse_evals/make_eval_lenchange_functions.sh
