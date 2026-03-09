#!/usr/bin/env bash
set -euo pipefail
set -x

bash projects/datasets_v3/pipelines/stage0/create_bijective_problems.sh
bash projects/datasets_v3/pipelines/stage0/create_nonbijective_problems.sh
bash projects/datasets_v3/pipelines/stage0/rejection_sampling.sh
bash projects/datasets_v3/pipelines/stage1/rft_bijective.sh
bash projects/datasets_v3/pipelines/stage1/rft_nonbijective.sh
