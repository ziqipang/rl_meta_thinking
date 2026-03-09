#!/usr/bin/env bash
set -euo pipefail
set -x

bash projects/datasets_v3/pipelines/stage2/prepare_response_with_none_data.sh
bash projects/datasets_v3/pipelines/stage2/rl_response_with_none.sh
