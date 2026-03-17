#!/usr/bin/env bash
# Wait for fwd10 experiment to finish, then launch fwd20.
set -euo pipefail

FWD10_PID=$1
echo "Waiting for fwd10 (PID=$FWD10_PID) to finish..."

while kill -0 "$FWD10_PID" 2>/dev/null; do
    sleep 60
done

echo "fwd10 finished. Checking exit status..."
wait "$FWD10_PID" 2>/dev/null || true

echo "Launching fwd20 experiment..."
cd /home/ubuntu/rl_meta_thinking
bash projects/datasets_v3_mix/pipelines/code_in_prompt/rl_both_fwd20.sh
