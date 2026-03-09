Stage-organized runnable shell pipelines.

- `stage0/`: dataset creation and rejection-sampling rollouts
- `stage1/`: RFT data prep, merged SFT training, and stable model promotion
- `stage2/`: RL data prep and RL training

Stage 2 also includes the shared multi-response "allow fewer answers" path:

- `stage2/prepare_response_with_none_data.sh`: build the v3 bijective/non-bijective datasets plus eval sets
- `stage2/rl_response_with_none.sh`: launch the mixed RL run on those datasets
- `stage2/stage_2_response_with_none.sh`: one-shot wrapper that prepares data and launches RL
