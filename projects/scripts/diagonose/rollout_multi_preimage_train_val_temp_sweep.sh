set -euo pipefail
set -x

# Diagnose train/val mismatch by rolling out the same checkpoint on:
#   - train (random subset, matched to val size) @ temperature 0.0 / 1.0
#   - val (full)                              @ temperature 0.0 / 1.0
#
# Outputs:
#   - Parquets: ${OUT_DIR}/rollout_{train|val}_temp{0p0|1p0}.parquet
#   - HTMLs:    ${OUT_DIR}/rollout_{train|val}_temp{0p0|1p0}.html

MODEL_PATH="disk/checkpoints/numerical-task-rl/stage2-rl-level1to2-qwen2.5-7b-instruct-nb-v2-multi-preimage-k2-balanced/global_step_300/actor/huggingface"
TRAIN_PARQUET="${TRAIN_PARQUET:-data/numerical_task/stage2_rl_level1_multi_preimage_nb_v2_balanced/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-data/numerical_task/stage2_rl_level1_multi_preimage_nb_v2_balanced/test.parquet}"

OUT_DIR="${OUT_DIR:-outputs/diagonose/multi_preimage_trainval_temp_sweep_global_step_300}"
SEED="${SEED:-0}"

NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

PROMPT_KEY="${PROMPT_KEY:-prompt}"
DATA_SOURCE_KEY="${DATA_SOURCE_KEY:-data_source}"
REWARD_MODEL_KEY="${REWARD_MODEL_KEY:-reward_model}"

PROMPT_LEN="${PROMPT_LEN:-1024}"
RESPONSE_LEN="${RESPONSE_LEN:-8192}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-9216}"
N_SAMPLES="${N_SAMPLES:-2}"

mkdir -p "${OUT_DIR}"

TRAIN_SUBSET="${OUT_DIR}/train_subset_like_val.parquet"
python3 projects/scripts/diagonose/sample_parquet_subset.py \
  --train "${TRAIN_PARQUET}" \
  --val "${VAL_PARQUET}" \
  --out "${TRAIN_SUBSET}" \
  --seed "${SEED}"

for SPLIT in train val; do
  if [[ "${SPLIT}" == "train" ]]; then
    DATA_PATH="${TRAIN_SUBSET}"
  else
    DATA_PATH="${VAL_PARQUET}"
  fi

	  for TEMP in 0.0 1.0; do
	    CUR_N_SAMPLES="${N_SAMPLES}"
	    if [[ "${TEMP}" == "0.0" ]]; then
	      # verl/trainer/main_generation.py enforces greedy decoding to generate a single sample.
	      CUR_N_SAMPLES=1
	    fi

	    TAG="temp${TEMP}"
	    TAG="${TAG//./p}"

    OUT_PARQUET="${OUT_DIR}/rollout_${SPLIT}_${TAG}.parquet"
    OUT_HTML="${OUT_DIR}/rollout_${SPLIT}_${TAG}.html"

	    python3 -m verl.trainer.main_generation \
      trainer.nnodes="${NNODES}" \
      trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
      data.path="${DATA_PATH}" \
      data.prompt_key="${PROMPT_KEY}" \
      data.data_source_key="${DATA_SOURCE_KEY}" \
      data.reward_model_key="${REWARD_MODEL_KEY}" \
	      data.n_samples="${CUR_N_SAMPLES}" \
      data.batch_size="${GEN_BATCH_SIZE}" \
      data.output_path="${OUT_PARQUET}" \
      data.overwrite=True \
      model.path="${MODEL_PATH}" \
      +model.trust_remote_code=True \
      rollout.name=vllm \
      rollout.temperature="${TEMP}" \
      rollout.top_k=-1 \
      rollout.top_p=1.0 \
      rollout.prompt_length="${PROMPT_LEN}" \
      rollout.response_length="${RESPONSE_LEN}" \
      rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
      rollout.tensor_model_parallel_size=1 \
      rollout.gpu_memory_utilization="${GPU_MEM_UTIL}"

	    python3 projects/rollout_to_html.py "${OUT_PARQUET}" \
	      -o "${OUT_HTML}" \
	      --metrics \
	      --max-responses "${CUR_N_SAMPLES}"
	  done
	done

echo "Done. Outputs in: ${OUT_DIR}"
