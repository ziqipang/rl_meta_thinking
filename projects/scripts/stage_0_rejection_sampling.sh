set -x

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=data/numerical_task_v2/stage1_level1/train.parquet \
    data.prompt_key=prompt \
    data.n_samples=10 \
    data.output_path=data/numerical_task_v2/stage1_level1/rollout.parquet \
    data.batch_size=4096 \
    model.path=disk/models/Qwen2.5-7B-Instruct \
    +model.trust_remote_code=True \
    rollout.name=vllm \
    rollout.temperature=1.0 \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.prompt_length=1024 \
    rollout.response_length=8192 \
    rollout.max_num_batched_tokens=9216 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8


python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=data/numerical_task_v2/stage1_level1_nb_v2/train.parquet \
    data.prompt_key=prompt \
    data.n_samples=10 \
    data.output_path=data/numerical_task_v2/stage1_level1_nb_v2/rollout.parquet \
    data.batch_size=4096 \
    model.path=disk/models/Qwen2.5-7B-Instruct \
    +model.trust_remote_code=True \
    rollout.name=vllm \
    rollout.temperature=1.0 \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.prompt_length=1024 \
    rollout.response_length=8192 \
    rollout.max_num_batched_tokens=9216 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8