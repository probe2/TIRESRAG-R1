
export CUDA_HOME=/mnt/ceph_rbd/conda_env/cu124
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
echo "[INFO] CUDA_HOME=$CUDA_HOME"
if ! nvcc --version | grep -q "release 12.4"; then
    echo "[ERROR] nvcc not found or CUDA version mismatch"
    exit 1
fi
export NUMEXPR_MAX_THREADS=128
export RAY_DEDUP_LOGS=0
export DEBUG_MODE=1
export NCCL_TIMEOUT=7200
export RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=1
# Your wandb token
wandb_token=
# sudo rm -rf ~/.netrc
export WANDB_API_KEY=xx

# Path of training data
DATA_PATH=/TIRESRAG-R1/data/training_set/stage_2.jsonl
#
# Path of backbone model(DeepSeek-R1-Distill-Qwen-1.5B)
TOKENIZER_PATH=/mnt/ceph_rbd/model/Qwen2.5-3B-Instruct 
export CUDA_VISIBLE_DEVICES=1,2,3
export PYTHONPATH=TIRESRAG-R1/OpenRLHF-RAG
N_SAMPLES=5
EPISODE=2
WARMUP=0.015
TBS=112 
RBS=112 
KL=0 #0.001
LR=2e-6
MAX_LENGTH=15000
PORT=1278
TEMP=2.0

SAVE_MODEL_NAME=qwen_grpo

GROUP_METHOD=normal

LOG_BASE=log

mkdir -p /TIRESRAG-R1/results/$SAVE_MODEL_NAME
mkdir -p /TIRESRAG-R1/results/ckpts
mkdir -p /TIRESRAG-R1/results/$SAVE_MODEL_NAME/server
mkdir -p $LOG_BASE/server/

ray stop
HEAD_NODE_IP=127.0.0.1 


ray start --head --node-ip-address ${HEAD_NODE_IP} --num-gpus 3 --port 8266  --include-dashboard true --dashboard-port 8267 

sleep 5 #               -

ray job submit --address="http://127.0.0.1:8267" \
   --verbose \
   -- python3 -u -m  openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.5 \
   --colocate_actor_ref \
   --pretrain ${TOKENIZER_PATH} \
   --remote_rm_url http://localhost:${PORT}/get_reward \
   --save_path /TIRESRAG-R1/results/ckpts/$SAVE_MODEL_NAME \
   --ckpt_path /TIRESRAG-R1/results/ckpts/$SAVE_MODEL_NAME \
   --micro_train_batch_size 4 \
   --train_batch_size ${TBS} \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size ${RBS} \
   --advantage_estimator group_norm \
   --max_samples 100000 \
   --max_epochs 1 \
   --num_episodes ${EPISODE} \
   --lr_warmup_ratio ${WARMUP} \
   --n_samples_per_prompt $N_SAMPLES \
   --prompt_max_len 1024 \
   --generate_max_len $MAX_LENGTH \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate $LR \
   --critic_learning_rate 9e-6 \
   --init_kl_coef $KL \
   --prompt_data $DATA_PATH \
   --input_key question \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 20 \
   --vllm_sync_backend nccl \
   --max_ckpt_num 3 \
   --temperature $TEMP \
   --overlap_comm \
   --packing_samples \
   --use_kl_loss \
   --use_kl_estimator_k3 \
   --apply_chat_template \
   --enable_accuracy_filter \
   --accuracy_lower_bound 0.1 \
   --accuracy_upper_bound 0.9 \
   --use_wandb ${wandb_token} \
   --wandb_run_name $SAVE_MODEL_NAME
