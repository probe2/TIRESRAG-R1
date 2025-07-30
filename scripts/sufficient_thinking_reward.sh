#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
MODEL_PATH="/mnt/ceph_rbd/model/Qwen3-8B"
PORT=5006

if [ ! -z "$1" ]; then
  MODEL_PATH="$1"
fi

if [ ! -z "$2" ]; then
  PORT="$2"
fi

echo "Using model path: $MODEL_PATH"
echo "Using port: $PORT"

export PORT=$PORT

python3 -u ../reward/sufficient_vllm_server_question_context_answer_and_thinking.py --model_path "$MODEL_PATH"