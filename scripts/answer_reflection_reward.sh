export PYTHONPATH=$PYTHONPATH:/TIRESRAG-R1/OpenRLHF-RAG
python -u ../reward/answer_reflection_reward.py --template_type qwen_chat --data_path /TIRESRAG-R1/data/training_set/stage_2.jsonl --reward_pretrain /mnt/ceph_rbd/model/Qwen2.5-1.5B --log_file /TIRESRAG-R1/results/qwen.jsonl --port 1278
