
CUDA_VISIBLE_DEVICES=2,3 python -u -m flashrag.retriever.index_builder \
    --retrieval_method bge \
    --model_path /mnt/ceph_rbd/model/bge-large-en-v1.5 \
    --corpus_path /mnt/ceph_rbd/TIRESRAG-R1/corpus/kilt/kilt_knowledgesource_r1_search100.jsonl \
    --save_dir /mnt/ceph_rbd/TIRESRAG-R1/corpus/kilt/chunk_100_bge_en_v1.5 \
    --max_length 300 \
    --batch_size 2048 \
    --pooling_method cls \
    --faiss_type Flat \
    --save_embedding #\