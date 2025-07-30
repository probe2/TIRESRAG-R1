
from flask import Flask, request, jsonify
import faiss
import numpy as np
from FlagEmbedding import FlagModel
import os
import time
import sys
import torch 
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        corpus = file.readlines()
        corpus = [line.strip("\n") for line in corpus]
    return corpus

# 创建 Flask 应用
app = Flask(__name__)

@app.route("/queries", methods=["POST"])
def query():
    # 从请求中获取查询向量
    data = request.json

    queries = data["queries"]

    k = data.get("k", 3)
    search_start = time.time()

    # s = time.time()
    print('打印查询,', queries)
    with open('query.txt', 'w') as f:
        for query in queries:
            f.write(query + '\n')
    query_embeddings = model.encode_queries(queries)

    if isinstance(query_embeddings, torch.Tensor):
        query_embeddings = query_embeddings.detach().cpu().numpy()
    query_embeddings = query_embeddings.astype('float32')

    all_answers = []
    D, I = index.search(query_embeddings, k=k)  # 假设返回前3个结果
    print(f"搜索完成，用时: {time.time() - search_start:.2f}秒")

    for idx in I:
        answers_for_query = [corpus[i] for i in idx[:k]] # 找出该query对应的k个答案
        all_answers.append(answers_for_query)  # 将该query的答案列表存储

    return jsonify({"queries": queries, "answers": all_answers})


if __name__ == "__main__":
    data_type = sys.argv[1]
    port = sys.argv[2]

    model = FlagModel(
        "/mnt/ceph_rbd/model/bge-large-en-v1.5",
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        use_fp16=False,
        multi_process=False
    )
    model.model = model.model.to("cuda:3")
    model.device = "cuda:3"
    # model.pool = None
    print("模型已加载完毕")

    # 加载语料库
    if data_type =="kilt":
        file_path ="/mnt/ceph_rbd/len_rag_local/corpus/kilt/kilt_knowledgesource_r1_search100.jsonl"
    corpus = load_corpus(file_path)

    print(f"语料库已加载完毕-{len(corpus)}")

    # 加载建好的索引
    if data_type =="kilt":
        index_path ="/mnt/ceph_rbd/len_rag_local/corpus/kilt/chunk_100_bge_en_v1.5/bge_Flat.index"
    # index = faiss.read_index(index_path)
    index = faiss.read_index(index_path)

    print("索引已经建好")
# /opt/aps/workdir/input/data/enwiki-20171001-pages-meta-current-withlinks-abstracts.tsv

    app.run(host="0.0.0.0", port=port, debug=False)  # 在本地监听端口5003
    print("可以开始查询")
