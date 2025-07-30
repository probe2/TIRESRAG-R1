
from socket import ntohl
from flask import Flask, request, jsonify
import faiss
import numpy as np
from FlagEmbedding import FlagModel
import os
import time
import sys
import argparse
import json 
from tqdm import tqdm
import os
import sys
import time
import psutil
import threading
from datetime import datetime
import torch 
import time
import gc 
import threading

search_lock = threading.Lock()

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)  # 转换为GB

def format_size(size_bytes):
    """格式化文件大小显示"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def loading_monitor():
    """监控加载进度和资源使用"""
    start_memory = get_memory_usage()
    start_time = time.time()
    
    while loading:
        current_memory = get_memory_usage()
        elapsed_time = time.time() - start_time
        memory_change = current_memory - start_memory
        
        sys.stdout.write('\r' + ' ' * 80 + '\r')  # 清除当前行
        sys.stdout.write(
            f"加载中... 已用时: {elapsed_time:.1f}秒 | "
            f"内存使用: {current_memory:.1f}GB (增加: {memory_change:+.1f}GB) | "
            f"{datetime.now().strftime('%H:%M:%S')}"
        )
        sys.stdout.flush()
        time.sleep(1)
def read_index_with_monitor(index_path):
    """带监控的索引加载函数"""
    global loading
    loading = True
    # 检查是否存在预训练的IVF索引
    ivf_index_path = index_path 
    if "ivf"  in index_path.lower(): #os.path.exists(ivf_index_path):
        print(f"\n发现预训练的IVF索引，直接加载: {ivf_index_path}")
        try:
            cpu_ivf_index = faiss.read_index(ivf_index_path)
            print("预训练索引加载成功！")
            nlist = int(np.sqrt(cpu_ivf_index.ntotal))  # 确保nlist在所有路径中都定义
        except Exception as e:
            print(f"预训练索引加载失败: {e}，将重新训练索引")
            cpu_ivf_index = None
    elif "flat" in index_path.lower():
        cpu_ivf_index = faiss.read_index(index_path)
        nlist = int(np.sqrt(cpu_ivf_index.ntotal))  # 确保nlist在所有路径中都定义

    else:
        cpu_ivf_index = None

    # 获取文件信息
    file_size = os.path.getsize(index_path)
    print(f"\n开始加载索引文件:")
    print(f"文件路径: {index_path}")
    print(f"文件大小: {format_size(file_size)}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"初始内存使用: {get_memory_usage():.1f}GB")
    print("\n加载进度:")

    # 启动监控线程
    monitor_thread = threading.Thread(target=loading_monitor)
    monitor_thread.daemon = True
    monitor_thread.start()

    try:
        # 加载索引
        start_time = time.time()
        print("1. 加载CPU索引...")
        
        # 读取基础索引
        if cpu_ivf_index is None:
            cpu_index = faiss.read_index(index_path)
            dimension = cpu_index.d
            ntotal = cpu_index.ntotal
            # 配置IVF参数
            nlist = int(np.sqrt(ntotal))  # 聚类中心数量，根据数据量自动调整
            print(f"\n2. 配置IVF索引 (nlist={nlist})...")

            res = faiss.StandardGpuResources()
            gpu_quantizer = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(dimension))

            # 创建新的IVF索引
            # quantizer = faiss.IndexFlatL2(dimension)
            gpu_ivf_index = faiss.IndexIVFFlat(gpu_quantizer, dimension, nlist)
            # 训练索引
            print("\n 3. 训练IVF索引...")
            train_size = min(500000, ntotal)  # 最多使用50万个向量训练

            # 收集训练数据
            train_vectors = []
            train_indices = np.random.choice(ntotal, train_size, replace=False)
            # 确保 train_indices[0] 不大于 ntotal - train_size
            while train_indices[0] > ntotal - train_size - 1:
                train_indices = np.random.choice(ntotal, train_size, replace=False)

            if hasattr(cpu_index, 'reconstruct_n'):
                print('使用reconstruct_batch批量获取向量1')
                train_vectors = cpu_index.reconstruct_n(
                    int(train_indices[0]), 
                    len(train_indices)
                )
            else:
                train_vectors = np.vstack([
                    cpu_index.reconstruct(int(idx)).reshape(1, -1)
                    for idx in train_indices
                ])
            # for idx in train_indices:
            #     if hasattr(cpu_index, 'reconstruct'):
            #         vec = cpu_index.reconstruct(int(idx))
            #         train_vectors.append(vec)
            
            # train_vectors = np.array(train_vectors)
            gpu_ivf_index.train(train_vectors)
            del train_vectors  # 释放训练数据内存
            gc.collect()
            # 将索引转回CPU进行添加操作

            print("5. 将索引转回CPU进行向量添加...")
            cpu_ivf_index = faiss.index_gpu_to_cpu(gpu_ivf_index)
            del gpu_ivf_index  # 释放GPU索引内存
            del res
            gc.collect()
            print("4. 分批添加向量到IVF索引...")
            batch_size = 2000000
            for start_idx in tqdm(range(0, ntotal, batch_size)):
                end_idx = min(start_idx + batch_size, ntotal)
                
                # 使用reconstruct_batch批量获取向量
                if hasattr(cpu_index, 'reconstruct_n'):
                    print('使用reconstruct_batch批量获取向量')
                    batch_vectors = cpu_index.reconstruct_n(start_idx, end_idx - start_idx)
                else:
                    # 降级方案：如果没有reconstruct_batch方法
                    batch_vectors = np.vstack([
                        cpu_index.reconstruct(int(i)).reshape(1, -1)
                        for i in range(start_idx, end_idx)
                    ])
                
                # 添加批次到索引
                cpu_ivf_index.add(batch_vectors)
                del batch_vectors  # 释放批次向量内存     
                gc.collect()   
            # 释放原始索引内存

            # 保存CPU版本的索引
            print("\n7. 保存训练好的IVF索引...")
            faiss.write_index(cpu_ivf_index, ivf_index_path)
            print(f"索引已保存到: {ivf_index_path}")

            print(f"当前内存使用: {get_memory_usage():.1f}GB")

            del cpu_index
            gc.collect()
            print(f"当前内存使用: {get_memory_usage():.1f}GB")

        print("\n5. 准备GPU资源...")
        #创建GPU资源
        gpu_resources = []
        for i in [0, 1, 2]:  # 使用GPU 0和1
            res = faiss.StandardGpuResources()
            gpu_resources.append(res)

        print("\n6. 配置GPU选项...")
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True  # 在GPU间分片
        co.usePrecomputed = True  # 不预计算,节省gpu可以

        print("\n7. 将IVF索引转移到多GPU...")
        gpu_list = [i for i in range(torch.cuda.device_count())]
        gpu_index = faiss.index_cpu_to_gpu_multiple_py(
            gpu_resources,  # GPU资源列表
            gpus = gpu_list,        # GPU IDs
            index = cpu_ivf_index,     # CPU索引
            co = co            # GPU选项
        )
        # gpu_index = faiss.index_cpu_to_all_gpus(cpu_ivf_index, co) #直接转移到可用的gpu资源上
        del cpu_ivf_index
        gc.collect()
        # 设置搜索参数
        if("ivf" in index_path):
            nprobe = min(500, nlist//10)  # 默认检查10%的聚类中心，但不超过30
            gpu_index.nprobe = nprobe
            print(f"\n8. 设置搜索参数 (nprobe={nprobe})")

        loading = False
        monitor_thread.join()
        
        # 计算最终状态
        end_time = time.time()
        total_time = end_time - start_time
        final_memory = get_memory_usage()
        
        # 打印完成信息
        print("\n\n加载完成:")
        print(f"总用时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
        print(f"最终内存使用: {final_memory:.1f}GB")
        print(f"索引配置:")
        print(f"- 向量总数: {gpu_index.ntotal}")
        print(f"- 向量维度: {gpu_index.d}")
        print(f"- 聚类中心数(nlist): {nlist}")
        # print(f"- 搜索聚类数(nprobe): {nprobe}")
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 打印GPU使用情况
        for i in gpu_list:
            print(f"\nGPU {i} 显存使用: {torch.cuda.memory_allocated(i)/1024**3:.2f}GB")

          # 释放CPU索引内存
        return gpu_index

    except Exception as e:
        loading = False
        monitor_thread.join()
        print(f"\n\n加载失败: {str(e)}")
        raise


def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        corpus = []
        total = sum(1 for _ in file)
        file.seek(0)    
        for line in tqdm(file, total=total, desc="Loading corpus"):
            data = json.loads(line)
            corpus.append(data['contents'])
    return corpus

# 创建 Flask 应用
app = Flask(__name__)

@app.route("/queries", methods=["POST"])
def query():
    # 从请求中获取查询向量
    data = request.json
    
    embed_start = time.time()
    print("\n1. 生成查询向量...")

    queries = data["queries"]

    k = data.get("k", 3)

    # s = time.time()
    query_embeddings = model.encode_queries(queries)

    if isinstance(query_embeddings, torch.Tensor):
        query_embeddings = query_embeddings.detach().cpu().numpy()

    query_embeddings = query_embeddings.astype('float32')
    print(f"向量生成完成，用时: {time.time() - embed_start:.2f}秒")
    print(f"向量形状: {query_embeddings.shape}")

    all_answers = []
    search_start = time.time()
    print("\n2. 搜索相似文档...")

    # 添加批处理逻辑
    batch_size = 16  # 可根据GPU内存调整这个值     
    total_queries = query_embeddings.shape[0]
    all_D = []
    all_I = []

    for i in range(0, total_queries, batch_size):
        end_idx = min(i + batch_size, total_queries)
        batch_queries = query_embeddings[i:end_idx]
        
        # 添加错误处理
        try:
            with search_lock:
                batch_D, batch_I = index.search(batch_queries, k=k)
           # batch_D, batch_I = index.search(batch_queries, k=k)
            all_D.append(batch_D)
            all_I.append(batch_I)
        except Exception as e:
            print(f"批处理搜索错误 (索引 {i}~{end_idx}): {str(e)}")
            # 降级到CPU搜索作为备选方案
            try:
                print("尝试使用CPU进行搜索...")
                cpu_index = faiss.index_gpu_to_cpu(index)
                batch_D, batch_I = cpu_index.search(batch_queries, k=k)
                all_D.append(batch_D)
                all_I.append(batch_I)
            except Exception as e2:
                print(f"CPU搜索也失败: {str(e2)}")
                # 填充-1表示搜索失败
                batch_D = np.full((end_idx - i, k), -1.0, dtype=np.float32)
                batch_I = np.full((end_idx - i, k), -1, dtype=np.int64)
                all_D.append(batch_D)
                all_I.append(batch_I)

    # 合并批处理结果
    if all_D and all_I:
        D = np.vstack(all_D) if len(all_D) > 1 else all_D[0]
        I = np.vstack(all_I) if len(all_I) > 1 else all_I[0]
    else:
        # 如果所有批次都失败，创建一个全-1的结果
        D = np.full((total_queries, k), -1.0, dtype=np.float32)
        I = np.full((total_queries, k), -1, dtype=np.int64)

    print(f"搜索完成，用时: {time.time() - search_start:.2f}秒")
    # 添加结果检查
    valid_results = (I != -1).sum()
    total_expected = total_queries * k
    print(f"有效结果: {valid_results}/{total_expected} ({valid_results/total_expected*100:.1f}%)")

    for q_idx, idx in enumerate(I):
        print(f"查询 {q_idx+1}: 检索出的idx: {idx[:k]}")
        # 添加索引验证，确保不会因为无效索引而崩溃
        answers = []
        for i in idx[:k]:
            if i >= 0 and i < len(corpus):
                answers.append(corpus[i])
            else:
                answers.append("未找到匹配文档")
        all_answers.append(answers)

    return jsonify({"queries": queries, "answers": all_answers})


if __name__ == "__main__":
    data_type = sys.argv[1]
    port = sys.argv[2]
    print('开始搭建')
    model = FlagModel(
        "/mnt/ceph_rbd/model/bge-large-en-v1.5",
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        use_fp16=False,
        # device="cuda:1",
    )
    model.model = model.model.to("cuda:0")

    for name, param in model.model.named_parameters():
        print(f"参数 {name} 在设备: {param.device}")
        break  # 只打印第一个参数作为示例

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
    try:
        print("\n准备加载索引...")
        index = read_index_with_monitor(index_path)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
    print("索引已经建好")

    app.run(host="0.0.0.0", port=port, debug=False)  # 在本地监听端口5003
    print("可以开始查询")

