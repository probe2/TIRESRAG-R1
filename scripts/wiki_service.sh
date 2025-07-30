# 这里我假定了所有的index都是ivf量化的
#GPU
CUDA_VISIBLE_DEVICES=0 python -u ../retrieval/wiki_corpus_load.py kilt 5004 
#CPU
# python -u ../retrieval/wiki_corpus_load_cpu.py kilt 5004 
