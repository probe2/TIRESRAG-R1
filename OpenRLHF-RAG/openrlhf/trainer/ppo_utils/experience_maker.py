import time
from abc import ABC
from collections import Counter

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import random
random.seed(42)
#sht update
import ray
import torch
from datasets import load_dataset
import torch.nn as nn
from tqdm import tqdm
from datasets import Dataset
import copy
import requests
import time
from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray
import re
from collections import defaultdict
logger = init_logger(__name__)


# new
import json
import requests
from bs4 import BeautifulSoup
import wikipediaapi
from urllib.parse import unquote
from urllib.request import urlopen
from urllib.parse import urlparse
import wikipedia
from requests.exceptions import Timeout
from tqdm import tqdm #这个没有
import time
import concurrent #这个没有
from concurrent.futures import ThreadPoolExecutor
import pdfplumber #这个没有
from io import BytesIO
import re
import string
from typing import Optional, Tuple
#from nltk.tokenize import sent_tokenize #没有，但也没用上
#import nltk #没有，但也没用上
from typing import List

import multiprocessing #没有
from openai import OpenAI
import sys
import os
from datasets import load_dataset
import http.client
from time import sleep
from collections import defaultdict
import random
import math



def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


# Update 1229 For GRPO
def conditional_cat(attr1, attr2):
    if attr1 is not None and attr2 is not None:
        if isinstance(attr1, torch.Tensor):
            op = lambda x, y: torch.cat((x, y), dim=0)
        else:
            op = lambda x, y: x + y
        return op(attr1, attr2)
    return None

def bool_mapping(s):
    """布尔值映射为yes/no"""
    if s == "True": return "yes"
    elif s == "False": return "no"
    else: return s

def exact_match_score(prediction, ground_truth):
    return normalize_answer(bool_mapping(prediction)) == normalize_answer(
        bool_mapping(ground_truth)
    )

def cover_exact_match_score_1(prediction, ground_truth):

    pre_list = normalize_answer(bool_mapping(prediction)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")

    # 不考虑顺序和连续
    return all(ground in pre_list for ground in ground_list)

def f1_score(prediction, ground_truth):
    """计算F1分数
    Args:
        prediction: 模型预测的答案
        ground_truth: 标准答案
    Returns:
        (f1, precision, recall): F1分数及其组成部分
    """
    # 标准化处理预测和真实答案
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = 0

    # 特殊情况处理：yes/no/noanswer答案
    if (normalized_prediction in ["yes", "no", "noanswer"] and 
        normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC
    if (normalized_ground_truth in ["yes", "no", "noanswer"] and 
        normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC

    # 分词
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    
    # 计算共同词的数量
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return ZERO_METRIC
        
    # 计算precision和recall
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    # 计算F1
    f1 = (2 * precision * recall) / (precision + recall)
    return f1 #, precision, recall


def normalize_text(text):
    text = re.sub("[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：”“《》？]", " ", text.lower())
    text = re.sub("import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    text = re.sub("\s+", " ", text)
    return text.strip()

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


def extract_answer_math(s):
    return s.split("<answer>")[-1].split("</answer>")[0].strip()

def extract_all_answers(s):
    return [ans.strip() for ans in re.findall(r"<answer>(.*?)</answer>", s, re.DOTALL)]
def extract_all_answers_with_full_spans(s):
    results = []
    for match in re.finditer(r"<answer>(.*?)</answer>", s, re.DOTALL):
        answer_text = match.group(1).strip()
        full_start = match.start()  # 从 <answer> 开始
        full_end = match.end()      # 到 </answer> 结束
        results.append((answer_text, full_start, full_end))
    return results

# [其余函数保持不变...]
def normalize_answer(s):
    """标准化答案文本的处理函数"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["'", "'", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))

# End of Update


@dataclass
class Experience:
    """
    表示一次经验（用于 PPO 等算法中的 rollout）

    每个字段形状说明：
    sequences: (B, S) - token 序列（包含 prompt + response）
    action_log_probs: (B, A) - 当前策略下的 action 的对数概率
    base_action_log_probs: (B, A) - 旧策略（参考模型）下的 log probs
    values: (B, A) - 价值函数预测值（critic 输出）
    returns: (B, A) - 折扣累计奖励
    advantages: (B, A) - 优势函数值
    attention_mask: (B, S) - 自注意力 mask
    action_mask: (B, A) - 表示哪些 token 属于 action 部分
    retrieve_mask: (B, A) - 表示哪些 token 是非检索部分（用于 KL 计算）
    kl: (B, A) - KL 散度值
    info: 额外信息字典（包含 reward、num_actions 等）

    """
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    retrieve_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    retrieve_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None


    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.base_action_log_probs = to(self.base_action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.retrieve_mask = to(self.retrieve_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.retrieve_mask = pin_memory(self.retrieve_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self

    # CZP Update 1229 For GRPO
    def __add__(self, other):
        """支持 Experience 之间的合并，用于拼接 batch"""
        if not isinstance(other, Experience):
            return NotImplemented

        info = {}
        for k in self.info.keys():
            info[k] = conditional_cat(self.info[k], other.info[k])

        return Experience(
            sequences=conditional_cat(self.sequences, other.sequences),
            action_log_probs=conditional_cat(self.action_log_probs, other.action_log_probs),
            values=conditional_cat(self.values, other.values),
            returns=conditional_cat(self.returns, other.returns),
            advantages=conditional_cat(self.advantages, other.advantages),
            attention_mask=conditional_cat(self.attention_mask, other.attention_mask),
            action_mask=conditional_cat(self.action_mask, other.action_mask),
            retrieve_mask=conditional_cat(self.retrieve_mask, other.retrieve_mask),
            # retrieve_mask=None,
            info=info,
            kl=conditional_cat(self.kl, other.kl),
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    # End of Update


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    """
    """
    表示从模型生成的一个 batch 的样本。

    有两种格式：
    - batched 格式：所有 sample 使用 padding 对齐
    - packed 格式：将 prompt + response 拼接成一条长序列，不做 padding

    字段含义：
    sequences: 主体 token 序列，形状为 (B, S) 或 (1, total_length)
    attention_mask: 对应 attention 的 mask
    action_mask: 标记哪些 token 属于 response（动作）部分
    retrieve_mask: 标记哪些 token 是非检索区域（用于 KL）
    num_actions: 一个整数或张量，表示 response 的 token 数
    packed_seq_lens: 每条 sample 的长度，仅在 packed 格式下有效
    response_length: 每个 sample 的 response 长度
    total_length: 每个 sample 的总 token 长度
    retrieve_num: 每条 sample 中的检索次数
    pure_response_length: 每条 sample 中非检索区域 response token 长度
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    retrieve_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    retrieve_num: torch.Tensor
    pure_response_length: torch.Tensor
    exit_flag: torch.Tensor
    answer_num: torch.Tensor
class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,  # 当前 actor 模型（用于采样和 log_probs）
        critic: nn.Module,  # 当前 critic 模型（用于 value 计算）
        reward_model: nn.Module,  # 奖励模型，可以是本地模型也可以是远程调用
        initial_model: Actor,  # 参考模型（旧策略），用于 KL 估计
        tokenizer,  # 用于编码/解码 prompt 和生成
        prompt_max_len: int,  # prompt 最大长度
        kl_controller,  # KL 控制器（动态调整 KL 系数）
        strategy=None,  # 包含训练参数（args）和 rank 信息的辅助类
        remote_rm_url: str = None,  # 如果使用远程 reward model，提供其 URL
        remote_sufficient_url: str = None,  # 如果使用远程 summarizer，提供其 URL
        reward_fn=None,  # 可选自定义 reward 聚合函数（多 reward model 场景）
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.remote_sufficient_url = remote_sufficient_url
        self.initial_model = initial_model #initial model就是reference model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator
        # CZP Update 1229 For GRPO
        self.args = strategy.args
        print(self.args)
        # End of Update

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        """
        对输入的 prompts 进行 tokenizer 编码。
        - 当 padding=True 时，返回一个 batch 字典（张量格式）
        - 当 padding=False 时，返回一个 token id 的列表（适用于 packed 格式）
        """

        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], global_step: int, wandb: None, total_training_steps: int, **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        """
        核心函数：从 prompts 构建多个 Experience。

        流程如下：
        1. 使用 actor 生成 response，得到 Samples（包含序列、action_mask 等）
        2. 遍历 Samples，逐个转化为 Experience（logprobs、value、reward、KL）
        3. 对所有 Experience 执行 reward 后处理（比如 baseline 减均值）
        4. 计算 returns 和 advantages（GAE 或其他方式）
        """

        args = self.strategy.args
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            
            torch.distributed.barrier()
            torch.cuda.synchronize()

        # param_stats = ray.get(self.vllm_engines[0].get_param_stats.remote())
        # for stat in param_stats:
        #     print(stat)                

        samples_list = self.generate_samples(all_prompts, **generate_kwargs) #vllm生成，这里会给action_mask赋值，改这里 #TODO!!  #这里已经rollout了N个输出出来了。
        torch.distributed.barrier()# 同步所有进程
        torch.cuda.synchronize()  
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")
            torch.distributed.barrier()# 同步所有进程
            torch.cuda.synchronize()  

        experiences = []
        for i in tqdm(
            range(0, len(samples_list), args.n_samples_per_prompt * 20),
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            samples = samples_list[i : i + args.n_samples_per_prompt * 20]
            m = self.make_experience(samples, global_step, total_training_steps = total_training_steps)
            if(type(m) == list):
                experiences.extend(m)
            else:
                experiences.append(m)
        if args.enable_accuracy_filter and global_step > args.freezing_filter_steps:
            experiences = self.filter(experiences, global_step, wandb = wandb)

        experiences, rewards = self.process_experiences(experiences) 
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]  # 动作数量（每条样本）
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                retrieve_mask=experience.retrieve_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )
            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce_baseline", "rloo", "group_norm"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    experience.retrieve_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)

                experience_num = len(experience.sequences)

            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")
            
            if not getattr(self, "packing_samples", False): #如果开启packing，则不使用sum，而是使用list的sum
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")

        return experiences


    @torch.no_grad()
    def show_experience(self, experience: Experience):
        """
        打印 experience 中的各项内容，用于调试。
        会显示 tensor 的维度、部分内容，或 list 的长度。
        """

        if isinstance(experience.action_log_probs, list):
            print(
                "list len",
                len(experience.action_log_probs),
                "action_log_probs: ",
                experience.action_log_probs[0].size(),
            )
            # print("action_log_probs: ", experience.action_log_probs[0].size())
        else:
            print("tensor action_log_probs: ", experience.action_log_probs.size())
        if isinstance(experience.sequences, list):
            torch.set_printoptions(threshold=500)
            list_data = experience.sequences[0].tolist()
            print("experience.sequences-try: ", list_data)

            print("experience.sequences-1", experience.sequences[0].size(), "list len", len(experience.sequences))
            torch.set_printoptions(threshold=10)
        elif experience.sequences is not None:
            print("tensor experience.sequences", experience.sequences.size())
        else:
            print("experience.sequences is None")
        if isinstance(experience.values, list):
            print("experience.values", experience.values[0].size(), "list len", len(experience.values))
            # print("experience.values", experience.values[0].size())
        elif experience.values is not None:
            print("tensor experience.values", experience.values.size())
        else:
            print(f"experience.values is None")
        if isinstance(experience.returns, list):
            print("experience.returns: ", experience.returns[0].size(), "list len", len(experience.returns))
            # print("experience.returns: ", experience.returns[0].size())
        elif experience.returns is not None:
            print("tensor experience.returns: ", experience.returns.size())
        else:
            print(f"experience.returns is None")
        if isinstance(experience.advantages, list):
            list_data = experience.advantages[0].tolist()
            print("experience.advantages-try: ", list_data)
            print("experience.advantages-1: ", experience.advantages[0].size(), "list len", len(experience.advantages))
        elif experience.advantages is not None:
            print("tensor experience.advantages: ", experience.advantages.size())
        else:
            print(f"experience.advantages is None")
        if isinstance(experience.attention_mask, list):
            print(
                "experience.attention_mask: ",
                experience.attention_mask[0].size(),
                "list len",
                len(experience.attention_mask),
            )
            # print("experience.attention_mask: ", experience.attention_mask[0].size())
        elif experience.attention_mask is not None:
            print("tensor experience.attention_mask: ", experience.attention_mask.size())
        else:
            print(f"experience.attention_mask is None")
        if isinstance(experience.action_mask, list):
            print(
                "experience.action_mask: ", experience.action_mask[0].size(), "list len", len(experience.action_mask)
            )
            # print("experience.action_mask: ", experience.action_mask[0].size())
        elif experience.action_mask is not None:
            print("tensor experience.action_mask: ", experience.action_mask.size())
        else:
            print(f"experience.action_mask is None")
        if isinstance(experience.kl, list):
            print("experience.kl: ", experience.kl[0].size(), "list len", len(experience.kl))
            # print("experience.kl: ", experience.kl[0].size())
        elif experience.kl is not None:
            print("tensor experience.kl: ", experience.kl.size())
        else:
            print(f"experience.kl is None")

        for k in experience.info:
            if isinstance(experience.info[k], torch.Tensor):
                print(k, experience.info[k].size(), experience.info[k])
                # print(experience.info[k])
            elif isinstance(experience.info[k], list):
                print(f"{k} list len: {len(experience.info[k])}, first: {experience.info[k][0]}")


    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], **generate_kwargs) -> List[Samples]:
        """
        使用 actor 模型生成样本，每次处理一个 micro_batch。

        输入：
        - all_prompts: 所有输入的 prompt 文本
        - generate_kwargs: 生成时使用的参数（如 temperature, top_k 等）

        输出：
        - samples_list: 一个列表，包含多个 Samples（每个对应一个 batch）
        """

        kill # 占位用于调试，训练前需注释掉
        assert not getattr(self, "packing_samples", False) # 当前版本不支持 packed 格式
        args = self.strategy.args
        self.actor.eval() # 切换到评估模式，避免 dropout 等操作影响生成
        # sample multiple response
        # 将每个 prompt 复制多份（用于同一个 prompt 生成多个样本）
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            # 生成样本（response）
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            # 构造 Samples 对象

            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        """
        将一个 Samples 对象转化为一个完整的 Experience。
        包括计算：log_probs、values、reward、KL divergence 等。

        输入：
        - samples: 一个 Samples 对象，包含生成的序列及其掩码信息

        输出：
        - experience: 一个 Experience 对象，包含训练所需的所有信息
        """
        kill
        # 切换所有模型到 eval 模式（禁用 dropout、层归一化中的随机性）
        self.actor.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()
        
        # 从样本中提取各类信息
        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions

        # 用当前策略模型计算 action 的 log_prob
        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # 用参考策略（旧策略）计算 log_prob，用于 KL 散度计算
        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        # 如果有 critic 模型，则获取 value 值（用于后续 advantage、return）
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # 获取 reward 值（支持本地和远程 reward model）
        if self.remote_rm_url is not None:
            # remote RM             # 远程 reward model（将序列 decode 成文本后 POST 请求）
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
        else:
            # local RM
            # 本地 reward model，直接前向
            r = self.reward_model(sequences, attention_mask)
        
        # 是否计算 KL 散度
        if (self.initial_model is not None) and (not self.strategy.args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)
        
        # 记录信息字典（后续可用于 logging 或 reward 后处理）
        info = {
            "kl": masked_mean(kl, action_mask, retrieve_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        # reset model state
        # 将 actor 和 critic 切换回训练模式
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        # 返回 Experience 对象（仍需后续计算 advantage 和 return）
        return Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )
    @torch.no_grad()
    def filter(self, experiences: List[Experience], global_step: int, wandb:None) -> List[Experience]:
        """
        Filter experiences based on accuracy reward.

        Output:
        - experiences: List of filtered Experience
        """

        args = self.strategy.args
        accuracy_rewards = torch.cat([experience.info["answer_rewards"] for experience in experiences]) 
       # accuracy_rewards = torch.cat([experience.info["reward"] for experience in experiences])
        accuracy_rewards = accuracy_rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
        accuracy_rewards_min = torch.min(accuracy_rewards, dim=-1).values
        accuracy_rewards_max = torch.max(accuracy_rewards, dim=-1).values
        accuracy_counts = Counter(accuracy_rewards_min.tolist())
        print("Accuracy min distribution:", " ".join(f"{k:.2f}:{v}" for k, v in sorted(accuracy_counts.items())))
        accuracy_counts = Counter(accuracy_rewards_max.tolist())
        print("Accuracy max distribution:", " ".join(f"{k:.2f}:{v}" for k, v in sorted(accuracy_counts.items())))

        assert len(experiences) % len(accuracy_rewards) == 0
        group_len = args.n_samples_per_prompt
        grouped_experience = [experiences[i : i + group_len] for i in range(0, len(experiences), group_len)]
        filtered_experiences = []
        min_count = 0 
        max_count = 0 
        zhongjian_count = 0 
        for group, reward_min, reward_max in zip(grouped_experience, accuracy_rewards_min, accuracy_rewards_max):
            # if reward_max.item() - reward_min.item() <= args.accuracy_lower_bound: #如果最大最小值就差0.1，就认为这个group的reward都一样，就丢掉这个group
            #     zhongjian_count += args.n_samples_per_prompt
            if reward_max.item() <= args.accuracy_lower_bound:
                max_count += args.n_samples_per_prompt
                continue 
            elif reward_min.item() >= args.accuracy_upper_bound:
                min_count += args.n_samples_per_prompt
                continue
            else:
            # #if args.accuracy_lower_bound <= reward_min.item() <= args.accuracy_upper_bound:
                filtered_experiences.extend(group)

            # if reward_max.item() <= args.accuracy_lower_bound:
            #     max_count += args.n_samples_per_prompt
            #     continue 
            # elif reward_min.item() >= args.accuracy_upper_bound:
            #     min_count += args.n_samples_per_prompt
            #     continue
        print(f"Filtered { len(experiences) - len(filtered_experiences)} \% {(len(experiences) - len(filtered_experiences))/len(experiences)} experiences.")
        if self.strategy.is_rank_0():
            print('打印filter的step', global_step)
            log = {
                "train/filtered_>0.9_experience": min_count/ len(experiences),
                "train/filtered_<0.1_experience": max_count/ len(experiences),
                "train/(max-min)<0.1_experience": zhongjian_count/ len(experiences)
            }
            wandb.log(log, step=global_step)

        return filtered_experiences


    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        """
        对多个 Experience 进行 reward 的后处理。
        用于根据策略（如 RLOO、group_norm 等）调整 reward，提升训练稳定性。

        输入：
        - experiences: 一个 Experience 对象列表

        输出：
        - 处理后的 experiences（结构不变）
        - 每个 Experience 中的 reward（tensor），可直接用于计算 return 和 advantage
        """
        args = self.strategy.args
        # reward shaping for RLOO
        # 针对 RLOO 策略的 reward 处理
        device = torch.cuda.current_device()
        if args.advantage_estimator == "rloo":
            # 合并所有 reward（形状变为 [batch_size]）
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            # 为每个样本记录是否回答正确（准确率信息）
            for experience in experiences:
                experience.info['acc_info'] = (experience.info["reward"] == 1).float().reshape(-1)
            # reshape 成 [n_prompts, n_samples_per_prompt]（每个 prompt 有多个 sample）
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            # acc_info = (rewards == 1).float()
            # expe
            # 计算 leave-one-out baseline（去掉自身的平均 reward）
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            # 计算 advantage = reward - baseline
            rewards = rewards - baseline
            # reshape 回原始 shape，按 batch 切分
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences)) #将张量分成 len(experiences) 个相等大小的块
            return experiences, rewards

        # 针对 group normalization 策略
        elif args.advantage_estimator == "group_norm":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            print("打印group_norm前的reward:",rewards)

            # 对每组样本进行标准化（均值为0，方差为1）
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
            difficulty = torch.cat([experience.info["difficulty"] for experience in experiences])
            penalty = torch.cat([experience.info["penalty"] for experience in experiences])
            # reshape 回原始 shape，按 batch 切分
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            print("打印group_norm后的reward:",rewards)
            rewards = torch.cat(rewards)
            rewards = ((rewards - penalty) * difficulty).chunk(len(experiences))
            print("打印加了penalty和difficulty的reward:",rewards)
            ### 计算advantage_valued_porportion
            # 生成 0/1 mask
            for exp, ri in zip(experiences, rewards):
                val = (ri != 0).float()
                exp.info['advantage_valued_porportion'] = val
            return experiences, rewards

        # 针对 REINFORCE + baseline 策略
        elif args.advantage_estimator == "reinforce_baseline":
            # REINFORCE++-baseline removed the / std and K3 kl loss in GRPO.
            # `/ std` is not needed in RL variance reduction theory, and `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
           
            # 直接减去均值作为 baseline（不做归一化）
            rewards = rewards - rewards.mean(-1, keepdim=True)
            difficulty = torch.cat([experience.info["difficulty"] for experience in experiences]) #对reinforce++而言，只有一个ppo，所以没有difficulty的计算了.
            penalty = torch.cat([experience.info["penalty"] for experience in experiences])

            # reshape 回原始 shape，按 batch 切分
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            rewards = torch.cat(rewards)
            rewards = (rewards - penalty).chunk(len(experiences))

            return experiences, rewards

        # CZP Update 1229 For GRPO
        # if self.advantage_estimator in ["group_norm"]:
        #     return [sum(experiences)], [experience.info["reward"] for experience in [sum(experiences)]]
        # End of Update

        # default rewards
        # 默认情况下，直接返回原始 reward
        return experiences, [experience.info["reward"] for experience in experiences]

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advantage 的计算方式如下：
        Adv_t = δ_t + (γλ) * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...
        其中 δ_t = r_t + γ * V_{t+1} - V_t

        Return 的计算方式是：
        Ret_t = Adv_t + V_t
        参数：
        - values: critic 输出的每个 token 的 V 值，形状为 (B, A)
        - rewards: 每个 token 的即时奖励，形状为 (B, A)
        - action_mask: 用于屏蔽 padding 的二值 mask，形状为 (B, A)
        - gamma: 折扣因子
        - lambd: GAE 的平衡系数

        返回：
        - advantages: GAE 算法计算出的优势函数值，形状为 (B, A)
        - returns: 折扣累计回报（用于 value 回归目标），形状为 (B, A)
        """
        # kill
        if isinstance(values, list):
            # 处理 packed 格式的情况，values 和 rewards 是列表形式
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0  # 用于存储上一个 timestep 的 GAE 累计值
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        # 使用 action_mask 屏蔽掉无效的 token（padding）
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards
        
        # 从后向前计算 GAE
        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        # 将 reversed 的 advantage 序列反转为正常顺序
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values # return = advantage + value（作为 critic 回归目标）
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns( # returns: 累计回报 它是某个状态开始之后，模型可以期望获得的 总奖励,用来训练critic，即模型对未来奖励的预测。而advantage = return - value , value是critic预测当前状态的期望汇报。advantage衡量某个 动作到底比模型期望的好多少。advantage用来训练actor，returns用来训练critic
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        retrieve_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """
        """
        使用 REINFORCE 算法（不使用 GAE）计算 cumulative return（累计回报）。

        对于每个位置 t 的 return，计算方式为：
            R_t = r_t + γ * r_{t+1} + γ^2 * r_{t+2} + ...

        输入：
        - rewards: 即时奖励张量，形状为 (B, A)
        - action_mask: 用于屏蔽 padding 的掩码，形状为 (B, A)
        - retrieve_mask: 用于屏蔽检索区域（不参与 reward），形状为 (B, A)
        - gamma: 折扣因子

        输出：
        - returns: 与 rewards 同形状，表示每个 token 的累计回报
        """
        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            # print(len(rewards))
            # print(type(retrieve_mask[0]))
            # kill
            # 如果是 packed 格式，rewards 是一个 list，每条样本长度不同

            returns = []
            for p, r in enumerate(rewards):
                # 每个样本单独计算返回值（递归调用）
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, retrieve_mask[p].unsqueeze(0), gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        # 应用 action mask：屏蔽 padding 区域

        if action_mask is not None:
            rewards = action_mask * rewards

        # print("cumulative-reward:",rewards)
        # kill
        # Calculate returns by accumulating
        # discounted rewards
        # 从后往前逐步累加 reward，乘以 gamma
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return
        # 应用 retrieve mask：屏蔽掉检索内容区域的 return（这部分不计算 reward）
        if retrieve_mask is not None:
            returns = returns * retrieve_mask
        print('打印模型的retrieval mask:', retrieve_mask)
        # print("cumulative-returns-size: ", returns.size())
        return returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines # 用于生成的 vLLM 引擎（可选）
        self.packing_samples = packing_samples # 是否启用 packed 格式（prompt + response 不加 padding）

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], global_step: int,wandb: None, total_training_steps: int, **generate_kwargs) -> List[Experience]:
        # 如果开启性能评估，记录时间统计项

        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }

        # 继承基类生成完整的 experience list
        experiences = super().make_experience_list(all_prompts, global_step, wandb, total_training_steps, **generate_kwargs) #一个list，只有[0]是一个tensor，装64个experience
        # 如果有 critic 模型，则将 experience 异步发送到远程 critic
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu) #所有远程对象的方法（包括模型方法）都需要用 .remote() 调用。
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, **generate_kwargs)

        # return self._generate_vllm(all_prompts, **generate_kwargs)
        return self._generate_vllm_with_retrieve(all_prompts, **generate_kwargs)


    @torch.no_grad()
    def make_experience(self, samples_list: Samples|List[Samples], global_step: int, total_training_steps: int) -> Experience:
        # 与基础类相比，这里使用 ray 进行远程分布式模型调用，提取 logprobs、rewards 等信息
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        """
        将 Samples 转换为 Experience（远程版本）
        核心功能：
        - 使用 actor 本地计算 log_probs
        - 使用远程 initial_model 计算参考策略 log_probs
        - 使用远程 critic 计算 value
        - 使用远程 reward model 或 API 获取 reward
        - 计算 KL divergence
        """

        # kill
        self.actor.eval() #self.actor不是remote对象，Actor = ray.remote(Actor) self.actor = Actor.remote(...) # actor 是本地对象，直接切换为 eval 模式

        device = torch.cuda.current_device()
        expeiences = []
        queries_list = [] 
        exit_flags_list = []
        ### 单独用于计算reward
        r_refs = []
        for samples in samples_list:
        # extract values from samples
            sequences = samples.sequences
          #  print('打印make_experience里面的的sequences长度', sequences.shape) #这是 micro_rollout_batch_size pack在一起了，通过attention_mask来进行区分不同的例子.. 
            attention_mask = samples.attention_mask
            action_mask = samples.action_mask
            retrieve_mask = samples.retrieve_mask
            num_actions = samples.num_actions
            packed_seq_lens = samples.packed_seq_lens
            exit_flags = samples.exit_flag.to("cpu").tolist()
            sequences_cpu, attention_mask_cpu = (
                sequences.to("cpu"),
                attention_mask.to("cpu"),
            )
                    # rewards     # reward：支持远程 reward model，也支持 reward_model 为 ray actor 列表     
            # support remote RM API with ray
            if not self.remote_rm_url:
                for rm in self.reward_model:
                    r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens))
            else:
                # remote RM   # 远程 RM API 的输入是解码后的文本 queries
                if not self.packing_samples: # 如果未启用 packed 格式，直接解码
                    queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
                else:
                    sequences_list = []
                    offset = 0
                    tokens_list = sequences_cpu.tolist()[0]
                    for length in packed_seq_lens: #所以这里是用来解开pack的一个操作..
                        sequences_list.append(tokens_list[offset : offset + length])
                        offset += length
                    queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)
                    queries_list.extend(queries)
                    exit_flags_list.extend(exit_flags)
        for rm in self.remote_rm_url:
            r = remote_rm_fn_ray.remote(rm, queries=queries_list, group_size=self.args.n_samples_per_prompt, exit_flags=exit_flags_list, global_step = global_step, total_training_steps=total_training_steps) #主要是在这里进行修改，让queries是一个包含同一个question所有sample的queries，而不是只有一个query.
            r_refs.append(r)
        r_rewards = ray.get(r_refs) #len(r_rewards) = [micro_rollout_batch_size * n_samples_per_prompt]
        r_rewards = r_rewards[0] #因为只有一个reward fuction，所以只取0
        format_rewards = r_rewards["纯format分数"]
        answer_rewards = r_rewards["纯answer分数"]
        previous_right_final_incorrect_results_wait_enough_information = r_rewards["前面对了后面错了有足够信息"]
        previous_incorrect_final_right_results_wait_enough_information = r_rewards["前面错了后面对了有足够信息"]
        previous_right_final_incorrect_results_wait_not_enough_information = r_rewards["前面对了后面错了没有足够信息"]
        previous_incorrect_final_right_results_wait_not_enough_information = r_rewards["前面错了后面对了没有足够信息"]
        no_change_results = r_rewards["没有反思的"]
        no_change_reflect_results = r_rewards["没有改变的reflect"]
        exit_lose_reward = r_rewards["exit_lose_reward"]
        sufficient_premilinary_match = r_rewards["sufficient_premilinary_match"]
        previous_right_final_incorrect = r_rewards["反思前正确的，后面错误的"]
        previous_incorrect_final_right = r_rewards["反思前错误的，后面正确的"]
        thinking_rewards = r_rewards["thinking_reward"]
        difficulty = r_rewards["difficulty"]
        penalty = r_rewards["penalty"]
        # print('打印make experience里面的r_rewards长度', len(r_rewards))
        for index, samples in enumerate(samples_list): #len(samples_list) = n_samples_per_prompt  因为我在外面的make_experience_list的for循环是这么写的
        # extract values from samples
            sequences = samples.sequences
            attention_mask = samples.attention_mask
            action_mask = samples.action_mask
            retrieve_mask = samples.retrieve_mask
            num_actions = samples.num_actions
            packed_seq_lens = samples.packed_seq_lens
            # Remote: 获取 base log_probs 和 critic value     # 将 initial model 设置为远程执行，并把输入迁移到 CPU，避免显存炸
            start = time.time()
            sequences_cpu, attention_mask_cpu = (
                sequences.to("cpu"),
                attention_mask.to("cpu"),
            )

            # init log probs
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )

            # values     # critic 值，如果启用 colocate_critic_reward，要先强制执行完释放显存
            if self.critic is not None:
                value_ref = self.critic.forward.remote(
                    sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
                )
                # avoid CUDA OOM when colocate models #Critic 和 Reward model 在同一个 GPU，必须交替执行，防止显存爆掉
                if self.strategy.args.colocate_critic_reward:
                    ray.get([value_ref]) #ray.get() 会强制等待这个远程任务完成并释放资源
                    ray.get([self.critic.empty_cache.remote()])
            else:
                value_ref = ray.put(None)

            if self.strategy.args.colocate_actor_ref:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])


            # log probs     # 本地 actor 计算 log probs
            action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)

            actor_value_rm_time = time.time() - start

            # wait initial/critic/reward model done
            start = time.time()
            # 等待远程任务完成，获取 log_probs、value 和 reward 
            # ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
            ref_values = ray.get([base_action_log_probs_ref, value_ref])
            wait_time = time.time() - start

            sum_of_rewards = r_rewards["总的分数"]
            rewards_list = []
            for jdex in range(len(packed_seq_lens)):
                rewards_list.append(sum_of_rewards[index*len(packed_seq_lens) + jdex])
            # print(f'打印第{index}个rewards_list', rewards_list)
            rewards = [torch.tensor(rewards_list)]      # [torch.tensor([r_rewards[0][index]])] ## 目前这里只考虑一个远程reward 计算方式,所以只取0 这个0目前是放在前面就替代了.,而且要是list[list]格式. 然后第一维度代表奖励计算方式个数 也就是remote_rm_url的个数,第二个维度才代表不同例子的奖励。
            base_action_log_probs, value = ref_values[0], ref_values[1]
            base_action_log_probs = base_action_log_probs.to(device)
            if value is not None:
                value = value.to(device)
            rewards = [r.to(device) for r in rewards]
            # print('打印make_experience里面的rewards长度', len(rewards), len(rewards[0]))
            r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0] 
            # print('打印make_experience里面的rewards长度', len(r))

            # avoid CUDA OOM when colocate models
            if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
                ray.get([self.reward_model[0].empty_cache.remote()])

            if self.strategy.args.colocate_actor_ref:
                torch.cuda.empty_cache()

            if (self.initial_model is not None) and (not self.strategy.args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    action_mask=action_mask,
                    retrieve_mask=retrieve_mask,
                    use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

            if not self.packing_samples:
                kl_mean = masked_mean(kl, action_mask,retrieve_mask=retrieve_mask, dim=-1,)
            else:
                # convert tensor into list of tensors so that it's easier to manipulate
                # within dataset.
                sequences = unpacking_samples(sequences, packed_seq_lens)
                attention_mask = None     # packed 格式下 attention_mask 统一为 None（不再需要 padding 位置 mask）
                action_log_probs = unpacking_samples(action_log_probs, num_actions) #    # 将action_log_probs 拆分成每个样本一个 tensor（因为每个样本 action 长度不同）
                retrieve_mask = unpacking_samples(retrieve_mask, num_actions)
                if value is not None: #    # 如果有 value（critic 输出），也要拆分
                    value = unpacking_samples(value, num_actions)

                if base_action_log_probs is not None:
                    base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

                kl = unpacking_samples(kl, num_actions) #    # KL 拆分为每个样本一个 tensor
                kl_mean = torch.zeros(len(kl), device=device)  # 预先创建一个张量，长度为 kl 的长度
                for i, each_kl in enumerate(kl): #    # 遍历每个样本，计算对应的 masked KL 平均值
                    kl_mean[i] = masked_mean(each_kl, action_mask,retrieve_mask=retrieve_mask[i], dim=-1,)

                if not self.strategy.args.use_kl_loss: #   # 如果禁用了 KL 损失，直接将 base_action_log_probs 设为 None（后续不使用）

                    base_action_log_probs = None

            info = {
                "kl": kl_mean,
                "reward": r, #    # 每个 sample 的奖励值（可能是标量或向量，取决于 reward model）
                "response_length": samples.response_length,
                "total_length": samples.total_length,
                "num_actions": num_actions, #    # 每个 sample 的 action 数量（即 response 的 token 数，和 response_length 含义接近）
                "retrieve_num": samples.retrieve_num, #    # 每个 sample 中检索片段的数量（即包含多少次 <|begin_of_documents|>）
                "pure_response_length": samples.pure_response_length, #    # 每个 sample 中 response 中非检索部分的 token 数（真实回答长度）
                "format_rewards": torch.tensor([format_rewards[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "answer_rewards": torch.tensor([answer_rewards[index]], device=device, dtype=torch.float), #   因为后面需要再buffer append的时候解开packing，所以这里需要加个[]构成一个list。
                "previous_right_final_incorrect_rewards_wait_enough_information": torch.tensor([previous_right_final_incorrect_results_wait_enough_information[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "previous_incorrect_final_right_rewards_wait_enough_information": torch.tensor([previous_incorrect_final_right_results_wait_enough_information[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "previous_right_final_incorrect_rewards_wait_not_enough_information": torch.tensor([previous_right_final_incorrect_results_wait_not_enough_information[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "previous_incorrect_final_right_rewards_wait_not_enough_information": torch.tensor([previous_incorrect_final_right_results_wait_not_enough_information[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "no_reflect_porportion": torch.tensor([no_change_results[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "no_change_with_reflect_rewards": torch.tensor([no_change_reflect_results[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "exit_lose_reward": torch.tensor([exit_lose_reward[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "sufficient_premilinary_match": torch.tensor([sufficient_premilinary_match[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "exit_porpotions": samples.exit_flag,
                "answer_number": samples.answer_num,
                "thinking_reward": torch.tensor([thinking_rewards[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "reflect_previous_right_final_incorrect": torch.tensor([previous_right_final_incorrect[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "reflect_previous_incorrect_final_right": torch.tensor([previous_incorrect_final_right[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "difficulty": torch.tensor([difficulty[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "penalty": torch.tensor([penalty[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
            }

            if self.strategy.args.perf: #    # 每个 sample 中 response 中非检索部分的 token 数（真实回答长度）
                self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
                self.perf_stats["wait_time"] += wait_time

            experience = Experience(
                sequences,  # token 序列（含 prompt 和 response）
                action_log_probs,  # 当前策略模型的 log_probs
                base_action_log_probs,  # 初始参考策略的 log_probs
                value,  # Critic 给出的 V 值
                None,  # returns 先留空，之后计算
                None,  # advantages 同上
                attention_mask,  # 注意力掩码（非 packed 时用）
                action_mask,  # 哪些 token 是 action（即 response 部分）
                retrieve_mask,  # 哪些 token 是非检索区域（KL 用）
                info,  # 额外信息字典（reward、length 等）
                kl,  # 每个 token 的 KL 散度（后续可选用）
            )
            experience = experience.to_device("cpu")
            expeiences.append(experience)
        self.actor.train()  # reset model state # 将 actor 模型重新设置为 train 模式（因为前面切到 eval）
        return expeiences


    def _generate_vllm(self, all_prompts: List[str], **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank() # 获取当前进程的 rank（用于多卡分配）
        world_size = torch.distributed.get_world_size()  # 获取总的进程数（即 GPU 数量）

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        # 根据 rank 分配 vLLM 引擎（若引擎数量少于 world_size，就循环使用）
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args # 获取训练参数

        # 设置采样参数，例如温度、top_p、生成 token 数等
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),  # 默认温度 1.0
            top_p=kwargs.get("top_p", 1.0),  # 默认 top-p 为 1.0（不截断）
            top_k=kwargs.get("top_k", -1),  # 默认 top-k 为 -1（不截断）
            max_tokens=kwargs.get("max_new_tokens", 1024),  # 最大生成长度
            min_tokens=kwargs.get("min_new_tokens", 1),  # 最小生成长度
            skip_special_tokens=kwargs.get("skip_special_tokens", False),  # 是否跳过特殊符号
            include_stop_str_in_output=True,  # 是否在输出中包含停止字符串
        )

        # Expand prompt list based on the number of samples per prompt
        # 每个 prompt 扩展成多个 sample（重复 n_samples_per_prompt 次）
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        # 对所有 prompts 进行编码（不使用 padding）
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        all_output_refs = []  # 存放所有远程生成任务的引用
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms) # 每个引擎处理的 prompt 数
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]  # 拿到该引擎负责的一批 prompt
            if args.random_temperature:  # 若启用随机温度策略
                values = [i / 10 for i in range(5, 11)]  # 可选温度值列表（0.5 ~ 1.0）
                sampling_params.temperature = random.choice(values)  # 随机选择一个温度
            print("temperature", sampling_params.temperature)  # 打印当前温度
            if prompt_token_ids:  # 如果这批 prompt 不为空
                all_output_refs.append(  # 向对应的 vLLM 引擎提交生成任务（异步）
                    llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
                )

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])  # 获取所有引擎的生成结果，并合并成一个列表

        samples_list = []  # 最终返回的样本列表
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):  # 每次取一个小 batch
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]  # 当前小 batch 的输出结果
            if not self.packing_samples:  # 如果不使用 packed 格式
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0  # 初始化输入和输出最大长度
                for output in outputs:  # 计算该 batch 中输入和输出的最大长度
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id  # 获取特殊 token id
                sequences = []  # 用于存放拼接后的输入+输出序列
                for output in outputs:
                    input_len = len(output.prompt_token_ids)  # 当前样本的输入长度
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)  # 左填充
                    output_len = len(output.outputs[0].token_ids)  # 当前样本的输出长度
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)  # 右填充
                    sequences.append(input_ids + output_ids)  # 拼接后加入序列列表

                sequences = torch.tensor(sequences)  # 转换为 tensor
                # 处理生成的序列，得到 attention mask 和 action mask
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                # 将数据移到 GPU
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                retrieve_mask = retrieve_mask.to("cuda")  # retrieve_mask 这里是未定义的，会报错，需上层处理
                # 构建 Samples 对象，并添加到结果列表中
                samples_list.append(
                    Samples(
                        sequences=sequences,  # 输入输出拼接序列
                        attention_mask=attention_mask,  # attention 掩码
                        action_mask=action_mask,  # 标注输出部分的位置
                        retrieve_mask=retrieve_mask,  # 标注非检索区域
                        num_actions=action_mask.size(1),  # 动作数量（输出 token 数）
                        packed_seq_lens=None,  # 非 packed 模式，此处为 None
                        response_length=action_mask.float().sum(dim=-1),  # 每个样本的输出 token 数
                        total_length=attention_mask.float().sum(dim=-1),  # 每个样本的总长度
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []              # 存储拼接后的所有 token 序列
                packed_seq_lens = []        # 每条样本的长度（prompt + response）
                attention_mask = []         # 用于模型区分不同样本
                retrieve_mask=[]            # 检索部分的掩码（0 表示该 token 属于检索部分）
                num_actions = []            # response 的 token 数（最少为1）                
                for i, output in enumerate(outputs):  # 遍历每个 output 样本
                    input_len = len(output.prompt_token_ids)         # prompt 的长度
                    output_len = len(output.outputs[0].token_ids)    # response 的长度
                    packed_seq_lens.append(input_len + output_len)   # 添加 packed 长度
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))  # 拼接 prompt 和 response
                    attention_mask.extend([i + 1] * (input_len + output_len))  # 相同位置填入样本编号

                    # 构建 retrieve_mask，默认 response 的所有位置都是 1
                    # response_seq = list(output.outputs[0].token_ids)
                    # retrieve_mask_now = [1]*len(response_seq)
                    
                    # 找到 retrieve 的标记位置（开始和结束）
                    # 构建 retrieve_mask，默认 response 的所有位置都是 1
                    response_seq = list(output.outputs[0].token_ids)
                    retrieve_mask_now = [1]*len(response_seq)
                    
                    # 找到 retrieve 的标记位置（开始和结束）
                    start_indices = [g for g, x in enumerate(response_seq) if x == 151657]
                    end_indices = [g for g, x in enumerate(response_seq) if x == 151658]
                    assert len(start_indices) == len(end_indices), "KL: start_indices 和 end_indices 长度不一致"
                    
                    # 将标记区间位置设为 0（属于 retrieve 的部分）
                    for start, end in zip(start_indices, end_indices):
                        for h in range(start, end + 1):
                            retrieve_mask_now[h] = 0

                    retrieve_mask.extend(retrieve_mask_now)  # 添加到总掩码中
                    num_actions.append(max(1, output_len))   # 记录 response 长度
                # 将拼接的内容转为 tensor 并存入 Samples 对象中
                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                retrieve_mask = torch.tensor(retrieve_mask, device="cuda").unsqueeze(0)

                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                # print("sequences_size:",sequences.size())
                # print("attention_mask_size:",attention_mask.size())
                # print("vllm-retrieve_mask_size:",retrieve_mask.size())
                # print("response_length_size:",response_length.size())
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                # 将处理好的样本加入结果列表

                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        retrieve_mask = retrieve_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                    )
                )
        return samples_list


    def _generate_vllm_with_retrieve(self, all_prompts: List[str], **kwargs) -> List[Samples]:
        url_wiki = "http://0.0.0.0:5004/queries"  # 检索服务器地址
        from vllm import SamplingParams

        # 获取当前进程 rank 和总进程数
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        print('打印generate_vllm_with_retrieve的相关信息', rank, world_size) #果然这里是每个gpu都会有一个进程来调用这个方法。
        # 为当前 rank 分配使用的 vllm 实例
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )
      #  te = kwargs.get("temperature", 1.0)
       # print('打印模型运行的temperature',)
        # 将 prompt 复制多份，每个 prompt 对应多个样本
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        # 将每个 prompt 拆成 idx + prompt 字典
        prompts_w_idx_dict = []
        for idx_w_prompt in all_prompts:
            idx, prompt = idx_w_prompt.split("<|idx_prompt_split|>")
            prompt, gold_answer = prompt.split("<pro_answer>")
            prompts_w_idx_dict.append({"idx": idx, "prompt": prompt, "gold_answer": gold_answer})
        # 从这里开始加检索功能
        # all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]
        stop_tokens = ["</search>"] # 停止符号
        batch_size = (len(prompts_w_idx_dict) + len(llms) - 1) // len(llms)
        print("llms-len:",len(llms)) #只有一个llms
        all_outputs = []
        for i, llm in enumerate(llms): # 每个llm都执行若干group推理(每个gruop要进行多次rollout)
            # 为每个 LLM 分配一部分 prompt
            print('第i个llm:',i,rank, len(prompts_w_idx_dict), i * batch_size, (i + 1) * batch_size)
            idx_w_prompt_part = prompts_w_idx_dict[i * batch_size : (i + 1) * batch_size]
            print(f'打印第{rank}个llm的prompt:',idx_w_prompt_part[0])

            # print("idx_w_prompt_part:",idx_w_prompt_part)
            data_keys = ["prompt" , "idx"]
            ds = Dataset.from_dict({key: [d[key] for d in idx_w_prompt_part] for key in data_keys})
            # print("ds:",ds)

            finished_all_list=[]  # 当前部分所有完成的样本
            continued_answer = copy.deepcopy(idx_w_prompt_part) # 用于逐步生成并添加 retrieved 文档

            for t in range(11): # 每条都执行11次推理，目的就是推理出 完整的 solution，尽可能让需要检索的query全出现, 11条推理是指这个prompt最多使用11次，因为检索推理是多次生成的。
                finished_texts = []
                continued_texts = []
                sampling_params = SamplingParams(temperature=1, top_p=0.95, max_tokens=512, stop=stop_tokens) #这是26_8 所使用的参数
               # sampling_params = SamplingParams(temperature=2, top_p=0.6, top_k=30, max_tokens=512, stop=stop_tokens)

                outputs_ray = llm.generate.remote(ds['prompt'], sampling_params)
                outputs = ray.get(outputs_ray)
                query_list=[]
                for q, output in enumerate(outputs):

                    prompt = output.prompt
                    idx = continued_answer[q]["idx"]
                    gold_answer = continued_answer[q]["gold_answer"]
                    stop_reason = output.outputs[0].stop_reason
                    generated_text = output.outputs[0].text
                    # 初次推理时记录 prompt 的 token
                    if "prompt_ids" not in continued_answer[q]:
                        input_token_ids = list(output.prompt_token_ids)
                    else:
                        input_token_ids = continued_answer[q]["prompt_ids"]

                    retrieve_num_count = continued_answer[q].get("retrieve_num_count",0)

                    all_token_ids = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
                    output_token_ids = all_token_ids[len(input_token_ids):]

                    # if 'actual_response_ids' not in continued_answer[q]:
                    #     actual_response_ids = [list(output.outputs[0].token_ids)]
                    # else:
                    #     continued_answer[q]['actual_response_ids'].append(list(output.outputs[0].token_ids))
                    #     actual_response_ids = continued_answer[q]['actual_response_ids'].copy()
                    
                    if 'retrieval_mask_tokens' not in continued_answer[q]:
                        print('retrieval_mask_tokens里面的token_ids:', output.outputs[0].token_ids)
                        retrieval_mask_tokens = [[1] * len(output.outputs[0].token_ids)] #坑的地方在于这个token id是包含</search>的，但是text里面不包含
                        # retrieval_mask_tokens[:len(input_token_ids)] = [0] * len(input_token_ids)
                    else:
                        retrieval_mask_tokens = continued_answer[q]['retrieval_mask_tokens'].copy()
                        linshi = []
                        for i in range(len(output.outputs[0].token_ids)):
                            linshi.append(1)
                        retrieval_mask_tokens.append(linshi)
                    # print('actual_response_ids:', actual_response_ids)
                    if t == 8: #检索次数太多了，直接停掉，就是未完成
                        original_data = {
                            "idx":idx,
                            "prompt_ids":input_token_ids,
                            "response_ids":output_token_ids,
                            "retrieve_num_count" : retrieve_num_count,
                          #  "actual_response_ids": actual_response_ids,
                            "retrieval_mask_tokens": [item for sublist in retrieval_mask_tokens for item in sublist],
                            "gold_answer": gold_answer
                                }

                        if original_data["response_ids"] == None:
                            print(f"Here 1,Why this response_ids is None???,{original_data}")
                            time.sleep(10)
                        finished_texts.append(original_data)
                        continue

                    if "<search>" in generated_text and stop_reason=="</search>":
                        # print('打印查询query', generated_text)
                        query = generated_text.split("<search>")[-1].split("</search>")[0]
                        query = query.replace('"',"").strip()
                        query = " ".join(query.split())

                        if query: #开始检索

                            m = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(" "+generated_text.strip()))
                            m.extend([690,1836,1339])
                            retrieval_mask_tokens[-1] = [1] * len(m)
                            all_token_ids = list(output.prompt_token_ids) + m
                            output_token_ids = all_token_ids[len(input_token_ids):] #因为我们这里实际上对output进行了修改，所以要更改一下.. 

                            query_list.append(query)
                            retrieve_num_count += 1
                            original_data = {
                                "idx":idx,
                                "prompt": prompt + " " + generated_text.strip(), #这里r1-search用了generate_text.strip()，为什么要用这个呢..
                                "prompt_ids":input_token_ids,
                                "response_ids":output_token_ids,
                                "retrieve_num_count" : retrieve_num_count,
                               # "actual_response_ids": actual_response_ids,
                                "retrieval_mask_tokens": retrieval_mask_tokens,
                                "gold_answer": gold_answer
                                }
                            if original_data["response_ids"] == None:
                                print(f"Here 2,Why this response_ids is None???,{original_data}")
                                time.sleep(10)
                            continued_texts.append(original_data)
                            # print("(1)ori-dict:",original_data)
                        else: #这个是query没有按照规定格式的 ，直接停止了，之后可能需要优化
                            original_data = {
                            "idx":idx,
                            "prompt_ids":input_token_ids,
                            "response_ids":output_token_ids,
                            "retrieve_num_count" : retrieve_num_count,
                        #    "actual_response_ids": actual_response_ids,
                            "retrieval_mask_tokens": [item for sublist in retrieval_mask_tokens for item in sublist],
                            "gold_answer": gold_answer
                                }
                            if original_data["response_ids"] == None:
                                print(f"Here 3,Why this response_ids is None???,{original_data}")
                                time.sleep(10)
                            finished_texts.append(original_data)
                            # print("(2)ori-dict:",original_data)
                    else: #生成结束                     # 没生成 query，说明结束了
                        original_data = {
                        "idx":idx,
                        "prompt_ids":input_token_ids,
                        "response_ids":output_token_ids,
                        "retrieve_num_count" : retrieve_num_count,
                       # "actual_response_ids": actual_response_ids,
                        "retrieval_mask_tokens": [item for sublist in retrieval_mask_tokens for item in sublist],
                        "gold_answer": gold_answer
                        }
                        finished_texts.append(original_data)
                        if original_data["response_ids"] == None:
                            print(f"Here 4,Why this response_ids is None???,{original_data}")
                            time.sleep(10)
                        # print("(3)ori-dict:",original_data)

                assert len(query_list) == len(continued_texts), "Error in len of query_list and continued_texts"
                # print("query-list:",query_list)
                if len(query_list)!=0:
                    topk = 5

                    response = requests.post(url_wiki, json={"queries": query_list, "k": topk}) # query_list实际上是query的列表。其它的似乎无法传入列表，要改成循环
                    
                    if response.status_code == 200:
                        print('查询结束，查询成功')
                        result = response.json()
                        answers = result["answers"]
                        for k in range(len(answers)):
                            retrieve_docs = answers[k] # answers[0]里包含topk个response。这里的k：第k个查询，第k个答案，一一对应
                            continued_text_now = copy.deepcopy(continued_texts[k]) #这里为什么是index k
                            if len(retrieve_docs)>0:
                                doc_content_list = []
                                for j in range(len(retrieve_docs)):
                                    doc_now = re.sub(r'^\d+\s+', '', retrieve_docs[j])
                                    doc_content_list.append(f"({j+1}){doc_now}\n")
                                doc_content = ''.join(doc_content_list)
                            else:
                                doc_content = "None"

                            continued_text_now["prompt"] = continued_text_now["prompt"] + " </search>\n\n"+ "<information>\n" +  doc_content + "</information>\n\n"

                            ### retrieval mask 的设置
                            m = []
                            for li in self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("<information>\n" +  doc_content + "</information>\n\n")):
                                  m.append(0)
                            continued_text_now["retrieval_mask_tokens"].append(m)
                            continued_texts[k] = continued_text_now
                            ### retrieval mask 的设置
                            # print("continued_text_now:",continued_text_now)
                    else:
                        raise Exception("Error in response: the status code is not 200!")

                finished_all_list.extend(finished_texts)             # 将本轮完成的样本加入列表
                # 如果已经全部完成，则退出循环
                if len(continued_texts)==0:
                    if len(finished_all_list) != len(idx_w_prompt_part):
                        time.sleep(20)
                        print("finished_all_list:",finished_all_list)
                        print("idx_w_prompt_part:",idx_w_prompt_part)
                        time.sleep(20)
                    assert len(finished_all_list) == len(idx_w_prompt_part) , "Error in len of finished_all_list and idx_w_prompt_part"
                    all_outputs.append(finished_all_list)
                    break
                else:                 # 否则继续下一轮生成
                    data_keys_again = continued_texts[0].keys()
                    ds = Dataset.from_dict({key: [d[key] for d in continued_texts] for key in data_keys_again})
                    continued_answer = copy.deepcopy(continued_texts)
        # 整合所有输出并排序
        all_outputs_cat = [item for sublist in all_outputs for item in sublist]  # 将所有子列表展开为一个完整列表
        if len(all_outputs_cat) != len(prompts_w_idx_dict):   # 检查展开后样本数量是否一致
            time.sleep(20)
            print("[all_outputs_cat,prompts_w_idx_dict]:",[all_outputs_cat,prompts_w_idx_dict])
            time.sleep(20)
        assert len(all_outputs_cat) == len(prompts_w_idx_dict), "Error in len of all_outputs and prompts_w_idx_dict"
        all_outputs_sorted = sorted(all_outputs_cat, key=lambda x: x['idx'])  # 按 idx 排序，保证顺序一致

        samples_list = []
        for i in range(0, len(all_outputs_sorted), args.micro_rollout_batch_size): # 按 micro batch 划分样本
            outputs = all_outputs_sorted[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:  # 不进行 packing 的逻辑（可拓展）
                pass
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                # NOTE: 将所有 prompt + response 拼接为一个长序列，模型可高效一次性处理多个样本
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []  # 拼接后的 token 序列
                packed_seq_lens = []  # 每条 sample 的总长度
                attention_mask = []  # 标记哪些 token 属于哪个样本
                retrieve_mask = []  # 标记哪些 token 是检索内容（为 0）
                num_actions = []  # 每个 sample 的 response token 数量（最小为 1）
                exit_flags = []  # 每个 sample 的 correction flag  记录如果这个sample是被纠正的，那么需要给他多加一点reward
                answer_nums = []
                retrieve_num = []  # 每条样本的检索次数
                pure_response_length_lis = []  # response 中非检索内容的 token 数
                
                for i, output in enumerate(outputs):
                   # try:
                        input_len = len(output["prompt_ids"])  # prompt 长度
                        #output_len = len(output["response_ids"])  # response 长度
                        exit_flag = 0 
                        answer_nums.append(0)
                        output_len = len(output["response_ids"])  # response 长度
                        packed_seq_lens.append(input_len + output_len)  # 总长度记录

                        sequences.extend(output["prompt_ids"] + output["response_ids"])  # 拼接输入输出
                        attention_mask.extend([i + 1] * (input_len + output_len))  # 为每个 token 打上 batch id

                        response_seq = output["response_ids"]
                        retrieve_mask_now = [1] * len(response_seq)  # 初始全为 1，表示非检索内容

                        # 判断使用的 tokenizer 类型（LLaMA vs Qwen），决定标记 token
                        if output["prompt_ids"][0] == 128000:  # llama 模型
                            start_tokens = [27, 91, 7413, 3659, 77027, 91, 397]# <|begin_of_documents|> 对应的 token
                            end_tokens = [408, 3659, 77027, 91, 1363]# <|end_of_documents|> 对应的 token
                        else:  # qwen 模型
                            # start_tokens = [27, 91, 7265, 3575, 75927, 91, 397]  # <|begin_of_documents|> 对应的 token
                            # end_tokens = [408, 3575, 75927, 91, 1339]  # <|end_of_documents|>
                            start_tokens = [27, 25069, 397] # <information>\n
                            end_tokens = [522, 25069, 1339] # </information>\n

                        is_in_masking = False  # 是否正在标记中
                        mask_start_idx = -1
                        start_count = 0
                        end_count = 0
                        # 1. 找出所有 <information>...</information> 的区间
                        # info_spans = []
                        # info_start = None
                        for m in range(len(response_seq)):
                            # 匹配开始 token 序列
                            if (not is_in_masking and m + len(start_tokens) <= len(response_seq) and
                                response_seq[m:m + len(start_tokens)] == start_tokens):
                                start_count += 1
                                is_in_masking = True
                                mask_start_idx = m
                                for n in range(mask_start_idx, mask_start_idx + len(start_tokens)):
                                    retrieve_mask_now[n] = 0  # 将开始 token 全部设为 0
                                # info_start = m
                            if is_in_masking:
                                retrieve_mask_now[m] = 0  # 检索段中 token 设置为 0
                                # 匹配结束 token 序列
                                if (m + len(end_tokens) <= len(response_seq) and
                                    response_seq[m:m + len(end_tokens)] == end_tokens):
                                    end_count += 1
                                    is_in_masking = False
                                    for u in range(m, m + len(end_tokens)):
                                        retrieve_mask_now[u] = 0  # 将结束 token 全部设为 0
                                    # if info_start is not None:
                                    #     info_spans.append((info_start, m + len(end_tokens) - 1))
                                    #     info_start = None
                                    mask_start_idx = -1
                        if start_count == end_count == output["retrieve_num_count"]:
                            # if(mask != retrieve_mask_now):
                            #     print('出问题了')
                            if( output["retrieval_mask_tokens"] !=retrieve_mask_now ):
                                print('新设置的retrieval mask出问题')
                            pass
                        else:
                            print(f"Important Bug!! Model genearte the retrieve symbols!! The num:{[start_count,end_count,output['retrieve_num_count']]},The detailed content:{output}")
                            # time.sleep(3)
                            # assert start_count == end_count == output["retrieve_num_count"], "Model genearte the retrieve symbols"

                        retrieve_mask.extend(output["retrieval_mask_tokens"])  # 添加当前 sample 的 mask
                        num_actions.append(max(1, output_len))  # 添加 response 长度
                        retrieve_num.append(output["retrieve_num_count"])  # 添加检索次数
                        exit_flags.append(exit_flag)
                        pure_response_length_lis.append(sum(output["retrieval_mask_tokens"]))  # 统计非检索 token 数
                    #     print('打印模型的retrieval mask:', retrieve_mask_now)
                    #     # print('打印模型的actual_response_ids:', actual_response_ids)
                    #    # print('打印模型的mask:', mask)
                    #     print('打印模型的output["retrieval_mask_tokens"]:', output["retrieval_mask_tokens"])
                    #     print('打印模型的response_seq:', self.tokenizer.decode(response_seq))
                    #     print('打印模型的response_seq id:', response_seq)
                        assert len(output["retrieval_mask_tokens"]) == len(response_seq), "Error in len of retrieval_mask_tokens and response_seq"
                    # except Exception as e:
                    #     print(f"Error occur:{e}, the data: {output},{output['prompt_ids']},{output['response_ids']}")


                # 转为 Tensor 形式
                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None  # 当前未使用 action_mask，貌似是因为已经使用了retrieve_mask来替代了action mask的功能。
                retrieve_mask = torch.tensor(retrieve_mask, device="cuda").unsqueeze(0)

                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)

                retrieve_nums = torch.tensor(retrieve_num, device="cuda", dtype=torch.float)
                pure_response_length = torch.tensor(pure_response_length_lis, device="cuda", dtype=torch.float)
                exit_flags = torch.tensor(exit_flags, device="cuda", dtype=torch.float)
                answer_nums = torch.tensor(answer_nums, device="cuda", dtype=torch.float)
                # 构建 Samples 对象加入列表
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        retrieve_mask=retrieve_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        retrieve_num=retrieve_nums,
                        pure_response_length=pure_response_length,
                        exit_flag=exit_flags,
                        answer_num=answer_nums
                    )
                )


        return samples_list
    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None
