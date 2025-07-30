from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .ring_attn_utils import convert_ring_attn_params
from .utils import log_probs_from_logits, reset_position_ids


class Actor(nn.Module):
    """
    Base class for Actor models in reinforcement learning.

    This class serves as a foundation for implementing various actor models, which are responsible for selecting actions based on the policy learned from the environment.

    Args:
        pretrain_or_model (str or nn.Module): 预训练模型路径，或已加载的模型实例。
        use_flash_attention_2 (bool, optional): 是否启用 Flash Attention 2。默认 False。
        bf16 (bool, optional): 是否使用 bfloat16 计算。默认 True。
        load_in_4bit (bool, optional): 是否以 4-bit 量化加载模型。默认 False。
        lora_rank (int, optional): LoRA 低秩分解的秩。默认 0（不启用）。
        lora_alpha (int, optional): LoRA 的 α 参数。默认 16。
        lora_dropout (float, optional): LoRA Dropout 比例。默认 0。
        target_modules (list, optional): 需要应用 LoRA 的目标模块。
        ds_config (dict, optional): DeepSpeed 配置。
        device_map (dict, optional): 指定模型加载到哪些设备。
        packing_samples (bool, optional): 是否启用样本 Packing（加速训练）。
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config) # 如果 DeepSpeed 启用了 ZeRO-3（参数分片优化），则初始化 HfDeepSpeedConfig。
            else:
                dschf = None

            if load_in_4bit: #这个没什么用
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            self.model = AutoModelForCausalLM.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False #Transformer Decoder 计算时，默认会缓存之前的 key-value，以加速生成。
                                    #但在 强化学习（RLHF）训练 中，每个 forward() 可能涉及 不同的输入长度，如果 KV 缓存未正确管理，可能会导致错误。


            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
        else:
            self.model = pretrain_or_model

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        generate_args = {
            "input_ids": input_ids,              # 输入token IDs
            "top_k": kwargs.get("top_k", None),  # top-k采样参数
            "top_p": kwargs.get("top_p", None),  # top-p(nucleus)采样参数
            "do_sample": kwargs.get("do_sample", True),  # 是否使用采样生成
            "early_stopping": True,              # 提前停止生成
            "temperature": kwargs.get("temperature", 1),  # 温度参数，控制采样随机性
            "use_cache": True,                   # 使用KV缓存加速生成
            "num_beams": kwargs.get("num_beams", 1),  # 束搜索的束宽
            "attention_mask": kwargs.get("attention_mask"),  # 注意力掩码
            "eos_token_id": kwargs.get("eos_token_id"),  # 结束符ID
            "pad_token_id": kwargs.get("pad_token_id"),  # 填充符ID
            "min_new_tokens": kwargs.get("min_new_tokens", 1),  # 最少生成的新token数
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        # Call generate
        sequences = self.model.generate(**generate_args)

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        return self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        # 创建注意力掩码：标记非EOS和非PAD的位置为1
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # 查找每个序列的结束位置并插入EOS token
        # 这段注释说明了以下代码的逻辑：从后往前找到最后一个有效token的位置，
        # 在其后添加EOS token以确保序列正确终止

        # The following code is equivalent to:
        #
        # for i in range(attention_mask.size(0)):
        #     for t in reversed(range(seq_length)):
        #         if attention_mask[i][t] > 0.5:
        #             attention_mask[i][min(t + 1, seq_length - 1)] = True
        #             sequences[i][min(t + 1, seq_length - 1)] = eos_token_id
        #             break
        #
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)
        # 处理Llama3和Qwen2模型可能在提示中间包含EOS token的情况
        # 找到每个序列的第一个有效token位置

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        # 创建位置索引张量

        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        # 更新注意力掩码：只保留从第一个有效token到EOS token的部分

        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)
        # 在RL中，state_i (当前token) + action_i (下一个token) -> state_i+1 (下一个token)
        # 提取状态序列，从输入末尾到生成序列倒数第二个token

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        # 创建动作掩码：标记非EOS和非PAD的位置(有效动作)

        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        # 确保第一个位置始终有效,过滤奖励计算: 在后续的强化学习训练中，只对掩码为1的位置计算奖励和更新策略
        # 至少一个有效动作: 确保每个样本至少有一个有效动作用于学习，即使生成的序列很短或立即遇到EOS
        # 初始响应保证: 在强化学习中，模型必须至少对输入提示产生一个响应token
        # 避免零梯度: 如果所有位置都无效，可能导致梯度消失，使训练无效
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Optional[Union[int, list[int]]] = None, #如果是 int：表示所有样本的 num_actions 相同，例如 num_actions=3，表示每个序列的最后 3 个 token 需要计算 log_probs。
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Returns action log probs"""
        if not self.packing_samples:
            # 从注意力掩码生成位置ID：累加掩码值减1
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
            # 将无效位置(掩码为0)填充为1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            # 如果使用样本打包：

            # convert attention_mask to position_ids
            if ring_attn_group is not None:
                # 分布式情况下转换参数
                sequences, attention_mask, position_ids = convert_ring_attn_params(
                    sequences, attention_mask, packed_seq_lens, ring_attn_group
                )
            else:
                # 从注意力掩码重置位置ID
                position_ids = reset_position_ids(attention_mask)
            # explicitly ignore attention_mask for packing_samples
            attention_mask = None
        # 前向传播，获取模型输出

        output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids) #在generate时，对packing samples实际上是用position_ids来判断
                                                                                                 #对不同的输入进行不同的输出。
         # 如果不需要计算动作概率，直接返回输出
        if num_actions is None:
            assert return_output
            return output
        # 从logits计算对数概率：将每个时间步的预测与下一个实际token比较
        # 这计算了序列中每个token(除了最后一个)预测下一个token的对数概率

        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:]) #表示每个时间步上正确标签的对数概率,[batch_size,seq_len-1]
        # 提取动作的对数概率
        # 不使用样本打包：直接取最后num_actions个位置的概率

        if not self.packing_samples:
            action_log_probs = log_probs[:, -num_actions:]
        else:
            # 使用样本打包：需要计算每个样本的每个位置的概率
            assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
            action_log_probs = []
            offset = 0
            # 遍历每个打包的样本

            for num_action, seq_len in zip(num_actions, packed_seq_lens):
                start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                # 计算每个样本的有效动作范围
                action_log_probs.append(log_probs[:, start:end])
                # 提取该样本的动作概率
                offset += seq_len
                # 更新偏移量

            action_log_probs = torch.cat(action_log_probs, dim=1) #依次计算

        if return_output:
            return (action_log_probs, output)
        else:
            return action_log_probs
    # 启用梯度检查点以节省内存

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    # 禁用梯度检查点

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
