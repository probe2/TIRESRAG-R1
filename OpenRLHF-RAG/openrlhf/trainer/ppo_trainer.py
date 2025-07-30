import os
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

# from len_rag.other_codes.ReSearch.verl.verl.workers import critic
from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, ValueLoss
from openrlhf.models.utils import masked_mean, unpacking_samples, compute_approx_kl
from openrlhf.utils.distributed_sampler import DistributedSampler

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveExperienceMaker, NaiveReplayBuffer
from torch.distributed import all_reduce, ReduceOp
import torch, math, deepspeed

def fallback_global_grad_norm(engine, norm_type=2):
    """Return global grad-norm for any DeepSpeed version."""
    total_norm_sq = torch.zeros(1, device=torch.cuda.current_device())

    for p in engine.module.parameters():           # engine.module 是裸模型
        # 找到梯度张量 (ZeRO-0/1/2/3 兼容)
        # print('打印p',p)
        grad = None
        if p.grad is not None:
            grad = p.grad
        elif hasattr(p, 'ds_grad') and p.ds_grad is not None:            # ZeRO‐2
            grad = p.ds_grad
        elif hasattr(p, 'ds_tensor') and p.ds_tensor.grad is not None:   # ZeRO‐3
            grad = p.ds_tensor.grad

            
        if grad is not None:
            print(f"grad norm = {grad.data.norm(2).item()}")
        else:
            print(f" grad is None")        
            continue
        with deepspeed.zero.GatheredParameters([p], enabled=getattr(grad, 'is_sharded', False)):
            total_norm_sq += grad.data.float().norm(norm_type) ** norm_type

    # 还要把各 rank 的 norm 做一次 all-reduce
    all_reduce(total_norm_sq, op=ReduceOp.SUM)
    return math.pow(total_norm_sq.item(), 1.0 / norm_type)

class PPOTrainer(ABC):
    """
    Trainer for Proximal Policy Optimization (PPO) algorithm.

    Args:
        strategy (Strategy): The training strategy to use.
        actor (Actor): The actor model in the PPO algorithm.
        critic (nn.Module): The critic model in the PPO algorithm.
        reward_model (nn.Module): The reward model for calculating rewards in the RLHF setup.
        initial_model (Actor): The initial model for reference logits to limit actor updates in RLHF.
        ema_model (Actor): The exponential moving average model for stable training.
        actor_optim (Optimizer): The optimizer for the actor model.
        critic_optim (Optimizer): The optimizer for the critic model.
        actor_scheduler (Scheduler): The learning rate scheduler for the actor.
        critic_scheduler (Scheduler): The learning rate scheduler for the critic.
        ema_beta (float, defaults to 0.992): EMA decay rate for model stability.
        init_kl_coef (float, defaults to 0.001): Initial coefficient for KL divergence.
        kl_target (float, optional): Target value for KL divergence.
        kl_horizon (int, defaults to 10000): Horizon for KL annealing.
        ptx_coef (float, defaults to 0): Coefficient for supervised loss from pre-trained data.
        micro_train_batch_size (int, defaults to 8): Micro-batch size for actor training.
        buffer_limit (int, defaults to 0): Maximum size of the replay buffer.
        buffer_cpu_offload (bool, defaults to True): If True, offloads replay buffer to CPU.
        eps_clip (float, defaults to 0.2): Clipping coefficient for policy loss.
        value_clip (float, defaults to 0.2): Clipping coefficient for value function loss.
        micro_rollout_batch_size (int, defaults to 8): Micro-batch size for generating rollouts.
        gradient_checkpointing (bool, defaults to False): If True, enables gradient checkpointing.
        max_epochs (int, defaults to 1): Number of epochs to train.
        max_norm (float, defaults to 1.0): Maximum gradient norm for gradient clipping.
        tokenizer (Callable, optional): Tokenizer for input data.
        prompt_max_len (int, defaults to 128): Maximum length for prompts.
        dataloader_pin_memory (bool, defaults to True): If True, pins memory in the data loader.
        remote_rm_url (str, optional): URL for remote reward model API.
        remote_sufficient_url (str, optional): URL for remote sufficient API.
        reward_fn (Callable, optional): Custom reward function for computing rewards.
        **generate_kwargs: Additional arguments for model generation.
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None,
        remote_sufficient_url: str = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        **generate_kwargs,
    ) -> None:
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn

        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.remote_sufficient_url = remote_sufficient_url
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        self.experience_maker = NaiveExperienceMaker(
            actor,
            critic,
            reward_model,
            initial_model,
            tokenizer,
            prompt_max_len,
            self.kl_ctl,
            strategy,
            remote_rm_url,
            remote_sufficient_url,
            reward_fn,
        )#experience_maker 负责：Actor 生成序列（策略）。critic 评估价值（基于 RLHF 评分）。Reward Model 计算奖励（可本地或远程）。存储 PPO 训练数据（Advantage 估计）。

        packing_samples = getattr(self.args, "packing_samples", False)
        self.replay_buffer = NaiveReplayBuffer(
            strategy, micro_train_batch_size, buffer_limit, buffer_cpu_offload, packing_samples
        )
        # replay_buffer 负责：存储多个 PPO 训练周期的数据。经验重放（Replay Buffer），用于 mini-batch 训练。避免策略更新太快，提高数据利用率。
        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available

        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(
        self,
        args,
        prompts_dataloader, #用于 RL 训练的提示数据。
        pretrain_dataloader, #用于监督学习的数据（可选）。
        consumed_samples=0,
        num_update_steps_per_episodes=1,
    ) -> None:
        # jjh add >>>
        print(num_update_steps_per_episodes, args.train_batch_size, args.max_epochs, args.rollout_batch_size, args.n_samples_per_prompt)
        # <<<
        num_rollouts_per_episodes = (       #计算 每轮 PPO 训练所需的 rollout 轮数。
            num_update_steps_per_episodes
            * args.train_batch_size
            // args.max_epochs
            // args.rollout_batch_size
            // args.n_samples_per_prompt
        ) #rollout 代表针对某个 prompt 生成的回复（response），并用于强化学习训练。在 PPO + RLHF 训练框架 下，每个 rollout 过程对应 模型与环境（即 Prompt + Reward Model）的一次完整交互。
        if(num_rollouts_per_episodes == 0):
            num_rollouts_per_episodes = 1
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        # Restore step and start_epoch
        steps = consumed_samples // args.rollout_batch_size + 1
        # jjh add >>>
        print(args.rollout_batch_size, num_rollouts_per_episodes)
        # <<<
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)
        total_training_steps = len(prompts_dataloader)  * args.num_episodes
        print('总共训练次数', total_training_steps)
        for episode in range(start_episode, args.num_episodes): # num_episodes 是训练的轮数,也就是epoch数目。
            if episode  > start_episode:
                tag = f"global_step{steps}"
                # self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)
                client_states = {"consumed_samples": steps * args.rollout_batch_size}
                self.strategy.save_ckpt(
                    self.actor.model,
                    os.path.join(args.ckpt_path, "_actor"),
                    tag,
                    args.max_ckpt_num,
                    args.max_ckpt_mem,
                    client_states,
                )
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )#这段代码的目的是 确保 DistributedSampler 在多 GPU 分布式训练时，每个 epoch 采样的数据一致，并在恢复训练时正确跳过已消费的样本。
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for rand_prompts in self.prompts_dataloader: # 采样多个 prompt（类似 batch） 就是整个dataset了
                # print('打印rand_prompts 以及rank', rand_prompts[0], self.strategy.get_rank())
                for i, experience in enumerate(
                    self.experience_maker.make_experience_list(rand_prompts, steps, self._wandb,total_training_steps = total_training_steps, **self.generate_kwargs)
                ):
                    # if i == 0:
                    #     output = self.tokenizer.batch_decode(
                    #         experience.sequences[0].unsqueeze(0), skip_special_tokens=True
                    #     )
                    #     self.strategy.print(output) #这里没有任何影响
                    self.replay_buffer.append(experience)  # 将数据存入 replay buffer
                #print('replay_buffer 添加数据长度',len(self.replay_buffer),self.strategy.get_rank())
                #torch.cuda.empty_cache()
                # self.replay_buffer.normalize("advantages", self.strategy)
                if self.strategy.args.enable_accuracy_filter:
                    if self.replay_buffer.is_full(): ##这里感觉还是有问题，会导致打印的step不对
                        print('打印更新的steps', steps)
                        print('replay_buffer 添加数据长度前',len(self.replay_buffer),self.strategy.get_rank())
                        self.replay_buffer.padding() #要补足replay buffer 的长度
                        print('replay_buffer 添加数据长度后',len(self.replay_buffer),self.strategy.get_rank())
                        if self.args.advantage_estimator != "group_norm":
                            self.replay_buffer.normalize("advantages", self.strategy)
                        status = self.ppo_train(steps) #每隔freezing_actor_steps 更新一次vllm engine
                        #print('ppo_train 结束',self.strategy.get_rank())
                        self.replay_buffer.clear()
                        torch.cuda.empty_cache()
                        #print('replay_buffer 清空',self.strategy.get_rank())
                    else:
                        status = {}
                else:
                    if self.args.advantage_estimator != "group_norm":
                        self.replay_buffer.normalize("advantages", self.strategy)
                    status = self.ppo_train(steps) #每隔freezing_actor_steps 更新一次vllm engine
                    self.replay_buffer.clear()
                    torch.cuda.empty_cache()
                # torch.distributed.barrier()
                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
                pbar.set_postfix(status) #set_postfix 是 tqdm 进度条的一个方法，用于在进度条后面显示额外的信息。它可以动态更新进度条右侧的状态信息。
                # print('打印logs前面的status', status, self.strategy.get_rank())
                # logs/checkpoints
                client_states = {"consumed_samples": steps * args.rollout_batch_size}
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                pbar.update()
                steps = steps + 1

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def ppo_train(self, global_steps=0): # actorppotrainer 继承了ppo_trainer了，这个相当于一个step的训练。
        torch.cuda.empty_cache()
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer, # 用experience_maker 生成的经验数据进行训练
            batch_size=self.replay_buffer.sample_batch_size, #这个是micro_train_batch_size 默认为1.. 
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs): #每个ppo step里面还有epoch和step.. #在 PPO 训练阶段，对 Replay Buffer 里的数据进行多次学习：
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar: #让 PPO 训练时，一次读取 一个 mini-batch 的 experience，并更新 Actor。对这个一个mini-batch的experience,每次都actor进行更新，但是vllm_engine不更新，直到所有的mini-batch experience都训练完。
                experience.to_device(device)

                # count_print = 0 
                # for name, param in self.actor.named_parameters():
                #     print(f"Param: {name}, mean: {param.data.float().mean().item()}, shape: {param.shape}, rank: {self.strategy.get_rank()}")
                #     # 只打印前几个参数，避免输出太多
                #     count_print+=1
                #     if count_print > 10:
                #         break
                # print('training_step 内开始',self.strategy.get_rank())
                status = self.training_step(experience, global_steps)
                #print('training_step 内内部结束',self.strategy.get_rank())
                # for DP
                # weighted mean for kl
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "rm": status["reward"],
                        "ret": status["return"],
                        "glen": status["response_length"],
                        "tlen": status["total_length"],
                        "kl": status["kl"],
                        "act_lr": status["actor_lr"],
                    }

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)
        #print('ppo_train 内部结束',self.strategy.get_rank())
        # torch.distributed.barrier()
        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    # if('return' in k):
                        # print('打印均值计算里面的return', v)
                    status_mean[k] += v
            for k in status_mean.keys():
                # print('打印status_mean均值以及长度',k, status_mean[k], len(status_list), self.strategy.get_rank(),status_mean[k]/len(status_list))
                status_mean[k] /= len(status_list) #因为advantage有正有负，所以batch_size*n_samples_per_prompt*max_epochs 个经验，所以均值计算的时候，需要除以这个数，然后得到的return mean就很小..
        # print('打印计算出来的平均return', status_mean['return'], len(status_list))
        torch.cuda.empty_cache()
        return status_mean

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        status = {}
        if global_steps > self.freezing_actor_steps:
            status = self.training_step_actor(experience)
        if self.critic is not None:
            status.update(self.training_step_critic(experience))
        return status

    def training_step_actor(self, experience: Experience) -> Dict[str, float]:
        """
        输入参数 experience：
        experience.sequences：输入序列（tokens）。
        experience.action_log_probs：先前策略的 log_probs（用于 PPO 策略损失计算）。
        experience.advantages：Advantage 值（衡量当前策略相对于基准的改进）。
        experience.action_mask：用于掩码 padding 的 token。
        experience.base_action_log_probs（可选）：参考模型的 log_probs（用于 KL 约束）。
        """
        self.actor.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list): #如果 experience.sequences 是 list，表示 packed 数据，单独处理。
            #这是packed 分支，暂时没必要看
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
            advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
            # action_mask = torch.cat(experience.action_mask, dim=0).unsqueeze(0) #None
            retrieve_mask = torch.cat(experience.retrieve_mask, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = torch.cat(experience.base_action_log_probs, dim=0).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_action_log_probs = experience.action_log_probs
            advantages = experience.advantages
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = experience.base_action_log_probs

        action_log_probs, output = self.actor( #让 Actor 计算当前策略下每个 token 的 log_probs，用于 PPO 训练：
            sequences,
            num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )

        # # loss function
        actor_loss = self.actor_loss_fn( # 
            action_log_probs, # 当前策略 π_new(a | s, θ) 的 log_probs
            old_action_log_probs,  # 先前策略 π_old(a | s, θ_old) 的 log_probs #限制策略优化步长，防止不稳定	
            advantages, # 优势函数，衡量当前策略相对于基准的改进，advantage主要是由reward计算来的，以及critic model。
            action_mask=experience.action_mask, #对于packing的，这个是none
            retrieve_mask=retrieve_mask,
        )

        if self.args.use_kl_loss:
            print('进入kl分支',self.strategy.get_rank())
            if self.initial_model is not None:
                kl, original_kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    action_mask=experience.action_mask,
                    retrieve_mask=retrieve_mask,
                    use_kl_estimator_k3 = self.args.use_kl_estimator_k3,
                    use_kl_estimator_k2 = self.args.use_kl_estimator_k2,
                )

            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device = action_log_probs.device)

            if not self.args.packing_samples:
                kl_mean = masked_mean(kl, experience.action_mask, dim=-1)
            else:
                # convert tensor into list of tensors so that it's easier to manipulate
                # within dataset.
                kl = unpacking_samples(kl, num_actions)
                # kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device = action_log_probs.device)
                kl_mean = torch.zeros(len(kl), device=action_log_probs.device)
                retrieve_mask_unpack = unpacking_samples(retrieve_mask, num_actions)
                print('打印kl_mean', kl_mean.shape, action_log_probs.shape)
                for i, each_kl in enumerate(kl):
                    print('打印每个kl',  each_kl.shape,  retrieve_mask_unpack[i].shape)
                    kl_mean[i] = masked_mean(each_kl, experience.action_mask, retrieve_mask=retrieve_mask_unpack[i], dim=-1)

                original_kl = unpacking_samples(original_kl, num_actions)
                original_kl_mean = torch.zeros(len(original_kl), device=action_log_probs.device)
                for i, each_original_kl in enumerate(original_kl):
                    # print('打印每个original_kl',  each_original_kl.shape,  retrieve_mask_unpack[i].shape)
                    original_kl_mean[i] = masked_mean(each_original_kl, experience.action_mask, retrieve_mask=retrieve_mask_unpack[i], dim=-1)

            kl_loss = kl_mean.mean()
            original_kl_loss = original_kl_mean.mean()
            experience.info["kl"] = kl_loss.item() #如果进入了kl out reward分支，那么是在这里会重新赋值kl给info.
            # experience.info["original_kl"] = original_kl_loss.item()
        else:
            kl_loss = 0


        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        # loss = actor_loss + aux_loss * self.args.aux_loss_coef
        loss = actor_loss + aux_loss * self.args.aux_loss_coef + kl_loss * self.kl_ctl.value
        # print('打印loss_rank', self.strategy.get_rank(), loss)
        #print('backward 开始',self.strategy.get_rank())
        self.strategy.backward(loss, self.actor, self.actor_optim)
        #print('backward 结束',self.strategy.get_rank())
        ###打印梯度
        # grad_norm = global_grad_norm(self.actor.model)   # ← 旧版 DeepSpeed 也能跑
        # print(f"{self.strategy.get_rank()} grad_norm = {grad_norm:.4f}")
        # engine = self.actor.model
        # grad_norm = (
        #     engine.optimizer.get_global_grad_norm()  # 新版
        #     if hasattr(engine.optimizer, "get_global_grad_norm")
        #     else fallback_global_grad_norm(engine)   # 旧版
        # )
        # print(f"{self.strategy.get_rank()} grad_norm = {grad_norm:.4f}")
        # ptx loss
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                attention_mask.bool(),
                inputs,
                self.ptx_loss_fn.IGNORE_INDEX,
            )

            output = self.actor(inputs, attention_mask=attention_mask, return_output=True)
            ptx_log_probs = output["logits"]

            # loss function
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            # mixtral
            if self.aux_loss:
                aux_loss = output.aux_loss
            else:
                aux_loss = 0
            loss = ptx_loss + aux_loss * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * loss, self.actor, self.actor_optim)
        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        # torch.distributed.barrier()
        # print('actor 更新结束',self.strategy.get_rank())
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        status = {"policy_loss": actor_loss.item(), "actor_lr": self.actor_scheduler.get_last_lr()[0],'original_kl_loss':original_kl_loss.item() if self.args.use_kl_loss else 0}
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss.item()
        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                print('打印status的k前面', k, v, self.strategy.get_rank()) 
                status[k] = v.mean().item() #这里的return其实上是算上kl的reward的sum值，而这里的reward是单纯的由reward_fn计算出来的。
                
                print('打印status[k]的k后面', status[k], self.strategy.get_rank())
        return status

    def training_step_critic(self, experience: Experience) -> Dict[str, float]:
        self.critic.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_values = torch.cat(experience.values, dim=0).unsqueeze(0)
            returns = torch.cat(experience.returns, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_values = experience.values
            returns = experience.returns
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask

        # critic loss
        values, output = self.critic(
            sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )
        # loss function
        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask,
        )
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, experience.action_mask, experience.retrieve_mask).item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }
        return status

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                if self.experience_maker.perf_stats is not None:
                    logs.update({f"perf/experience_maker/{k}": v for k, v in self.experience_maker.perf_stats.items()})
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
                if self.experience_maker.perf_stats is not None:
                    for k, v in self.experience_maker.perf_stats.items():
                        self._tensorboard.add_scalar(f"perf/experience_maker/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0:
            # self.evaluate(self.eval_dataloader, global_step)
            pass
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)

    def _save_checkpoint(self, args, tag, client_states):
        # self.strategy.save_ckpt(
        #     self.actor.model,
        #     os.path.join(args.ckpt_path, "_actor"),
        #     tag,
        #     args.max_ckpt_num,
        #     args.max_ckpt_mem,
        #     client_states,
        # )
        self.strategy.save_model(self.actor.model, self.tokenizer, os.path.join(args.ckpt_path, "_actor_model"))
        if self.critic is not None:
            self.strategy.save_ckpt(
                self.critic, os.path.join(args.ckpt_path, "_critic"), tag, args.max_ckpt_num, args.max_ckpt_mem
            )
