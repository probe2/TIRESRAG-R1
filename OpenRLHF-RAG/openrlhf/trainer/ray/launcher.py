import logging
import os
import socket
from typing import Callable, Dict, List, Optional, Type

import ray
import torch
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.utils.deepspeed import DeepspeedStrategy

from openrlhf.trainer.ray.utils import ray_noset_visible_devices

class DistributedTorchRayActor: # 初始化分布式训练环境，确保多个 GPU 进程可以正确通信
    def __init__(self, world_size, rank, master_addr, master_port):
        """
        :param world_size: 参与分布式训练的总进程数（总 GPU 数）
        :param rank: 当前进程的 Rank（唯一 ID）
        :param master_addr: 主节点 IP 地址（如果未提供，将自动获取）
        :param master_port: 主节点端口（如果未提供，将自动分配可用端口）
        """
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the *_VISIBLE_DEVICES
        # environment variable for each actor, unless
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set, so
        # set local rank to 0 when the flag is not applicable.
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port


class BasePPORole(DistributedTorchRayActor):
    def _setup_distributed(self, strategy: DeepspeedStrategy):
        # configure strategy
        self.strategy = strategy
        strategy.setup_distributed()

    def init_model_from_pretrained(self, *args, **kwargs):
        raise NotImplementedError()


@ray.remote(num_gpus=1)# 这个类会作为 Ray 远程 Actor 运行，每个实例占用 1 张 GPU, Ray 会在 GPU 资源充足的计算节点上调度这个 Actor，确保它独占 1 张 GPU。
#如果没有足够的 GPU 资源，Ray 会等待可用资源，而不会直接创建 Actor。
class ReferenceModelRayActor(BasePPORole):# 继承自 BasePPORole，表示是 PPO 训练的一部分
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):

        print('分布初始化开始')
        self._setup_distributed(strategy)
        print('分布初始化完成')
         # 初始化模型，加载预训练参数
        model = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            packing_samples=strategy.args.packing_samples,
        )
        strategy.print(model) # 打印模型结构（用于 Debug）

        if strategy.args.ref_reward_offload: # 如果启用了 `ref_reward_offload`，则需要将模型的计算部分放入 CPU，避免显存占用过高
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)       # 使用 DeepSpeed 策略包装模型，并启用 RLHF 相关功能
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        进行 Reference Model 的前向计算（无梯度），返回 log_probs。
        :param sequences: 输入的 token 序列（LongTensor）
        :param num_actions: 动作数量（默认为 None）
        :param attention_mask: Attention Mask（用于避免 Padding 影响）
        :param return_output: 是否返回完整的模型输出
        :param packed_seq_lens: 如果使用样本 Packing，则需要提供 Packed 序列长度
        :return: 计算得到的 log_probs 张量
        """

        device = torch.cuda.current_device()
        with torch.no_grad():
            log_probs = self.model(
                sequences.to(device),
                num_actions,
                attention_mask.to(device),
                return_output=return_output,
                packed_seq_lens=packed_seq_lens,
            )
        return log_probs.to("cpu")

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()


@ray.remote(num_gpus=1)# 让 Ray 处理远程 Actor 调度，等于给这个actor分配了多少资源.类似于设置环境变量
class RewardModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        model = get_llm_for_sequence_regression(
            pretrain,
            "reward",
            normalize_reward=strategy.args.normalize_reward,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            value_head_prefix=strategy.args.value_head_prefix,
            packing_samples=strategy.args.packing_samples,
        )
        strategy.print(model)
        strategy.print("reward normalization status: {}".format(strategy.args.normalize_reward))
        strategy.print("mean: {}, std {}".format(model.mean, model.std))

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None, packed_seq_lens=None
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            reward = self.model(sequences.to(device), attention_mask.to(device), packed_seq_lens=packed_seq_lens)
        return reward.to("cpu")

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()


class PPORayActorGroup: #在多个 GPU 设备上创建 Ray Actor（用于训练不同的 PPO 组件，如 Actor、Critic、Reward Model）。异步初始化模型（从预训练权重加载）。管理模型的训练过程（如 Actor 模型如何和 Critic、Reward Model 交互）。存储和分配计算资源（如 GPU/CPU 资源分配、Placement Group）。
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        ray_actor_type (Type[BasePPORole]): PPO model type that this actor group serve on.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        num_nodes,  # 计算节点数
        num_gpus_per_node,  # 每个计算节点上的 GPU 数量
        ray_actor_type: Type[BasePPORole],  # PPO 组件的 Actor 类型
        pg: PlacementGroup = None,  # PlacementGroup（可选），用于 Ray 资源分配
        num_gpus_per_actor=1,  # 每个 Actor 分配的 GPU 数量，默认为 1
        resources: Dict[str, float] = None,  # 额外的资源，如 CPU 资源
        num_resources_per_node: int = None,  # 每个节点的额外资源数量
    ) -> None:
        self._num_nodes = num_nodes  # 记录计算节点数
        self._num_gpus_per_node = num_gpus_per_node  # 记录每个节点的 GPU 数量
        self.ray_actor_type = ray_actor_type  # 记录 Actor 的类型        
        # 记录额外的资源（如 CPU），在 Ray 资源调度中可能需要

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg, num_gpus_per_actor):
        """
        初始化 PPO 组件的 Actors
        Args:
            pg: PlacementGroup（可选），用于 GPU/CPU 资源分配
            num_gpus_per_actor: 每个 Actor 分配的 GPU 数量
        """
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type 如果每个节点有多个 GPU，并且没有指定 PlacementGroup，则创建一个新的 PlacementGroup

        # if self._num_gpus_per_node > 1 and pg is None:
        #     bundles = [
        #         {"GPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node} for _ in range(self._num_nodes)
        #     ]  # 为每个节点分配 GPU 和 CPU
        #     if self._resources:
        #         resources_name = list(self._resources.keys())[0]  # 取第一个资源名称
        #         for i in range(len(bundles)):
        #             bundles[i][resources_name] = self._num_resources_per_node  # 给每个 bundle 分配额外的资源
        #     # 创建 PlacementGroup，并等待资源准备就绪
        #     pg = placement_group(bundles, strategy="STRICT_SPREAD")
        #     ray.get(pg.ready())
        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [{"GPU": 1, "CPU": 1} for _ in range(self._num_nodes * self._num_gpus_per_node)]
            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node

            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())

        # 初始化 Master Actor（主 Actor）
        if pg:
            master_actor = self.ray_actor_type.options( #options() 是 Ray 内置的方法。Ray 提供 options() 方法，用于动态配置 Actor 资源。
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,  # 额外资源
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0 # 指定 PlacementGroup 和 bundle 索引
                ),
            ).remote(world_size, 0, None, None) # 远程创建 Actor 远程创建 Actor (.remote(...)) 是 Ray 的分布式计算机制，它允许在不同的 GPU/CPU 资源上并行执行任务。
            #如果我们直接创建 Python 对象，所有任务都会在单个 Python 进程中顺序执行，效率低下。而 remote(...) 使得多个 Actor 并行运行，提高吞吐量。Ray 允许在多个 GPU 设备上自动调度计算，避免了手动管理资源的麻烦。
        else:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(world_size, 0, None, None)
        self._actor_handlers = [master_actor] # 存储所有的 Actor 句柄

        # Create worker actors
        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                if pg:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=rank, # // self._num_gpus_per_node这里需要改.. 
                        ),
                    ).remote(world_size, rank, master_addr, master_port)
                else:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                    ).remote(world_size, rank, master_addr, master_port)
                self._actor_handlers.append(worker_actor)
        # print('actor handlers的数量', len(self._actor_handlers))
    def async_init_model_from_pretrained(
        self,
        *args,
        **kwargs,
    ):
        """
        从预训练模型初始化所有 Actor 的模型

        Returns:
            List: 远程对象引用列表（Ray Futures）
        """
        return [actor.init_model_from_pretrained.remote(*args, **kwargs) for actor in self._actor_handlers]

    def async_fit_actor_model(
        self,
        critic_model_group: "PPORayActorGroup",
        initial_model_group: "PPORayActorGroup",
        reward_model_groups: List["PPORayActorGroup"],
        remote_rm_urls: List[str] = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List = None,
        remote_sufficient_url: str = None,
    ):
        """Train actor model.

        Args:
            critic_model_group (PPORayActorGroup): critic model group.
            initial_model_group (PPORayActorGroup): reference model group.
            reward_model_groups (PPORayActorGroup): reward model groups.
            remote_rm_urls: remote RM APIs.
            reward_fn: reward calculate function, must be specified if using multiple reward models.
            vllm_engines: vllm engines for text generation, if not specified, generate text by actor model directly.

        Returns:
            List: list of remote object refs.
        """
        assert (
            (remote_rm_urls and len(remote_rm_urls) == 1)
            or (reward_model_groups and len(reward_model_groups) == 1)
            or reward_fn is not None
        ), "如果使用多个 Reward Model，必须提供 reward_fn"

        critic_actors = critic_model_group._actor_handlers if critic_model_group else None
        initial_actors = initial_model_group._actor_handlers

        refs = []
        # TODO(wuxibin): actor model choose critic/reward/initial model in a
        # round robin fashion, implement more efficient dispatching strategy.
        # 由于 Actor 数量可能大于 Critic 或 Initial Model，因此：i % len(critic_actors) 确保 critic_actors 被均匀分配到多个 Actor。i % len(initial_actors) 确保 initial_actors 也均匀分配。 reward actors 同理。
        for i, actor in enumerate(self._actor_handlers):
            critic_actor = critic_actors[i % len(critic_actors)] if critic_actors else None
            initial_actor = initial_actors[i % len(initial_actors)]

            reward_actors = []
            if not remote_rm_urls:
                for reward_model_group in reward_model_groups:
                    actors = reward_model_group._actor_handlers
                    reward_actors.append(actors[i % len(actors)]) # 将每个 Reward Model 的 Actor 添加到列表中 

            refs.append(
                actor.fit.remote( #actor.fit.remote(...) 触发 异步训练。
                    critic_model=critic_actor,
                    initial_model=initial_actor,
                    reward_model=reward_actors,
                    remote_rm_url=remote_rm_urls,
                    reward_fn=reward_fn,
                    vllm_engines=vllm_engines,
                    remote_sufficient_url=remote_sufficient_url,
                    # whether this actor should triger corresponding critic model training
                    critic_train_remote=(i < len(critic_actors)) if critic_actor else None, #让前几个 Actor 触发 Critic 训练（仅适用于 Critic Model 存在的情况）。
                )
            )
        ##在 Ray 中，异步任务（即 remote() 任务）是通过 Ray Actor 的 remote() 方法来定义和执行的。Ray 通过 remote 装饰器和 .remote() 方法，将 Python 函数或类方法转换为异步的远程任务。在 async_fit_actor_model 方法中，actor.fit.remote(...) 就是一个典型的 Ray 异步任务。
        #actor.fit.remote(...) 会触发 异步任务，它不会等待 fit 方法执行完，而是立即返回一个 ObjectRef（远程对象引用）。这些 ObjectRef 会存储在 refs 列表中，后续可以用 ray.get(refs) 获取任务的结果。
        return refs #返回所有远程任务的 ObjectRef，用于后续 ray.get() 获取结果。

    def async_save_model(self):
        """异步保存所有 Actor 模型（仅在 rank 0 上保存）。
        Returns:
            List: 远程对象引用列表（Ray Futures）
        """
        return [actor.save_model.remote() for actor in self._actor_handlers]

    def async_run_method(self, method_name, *args, **kwargs):
        """异步执行所有 Actor 的指定方法。

        Args:
            method_name: 需要执行的方法名
            *args, **kwargs: 传递给该方法的参数

        Returns:
            List: 远程对象引用列表（Ray Futures）
        """

        refs = []
        for actor in self._actor_handlers:
            method = getattr(actor, method_name)
            refs.append(method.remote(*args, **kwargs))
        return refs
