import torch
from vllm.worker.worker import Worker

from openrlhf.utils.distributed_util import init_process_group
from openrlhf.utils.logging_utils import init_logger
from .utils import get_physical_gpu_id

logger = init_logger(__name__)


class WorkerWrap(Worker):
    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl", use_ray=False
    ):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), f"default torch process group must be initialized"
        assert group_name != "", f"group name must not be empty"
        print('打印vllm初始化的时候的rank信息',torch.distributed.get_rank(),rank_offset,world_size)
        rank = torch.distributed.get_rank() + rank_offset
        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(world_size=world_size, rank=rank, backend=backend, group_name=group_name)
            self._model_update_group = group_name
        else:
            self._model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
        self._model_update_with_ray = use_ray
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )
    def update_model_directly(self, state_dict):
        model = self.llm.llm_engine.model_executor.model
        for name, param in state_dict.items():
            if name in model.state_dict():
                model.state_dict()[name].copy_(param.to(model.state_dict()[name].device))

    def update_weight(self, name, dtype, shape, empty_cache=False):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        # if torch.distributed.get_rank() == 0:
        # print(f"update weight: {name}, dtype: {dtype}, shape: {shape},rank {torch.distributed.get_rank()}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        # print('权重所在device', weight.device, weight.dtype)
        if self._model_update_with_ray:
            import ray.util.collective as collective
            torch.cuda.synchronize()
            collective.barrier(group_name=self._model_update_group)
            print('使用ray广播参数', torch.distributed.get_rank())
            collective.broadcast(weight, 0, group_name=self._model_update_group)
            print('使用ray广播参数完成', torch.distributed.get_rank())
        else:
            torch.distributed.broadcast(weight, 0, group=self._model_update_group)

        self.model_runner.model.load_weights(weights=[(name, weight)])        
        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False):
        # if torch.distributed.get_rank() == 0:
        #     print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"

        handle = ipc_handles[get_physical_gpu_id()]
        device_id = self.device.index
        func, args = handle #恢复由reduce_tensor(...) 创建的 handle。一个可以重新恢复Tensor的函数, 恢复Tensor时需要传进去的参数
        list_args = list(args)
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
        weight = func(*list_args)
        # if torch.distributed.get_rank() == 0:
        #     print('更新参数', weight)
        #print(f"Before loading, {name} weight to load: mean={weight.float().mean().item()}, std={weight.float().std().item()}, shape={weight.shape}")

        self.model_runner.model.load_weights(weights=[(name, weight)])
        #for name1, param in self.model_runner.model.named_parameters():
           # print("model name", name1)
       # param = dict(self.model_runner.model.named_parameters())[name]
        #print(f"After loading, param {name}: mean={param.data.float().mean().item()}, std={param.data.float().std().item()}, shape={param.shape}")

        torch.cuda.synchronize()