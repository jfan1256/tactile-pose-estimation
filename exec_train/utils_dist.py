import os
import torch
import torch.distributed as dist

# Print only on master process
def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print

# Is distributed available and initialized
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

# Get world size
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

# Get rank
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

# Check if main gpu
def is_main_process():
    return get_rank() == 0

# Initialize distributed mode (multi-gpu)
def init_distributed_mode(configs):
    # Set package
    if configs['package'] == 'gloo':
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        configs.rank, configs.world_size, configs.gpu = int(os.environ["RANK"]), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
        print(f"init_distributed_mode Case 1:\nargs.rank: {configs.rank}\nargs.world_size: {configs.world_size}\nargs.gpu: {configs.gpu}")
        configs.distributed = True
    elif 'SLURM_PROCID' in os.environ:
        configs.rank = int(os.environ['SLURM_PROCID'])
        configs.gpu = configs.rank % torch.cuda.device_count()
        print(f"init_distributed_mode Case 2:\nargs.rank: {configs.rank}\nargs.gpu: {configs.gpu}")
        configs.distributed = True
    else:
        print("init_distributed_mode Case 3:\nNot using distributed mode")
        configs.distributed = False
        return

    # Initialize process group
    torch.cuda.set_device(configs.gpu)
    configs.dist_backend = configs.package
    print(f"Distributed initialization complete | Rank: {configs.rank}, World Size: {configs.world_size}, Init Method: {configs.dist_url}", flush=True)
    torch.distributed.init_process_group(backend=configs.dist_backend, init_method=configs.dist_url, world_size=configs.world_size, rank=configs.rank)
    torch.distributed.barrier()
    setup_for_distributed(configs.rank == 0)
