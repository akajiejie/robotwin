if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent) # set to the root directory of ManiFlow
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    
import os
import torch.multiprocessing as mp
import hydra
from omegaconf import OmegaConf
import pathlib
from hydra.core.hydra_config import HydraConfig
import torch
import socket
# from train_multitask_ddp import DDPWorkspace  # Import the workspace class

# Register resolvers
OmegaConf.register_new_resolver("eval", eval, replace=True)

def find_free_port():
    """Find a free port on localhost"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'maniflow', 'config')),
    config_name="config_name"
)
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    num_gpus = cfg.training.num_gpus
    import random
    if num_gpus > 1:
        master_port = find_free_port()
        print(f"Using master port: {master_port}")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(master_port)
        os.environ['WORLD_SIZE'] = str(num_gpus)
        
        mp.spawn(
            run_training,
            args=(cfg, num_gpus, output_dir),
            nprocs=num_gpus,
            join=True
        )
    else:
        # Single GPU training
        run_training(0, cfg, 1, output_dir)

def run_training(local_rank, cfg, world_size, output_dir):
    if world_size > 1:
        os.environ['RANK'] = str(local_rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=local_rank
            )
            cfg.training.distributed = True  # Enable distributed mode
    else:
        cfg.training.distributed = False  # Disable distributed mode for single GPU

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    from robotwin_ddp import DDPWorkspace
    workspace = DDPWorkspace(cfg, local_rank, output_dir)
    workspace.run()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()