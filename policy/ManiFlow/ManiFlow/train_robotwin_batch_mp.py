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
# from train_multitask_ddp import DDPWorkspace  # Import the workspace class

# Register resolvers
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'maniflow', 'config')),
    config_name="config_name"
)
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    num_gpus = cfg.training.num_gpus
    MASTER_PORTS = [29500, 29501, 29502, 29503, 29504, 29505, 29506, 29507, 29508, 29509]
    import random
    if num_gpus > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(random.choice(MASTER_PORTS))
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
    from robotwin_batch_ddp import DDPWorkspace
    workspace = DDPWorkspace(cfg, local_rank, output_dir)
    workspace.run()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()