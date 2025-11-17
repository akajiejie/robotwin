if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from policy.ManiFlow.ManiFlow.train_robotwin_batch import TrainManiFlowWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'maniflow', 'config'))
)
def main(cfg):
    workspace = TrainManiFlowWorkspace(cfg)
    workspace.eval(cfg.eval_mode)

if __name__ == "__main__":
    main()
