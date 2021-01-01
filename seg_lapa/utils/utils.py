import os
from pathlib import Path
from typing import Optional

import wandb
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig

from seg_lapa.utils.path_check import get_project_root

LOGS_DIR = "logs"


def is_rank_zero():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", 0))
    if local_rank == 0 and node_rank == 0:
        return True

    return False


def generate_run_id(cfg: DictConfig):
    # Set the run ID: Read from config if resuming training, else generate unique id
    # TODO: read from cfg if resuming training - get from config dataclass! add method to resume training section.
    run_id = wandb.util.generate_id()
    return run_id


def create_log_dir(cfg: DictConfig, run_id: str) -> Optional[Path]:
    """Each run's log dir will have same name as wandb runid"""
    log_root_dir = get_project_root() / LOGS_DIR

    if is_rank_zero():
        log_dir = log_root_dir / run_id
        log_dir.mkdir(parents=True, exist_ok=True)

        # Save the input config file to logs dir
        OmegaConf.save(cfg, log_dir / "train.yaml")
    else:
        return log_root_dir / "None"

    return log_dir


def fix_seeds(random_seed: Optional[int]) -> None:
    """Fix seeds for reproducibility.
    Ref:
        https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        random_seed: If None, seeds not set. If int, uses value to seed.
    """
    if random_seed is not None:
        pl.seed_everything(random_seed)
