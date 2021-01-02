import os
import datetime
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


def create_log_dir(run_id: str, logs_root_dir: str) -> str:
    """Each run's log dir will have same name as wandb runid"""
    logs_root_dir = Path(logs_root_dir)
    if not logs_root_dir.is_absolute():
        # Any relative path is considered to be relative to the project root dir
        logs_root_dir = get_project_root() / logs_root_dir

    logs_root_dir = logs_root_dir / LOGS_DIR
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    exp_dir = logs_root_dir / f"{timestamp}-{run_id}"

    return exp_dir


def fix_seeds(random_seed: Optional[int]) -> None:
    """Fix seeds for reproducibility.
    Ref:
        https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        random_seed: If None, seeds not set. If int, uses value to seed.
    """
    if random_seed is not None:
        pl.seed_everything(random_seed)
