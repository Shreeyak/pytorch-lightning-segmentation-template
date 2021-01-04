import datetime
import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl

from seg_lapa.config_parse.logger_conf import DisabledLoggerConf, WandbConf
from seg_lapa.config_parse.train_conf import TrainConf
from seg_lapa.utils import path_check

LOGS_DIR = "log-media"


def is_rank_zero():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", 0))
    if local_rank == 0 and node_rank == 0:
        return True

    return False


def generate_log_dir_path(config: TrainConf) -> Path:
    """Generate the path to the log dir for this run.
    The directory structure for logs depends on the logger used.

    wandb - Each run's log dir's name will contain the wandb runid for easy identification

    Args:
        config: The config dataclass.
    """
    logs_root_dir = path_check.get_path(config.logs_root_dir, force_relative_to_project=True)

    # Exp directory structure would depend on the logger used
    logs_root_dir = logs_root_dir / LOGS_DIR / f"{config.logger.name}-logger"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if isinstance(config.logger, DisabledLoggerConf):
        exp_dir = logs_root_dir / f"{timestamp}"
    elif isinstance(config.logger, WandbConf):
        run_id = config.logger.get_run_id()
        exp_dir = logs_root_dir / f"{timestamp}_{run_id}"
    else:
        raise NotImplementedError(f"Generating log dir not implemented for logger: {config.logger}")

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
