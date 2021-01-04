from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import wandb
from omegaconf import DictConfig, OmegaConf
from pydantic.dataclasses import dataclass
from pytorch_lightning import loggers as pl_loggers

from seg_lapa.config_parse.conf_utils import asdict_filtered, validate_config_group_generic


@dataclass
class LoggerConf(ABC):
    name: str

    @abstractmethod
    def get_logger(self, *args):
        pass

    @abstractmethod
    def get_run_id(self, *args):
        """Loggers such as WandB generate a unique run id that can be used to resume runs"""
        pass


@dataclass
class DisabledLoggerConf(LoggerConf):
    @staticmethod
    def get_logger(*args):
        return False

    @staticmethod
    def get_run_id():
        return None


@dataclass
class WandbConf(LoggerConf):
    """Weights and Biases. Ref: wandb.com"""

    entity: str
    project: str
    run_name: Optional[str]
    run_id: Optional[str] = None  # Pass run_id to resume logging to that run.

    def get_logger(self, cfg: DictConfig, save_dir: Path) -> pl_loggers.WandbLogger:
        """Returns the Weights and Biases (wandb) logger object (really an wandb Run object)
        The run object corresponds to a single execution of the script and is returned from `wandb.init()`.

        Args:
            run_id: Unique run id. If run id exists, will continue logging to that run.
            cfg: The entire config got from hydra, for purposes of logging the config of each run in wandb.
            save_dir: Root dir to save wandb log files

        Returns:
            wandb.wandb_sdk.wandb_run.Run: wandb run object. Can be used for logging.
        """
        # Some argument names to wandb are different from the attribute names of the class.
        # Pop the offending attributes before passing to init func.
        args_dict = asdict_filtered(self)
        run_name = args_dict.pop("run_name")
        run_id = args_dict.pop("run_id")

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        wb_logger = pl_loggers.WandbLogger(
            name=run_name, id=run_id, save_dir=str(save_dir), config=cfg_dict, **args_dict
        )

        return wb_logger

    def get_run_id(self):
        """If a run_id has been provided by the user, resume logging to that run.
        Otherwise a random run-id will be generated
        """
        if self.run_id is None:
            self.run_id = wandb.util.generate_id()

        return self.run_id


valid_names = {
    "wandb": WandbConf,
    "disabled": DisabledLoggerConf,
}


def validate_config_group(cfg_subgroup: DictConfig) -> LoggerConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, dataclass_dict=valid_names, config_category="logger"
    )
    return validated_dataclass
