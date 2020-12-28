from typing import Optional, Union

from omegaconf import DictConfig
from pydantic.dataclasses import dataclass
import pytorch_lightning as pl

from seg_lapa.config_parse.dataset_conf import DatasetConf
from seg_lapa.config_parse.optimizer_conf import OptimConf
from seg_lapa.config_parse.trainer_conf import TrainerConf
from seg_lapa.config_parse.model_conf import ModelConf
from seg_lapa.config_parse.scheduler_conf import SchedulerConf
from seg_lapa.config_parse.logger_conf import LoggerConf
from seg_lapa.config_parse import dataset_conf, optimizer_conf, trainer_conf, model_conf, scheduler_conf, logger_conf


@dataclass(frozen=True)
class RandomSeedConf:
    random_seed: Union[bool, int]

    def fix_seeds(self):
        """Fix seeds for reproducibility.
        Ref:
            https://pytorch.org/docs/stable/notes/randomness.html

        If self.random_seed is false, will not set seeds.
        If True, a random value will be decided for seed.
        If int, will use that to set seed.
        """
        if self.random_seed is True:
            pl.seed_everything(None)
        elif isinstance(self.random_seed, int):
            pl.seed_everything(self.random_seed)


@dataclass(frozen=True)
class TrainConf:
    random_seed: RandomSeedConf
    dataset: DatasetConf
    optimizer: OptimConf
    model: ModelConf
    trainer: TrainerConf
    scheduler: SchedulerConf
    logger: LoggerConf


def parse_config(cfg: DictConfig) -> TrainConf:
    """Parses the config file read from hydra to populate the TrainConfig dataclass"""
    config = TrainConf(
        random_seed=RandomSeedConf(cfg.random_seed),
        dataset=dataset_conf.validate_config_group(cfg.dataset),
        model=model_conf.validate_config_group(cfg.model),
        optimizer=optimizer_conf.validate_config_group(cfg.optimizer),
        trainer=trainer_conf.validate_config_group(cfg.trainer),
        scheduler=scheduler_conf.validate_config_group(cfg.scheduler),
        logger=logger_conf.validate_config_group(cfg.logger),
    )

    return config
