from typing import Optional

from omegaconf import DictConfig
from pydantic.dataclasses import dataclass

from seg_lapa.config_parse.dataset_conf import DatasetConf
from seg_lapa.config_parse.optimizer_conf import OptimConf
from seg_lapa.config_parse.trainer_conf import TrainerConf
from seg_lapa.config_parse.model_conf import ModelConf
from seg_lapa.config_parse.scheduler_conf import SchedulerConf
from seg_lapa.config_parse.logger_conf import LoggerConf
from seg_lapa.config_parse import dataset_conf, optimizer_conf, trainer_conf, model_conf, scheduler_conf, logger_conf


@dataclass(frozen=True)
class TrainConf:
    random_seed: Optional[int]
    dataset: DatasetConf
    optimizer: OptimConf
    model: ModelConf
    trainer: TrainerConf
    scheduler: SchedulerConf
    logger: LoggerConf


def parse_config(cfg: DictConfig) -> TrainConf:
    """Parses the config file read from hydra to populate the TrainConfig dataclass"""
    config = TrainConf(
        random_seed=cfg.random_seed,
        dataset=dataset_conf.validate_config_group(cfg.dataset),
        model=model_conf.validate_config_group(cfg.model),
        optimizer=optimizer_conf.validate_config_group(cfg.optimizer),
        trainer=trainer_conf.validate_config_group(cfg.trainer),
        scheduler=scheduler_conf.validate_config_group(cfg.scheduler),
        logger=logger_conf.validate_config_group(cfg.logger),
    )

    return config
