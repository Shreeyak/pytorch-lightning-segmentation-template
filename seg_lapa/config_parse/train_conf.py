from omegaconf import DictConfig
from pydantic.dataclasses import dataclass

from seg_lapa.config_parse.dataset_conf import DatasetConf, validate_dataconf
from seg_lapa.config_parse.optimizer_conf import OptimConf, validate_optimconf
from seg_lapa.config_parse.trainer_conf import TrainerConf, validate_trainerconf
from seg_lapa.config_parse.model_conf import ModelConf, validate_modelconf
from seg_lapa.config_parse.scheduler_conf import SchedulerConf, validate_schedulerconf



@dataclass
class TrainConf:
    dataset: DatasetConf
    optimizer: OptimConf
    model: ModelConf
    trainer: TrainerConf
    scheduler: SchedulerConf
    # loggers: Any


def parse_config(cfg: DictConfig) -> TrainConf:
    """Parses the config file read from hydra to populate the TrainConfig dataclass"""
    config = TrainConf(
        dataset=validate_dataconf(cfg.dataset),
        model=validate_modelconf(cfg.model),
        optimizer=validate_optimconf(cfg.optimizer),
        trainer=validate_trainerconf(cfg.trainer),
        scheduler=validate_schedulerconf(cfg.scheduler),
    )

    return config
