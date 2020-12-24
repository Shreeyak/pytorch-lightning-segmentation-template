from omegaconf import DictConfig
from pydantic.dataclasses import dataclass

from seg_lapa.config_parse.dataset_conf import DatasetConf
from seg_lapa.config_parse.optimizer_conf import OptimConf
from seg_lapa.config_parse.trainer_conf import TrainerConf
from seg_lapa.config_parse.model_conf import ModelConf
from seg_lapa.config_parse.scheduler_conf import SchedulerConf

from seg_lapa.config_parse import dataset_conf
from seg_lapa.config_parse import optimizer_conf
from seg_lapa.config_parse import trainer_conf
from seg_lapa.config_parse import model_conf
from seg_lapa.config_parse import scheduler_conf


@dataclass(frozen=True)
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
        dataset=dataset_conf.validate_config_group(cfg.dataset),
        model=model_conf.validate_config_group(cfg.model),
        optimizer=optimizer_conf.validate_config_group(cfg.optimizer),
        trainer=trainer_conf.validate_config_group(cfg.trainer),
        scheduler=scheduler_conf.validate_config_group(cfg.scheduler),
    )

    return config
