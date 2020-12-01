from omegaconf import DictConfig
from pydantic.dataclasses import dataclass

from seg_lapa.config_parse.dataset_conf import DatasetConf, validate_dataconf
from seg_lapa.config_parse.optimizer_conf import OptimConf, validate_optimconf


@dataclass
class TrainConf:
    dataset: DatasetConf
    optimizer: OptimConf
    # scheduler: Any
    # model: Any

    # trainer: Any
    # loggers: Any
    # num_steps: int


def parse_config(cfg: DictConfig) -> TrainConf:
    """Parses the config file read from hydra to populate the TrainConfig dataclass"""
    config = TrainConf(
        dataset=validate_dataconf(cfg.dataset),
        optimizer=validate_optimconf(cfg.optimizer)
    )

    return config
