from typing import Any

from pydantic.dataclasses import dataclass

from seg_lapa.config_parse.dataset_conf import DatasetConf, get_dataconf
from seg_lapa.config_parse.optimizer_conf import OptimConf, get_optimconf


@dataclass
class TrainConf:
    dataset: DatasetConf
    optimizer: OptimConf
    # scheduler: Any
    # model: Any

    # trainer: Any
    # loggers: Any
    # num_steps: int
