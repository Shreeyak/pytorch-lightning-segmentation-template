from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from seg_lapa.config_parse.dataset_conf import DatasetConf
from seg_lapa.config_parse.optimizer_conf import OptimConf

@dataclass
class TrainConfig:
    num_steps: int = MISSING
    dataset: Any = MISSING
    optimizer: Any = MISSING
    # scheduler: Any = MISSING
    # model: Any = MISSING

    # trainer: Any = MISSING
    # loggers: Any = MISSING


cs = ConfigStore.instance()
cs.store(name="train", node=TrainConfig)

