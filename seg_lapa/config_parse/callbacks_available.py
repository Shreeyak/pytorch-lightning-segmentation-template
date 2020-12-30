"""Dataclasses just to initialize and return Callback objects"""
from typing import Optional

from pydantic.dataclasses import dataclass

from seg_lapa.config_parse.conf_utils import asdict_filtered
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from seg_lapa.callbacks import EarlyStopping, LogMedia


@dataclass(frozen=True)
class EarlyStopConf:
    min_delta: float
    patience: int

    def get_callback(self) -> Callback:
        return EarlyStopping(**asdict_filtered(self))


@dataclass(frozen=True)
class CheckpointConf:
    monitor: Optional[str]
    mode: str
    save_last: Optional[bool]
    period: int

    def get_callback(self, logs_dir) -> Callback:
        return ModelCheckpoint(dirpath=logs_dir, **asdict_filtered(self))


@dataclass(frozen=True)
class LogMediaConf:
    max_images_to_log: int
    logging_epoch_interval: Optional[int] = 0
    logging_batch_interval: Optional[int] = 0

    def get_callback(self) -> Callback:
        return LogMedia(**asdict_filtered(self))
