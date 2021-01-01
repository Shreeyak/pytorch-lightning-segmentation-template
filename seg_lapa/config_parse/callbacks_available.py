"""Dataclasses just to initialize and return Callback objects"""
from typing import Optional

from pydantic.dataclasses import dataclass
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping

from seg_lapa.config_parse.conf_utils import asdict_filtered
from seg_lapa.callbacks.log_media import LogMedia


@dataclass(frozen=True)
class EarlyStopConf:
    monitor: str
    min_delta: float
    patience: int
    mode: str

    def get_callback(self) -> Callback:
        return EarlyStopping(**asdict_filtered(self))


@dataclass(frozen=True)
class CheckpointConf:
    monitor: Optional[str]
    mode: str
    save_last: Optional[bool]
    period: int
    save_top_k: Optional[int]

    def get_callback(self, logs_dir) -> Callback:
        return ModelCheckpoint(dirpath=logs_dir, **asdict_filtered(self))


@dataclass(frozen=True)
class LogMediaConf:
    max_samples: int
    period_epoch: Optional[int] = 0
    period_step: Optional[int] = 0
    save_to_disk: bool = False

    def get_callback(self, logs_dir) -> Callback:
        return LogMedia(logs_dir=logs_dir, **asdict_filtered(self))
