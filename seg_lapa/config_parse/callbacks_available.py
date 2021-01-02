"""Dataclasses just to initialize and return Callback objects"""
from typing import Optional

from omegaconf import DictConfig
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
    verbose: bool = False

    def get_callback(self) -> Callback:
        return EarlyStopping(**asdict_filtered(self))


@dataclass(frozen=True)
class CheckpointConf:
    monitor: Optional[str]
    mode: str
    save_last: Optional[bool]
    period: int
    save_top_k: Optional[int]
    verbose: bool = False

    def get_callback(self, logs_dir) -> Callback:
        return ModelCheckpoint(dirpath=logs_dir, **asdict_filtered(self))


@dataclass(frozen=True)
class LogMediaConf:
    max_samples: int
    period_epoch: int
    period_step: int
    save_to_disk: bool
    save_latest_only: bool
    verbose: bool = False

    def get_callback(self, exp_dir: str, cfg: DictConfig) -> Callback:
        return LogMedia(exp_dir=exp_dir, cfg=cfg, **asdict_filtered(self))
