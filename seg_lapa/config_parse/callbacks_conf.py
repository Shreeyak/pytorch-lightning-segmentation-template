from abc import ABC, abstractmethod
from typing import Dict, Optional


from omegaconf import DictConfig
from pydantic.dataclasses import dataclass

from seg_lapa.config_parse.conf_utils import asdict_filtered, validate_config_group_generic
from pytorch_lightning.callbacks import ModelCheckpoint
from seg_lapa.callbacks import EarlyStopping, LogMedia


@dataclass(frozen=True)
class CallbacksConf(ABC):
    name: str

    @abstractmethod
    def get_callbacks_dict(self):
        return {}

    def get_callbacks_list(self, *args):
        callback_dict = self.get_callbacks_dict(*args)
        callback_list = list(callback_dict.values())
        return callback_list


@dataclass(frozen=True)
class DisabledCallbacksConf(CallbacksConf):
    def get_callbacks_dict(self) -> Dict:
        return {}


@dataclass(frozen=True)
class EarlyStopConf:
    """Dataclass just to initialize and return the Early Stopping Callback"""

    min_delta: float
    patience: int

    def get_callback(self):
        args_dict = asdict_filtered(self)
        return EarlyStopping(**args_dict)


@dataclass(frozen=True)
class CheckpointConf:
    """Dataclass just to initialize and return the Checkpoint Callback"""

    monitor: Optional[str]
    mode: str
    save_last: Optional[bool]
    period: int

    def get_callback(self, logs_dir):
        args_dict = asdict_filtered(self)
        checkpoint_callback = ModelCheckpoint(dirpath=logs_dir, **args_dict)
        return checkpoint_callback


@dataclass(frozen=True)
class LogMediaConf:
    """Dataclass just to initialize and return the Checkpoint Callback"""

    max_images_to_log: int
    logging_epoch_interval: Optional[int] = 0
    logging_batch_interval: Optional[int] = 0

    def get_callback(self):
        args_dict = asdict_filtered(self)
        checkpoint_callback = LogMedia(**args_dict)
        return checkpoint_callback


@dataclass(frozen=True)
class StandardCallbacksConf(CallbacksConf):
    """Get a dictionary of all the callbacks."""

    early_stopping: Optional[Dict] = None
    checkpoints: Optional[Dict] = None
    log_media: Optional[Dict] = None

    def get_callbacks_dict(self, logs_dir) -> Dict:
        """Get all available callbacks and return as a dict"""
        if self.early_stopping is not None:
            early_stop = EarlyStopConf(**self.early_stopping).get_callback()
        if self.checkpoints is not None:
            checkpoint = CheckpointConf(**self.checkpoints).get_callback(logs_dir)
        if self.log_media is not None:
            log_media = LogMediaConf(**self.log_media).get_callback()

        return {"early_stopping": early_stop, "checkpoint": checkpoint, "log_media": log_media}


valid_names = {
    "disabled": DisabledCallbacksConf,
    "standard": StandardCallbacksConf,
}


def validate_config_group(cfg_subgroup: DictConfig) -> CallbacksConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, dataclass_dict=valid_names, config_category="callback"
    )
    return validated_dataclass
