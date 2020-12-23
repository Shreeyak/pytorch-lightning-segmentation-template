from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass

from seg_lapa.networks.deeplab.deeplab import DeepLab

@dataclass
class ModelConf(ABC):
    name: str

    @abstractmethod
    def get_model(self):
        pass


@dataclass
class Deeplabv3Conf(ModelConf):
    backbone: str
    output_stride: int
    num_classes: int
    sync_bn: bool  # Can use PL to sync batchnorm. This enables custom batchnorm code.
    enable_amp: bool = False  # Should always be false, since PL takes case of 16bit training

    def get_model(self) -> torch.nn.Module:
        # Clean the arguments
        args = vars(self)
        args.pop('name', None)
        args.pop('__initialised__', None)

        return DeepLab(**args)


valid_options = {
    "deeplabv3": Deeplabv3Conf,
}


def validate_modelconf(cfg_model: DictConfig) -> DeepLab:
    try:
        modelconf = valid_options[cfg_model.name](**cfg_model)
    except KeyError:
        raise ValueError(f"Invalid Config: '{cfg_model.name}' is not a valid optimizer. "
                         f"Valid Options: {list(valid_options.keys())}")

    return modelconf
