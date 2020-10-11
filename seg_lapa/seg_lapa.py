import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .networks.deeplab import deeplab


class Deeplabv3plus(pl.LightningModule):

    def __init__(self):
        super().__init__()

