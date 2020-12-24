import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .aspp import build_aspp
from .backbone import build_backbone
from .decoder import build_decoder
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class DeepLab(nn.Module):
    def __init__(
        self, backbone="drn", output_stride=8, num_classes=11, sync_bn=False, freeze_bn=False, enable_amp=False
    ):
        super(DeepLab, self).__init__()

        if backbone == "drn" and output_stride != 8:
            raise ValueError(f'The "drn" backbone only supports output stride = 8. Input: {output_stride}')

        # Ref for sync_bn: https://hangzhang.org/PyTorch-Encoding/tutorials/syncbn.html
        # Sync batchnorm is required when the effective batchsize per GPU is small (~4)
        if sync_bn is True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

        self.enable_amp = enable_amp

    def forward(self, inputs):
        with autocast(enabled=self.enable_amp):
            """Pytorch Automatic Mixed Precision (AMP) Training
            Ref: https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast

            - For use with DataParallel, we must add autocast within model definition.
            """
            x, low_level_feat = self.backbone(inputs)
            x = self.aspp(x)
            x = self.decoder(x, low_level_feat)
            x = F.interpolate(x, size=inputs.size()[2:], mode="bilinear", align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if (
                        isinstance(m[1], nn.Conv2d)
                        or isinstance(m[1], SynchronizedBatchNorm2d)
                        or isinstance(m[1], nn.BatchNorm2d)
                    ):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if (
                        isinstance(m[1], nn.Conv2d)
                        or isinstance(m[1], SynchronizedBatchNorm2d)
                        or isinstance(m[1], nn.BatchNorm2d)
                    ):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


if __name__ == "__main__":
    model = DeepLab(backbone="drn", output_stride=8)
    model.eval()
    inputs = torch.rand(1, 3, 512, 512)
    output = model(inputs)
    print(output.size())
