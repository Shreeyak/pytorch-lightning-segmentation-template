import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class IouMetric:
    iou_per_class: np.ndarray
    tp: torch.Tensor
    fp: torch.Tensor
    fn: torch.Tensor
    total_px: torch.Tensor


class Iou:
    def __init__(self, num_classes: int):
        """Calculate the Intersection over Union for multi-class segmentation

        This is a fast implementation of IoU that can run over the GPU

        The Iou constructs a confusion matrix for every call (batch) and accumulates the results.
        At the end of the epoch, metrics such as IoU can be extracted.
        """
        self.num_classes = num_classes
        self.conf_mat_flattened = None  # Avoid having to assign device by initializing as None
        self.count = 0  # This isn't used as of now. Can be used to normalize the accumulated conf matrix

    def reset(self) -> None:
        self.conf_mat_flattened = None
        self.count = 0

    def accumulate(self, prediction: torch.Tensor, label: torch.Tensor) -> None:
        """Calculates and accumulates the confusion matrix for a batch"""
        label = label.detach().view(-1).long()
        prediction = prediction.detach().view(-1).long()

        # Note: DO NOT pass in argument "minlength". It cause huge slowdowns on GPU (~100x, tested with Pytorch 1.6.0)
        conf_mat = torch.bincount(self.num_classes * label + prediction)

        # Length of bincount depends on max value of inputs. Pad confusion matrix with zeros to get correct size.
        size_conf_mat = self.num_classes * self.num_classes
        if len(conf_mat) < size_conf_mat:
            req_padding = torch.zeros(size_conf_mat - len(conf_mat), dtype=torch.long, device=prediction.device)
            conf_mat = torch.cat((conf_mat, req_padding))

        # Accumulate result
        if self.conf_mat_flattened is None:
            self.conf_mat_flattened = conf_mat
        else:
            self.conf_mat_flattened += conf_mat
        self.count += 1

    def get_accumulated_confusion_matrix(self) -> np.ndarray:
        """Extract the accumulated confusion matrix"""
        conf_mat = self.conf_mat_flattened.reshape((self.num_classes, self.num_classes))
        return conf_mat.cpu().numpy()

    def get_iou(self) -> IouMetric:
        """Calculate the IoU based on accumulated confusion matrix"""
        if self.conf_mat_flattened is None:
            return IouMetric(
                iou_per_class=np.array([0] * self.num_classes),
                tp=torch.Tensor(0),
                fn=torch.Tensor(0),
                fp=torch.Tensor(0),
                total_px=torch.Tensor(0),
            )

        conf_mat = self.conf_mat_flattened.reshape((self.num_classes, self.num_classes))

        tp = torch.diagonal(conf_mat)
        fn = torch.sum(conf_mat, dim=0) - tp
        fp = torch.sum(conf_mat, dim=1) - tp
        total_px = torch.sum(conf_mat)

        eps = 1e-6
        iou_per_class = (tp + eps) / (fn + fp + tp + eps)  # Use epsilon to avoid zero division errors

        # Bring to CPU
        iou_per_class = iou_per_class.cpu().numpy()

        # Compile into dict
        data_r = IouMetric(iou_per_class=iou_per_class, tp=tp, fn=fn, fp=fp, total_px=total_px)

        return data_r


# Tests
def test_iou():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create Fake label and prediction
    label = torch.zeros((1, 4, 4), device=device)
    pred = torch.zeros((1, 4, 4), device=device)
    label[:, :3, :3] = 1
    pred[:, -3:, -3:] = 1
    expected_iou = [2.0 / 12, 4.0 / 14]

    iou_meter = Iou(num_classes=2)
    iou_meter.accumulate(pred, label)
    metrics_r = iou_meter.get_iou()
    iou_per_class = metrics_r.iou_per_class

    assert (iou_per_class - expected_iou).sum() < 1e-6
    print("Testing IOU: passed")


if __name__ == "__main__":
    # Run tests
    test_iou()
