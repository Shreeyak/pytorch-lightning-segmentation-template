from dataclasses import dataclass

import numpy as np
import torch
from pytorch_lightning import metrics


@dataclass
class IouMetric:
    iou_per_class: torch.Tensor
    miou: torch.Tensor  # Mean IoU across all classes
    tp: torch.Tensor  # True Positive
    fp: torch.Tensor  # False Positive
    fn: torch.Tensor  # False Negative


# class Iou:
#     def __init__(self, num_classes: int):
#         """Calculate the Intersection over Union for multi-class segmentation
#
#         This is a fast implementation of IoU that can run over the GPU
#
#         The Iou constructs a confusion matrix for every call (batch) and accumulates the results.
#         At the end of the epoch, metrics such as IoU can be extracted.
#         """
#         self.num_classes = num_classes
#         self.acc_confusion_matrix = None  # Avoid having to assign device by initializing as None
#         self.count_samples = 0  # This isn't used as of now. Can be used to normalize the accumulated conf matrix
#
#     def reset(self) -> None:
#         self.acc_confusion_matrix = None
#         self.count_samples = 0
#
#     def accumulate(self, prediction: torch.Tensor, label: torch.Tensor) -> None:
#         """Calculates and accumulates the confusion matrix for a batch
#
#         Args:
#             prediction: The predictions of the network (after argmax of the probabilities).
#                         Shape: [N, H, W]
#             label: The label (ground truth). Each pixel contains an int corresponding to the class it belongs to.
#                         Shape: [N, H, W]
#         """
#         label = label.detach().view(-1).long()  # .cpu()
#         prediction = prediction.detach().view(-1).long()  # .cpu()
#
#         # Note: DO NOT pass in argument "minlength". It cause huge slowdowns on GPU (~100x, tested with Pytorch 1.6.0)
#         conf_mat = torch.bincount(self.num_classes * label + prediction)
#
#         # Length of bincount depends on max value of inputs. Pad confusion matrix with zeros to get correct size.
#         size_conf_mat = self.num_classes * self.num_classes
#         if len(conf_mat) < size_conf_mat:
#             req_padding = torch.zeros(size_conf_mat - len(conf_mat), dtype=torch.long, device=prediction.device)
#             conf_mat = torch.cat((conf_mat, req_padding))
#
#         # Accumulate result
#         conf_mat = conf_mat.reshape((self.num_classes, self.num_classes))
#         if self.acc_confusion_matrix is None:
#             self.acc_confusion_matrix = conf_mat
#         else:
#             self.acc_confusion_matrix += conf_mat
#         self.count_samples += 1
#
#     def get_accumulated_confusion_matrix(self) -> np.ndarray:
#         """Extract the accumulated confusion matrix"""
#         return self.acc_confusion_matrix.cpu().numpy()
#
#     def get_iou(self) -> IouMetric:
#         """Calculate the IoU based on accumulated confusion matrix"""
#         if self.acc_confusion_matrix is None:
#             return IouMetric(
#                 iou_per_class=np.array([0] * self.num_classes),
#                 tp=torch.Tensor(0),
#                 fn=torch.Tensor(0),
#                 fp=torch.Tensor(0),
#                 total_px=torch.Tensor(0),
#             )
#
#         conf_mat = self.acc_confusion_matrix
#
#         tp = torch.diagonal(conf_mat)
#         fn = torch.sum(conf_mat, dim=0) - tp
#         fp = torch.sum(conf_mat, dim=1) - tp
#         total_px = torch.sum(conf_mat)
#
#         eps = 1e-6
#         iou_per_class = (tp + eps) / (fn + fp + tp + eps)  # Use epsilon to avoid zero division errors
#
#         # Bring to CPU
#         iou_per_class = iou_per_class.cpu().numpy()
#         mean_iou = iou_per_class.mean()
#
#         data_r = IouMetric(iou_per_class=iou_per_class, miou=mean_iou, tp=tp, fn=fn, fp=fp, total_px=total_px)
#
#         return data_r


class IouSync(metrics.Metric):
    def __init__(self, num_classes=11, device=torch.device("cpu")):
        """Calculates the metrics iou, true positives and false positives/negatives for multi-class classification
        problems such as semantic segmentation.
        Because this is an expensive operation, we do not compute or sync the values per step
        """
        super().__init__(compute_on_step=False, dist_sync_on_step=False)

        self.num_classes = num_classes
        self.device = device
        # The number of pixels in a set of images can be very large. Divide conf matrix by a factor to reduce max value.
        self.normalize_factor = 10000
        # self.register_buffer("normalize_factor", torch.Tensor(10000))

        self.acc_confusion_matrix = None  # The accumulated confusion matrix
        self.count_samples = None  # Number of samples seen
        self.add_state(
            "acc_confusion_matrix",
            default=torch.zeros((self.num_classes, self.num_classes), dtype=torch.float32, device=self.device),
            dist_reduce_fx="sum",
        )
        self.add_state("count_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prediction: torch.Tensor, label: torch.Tensor):
        """Calculate the confusion matrix and accumulate it

        Args:
            prediction: Predictions of network (after argmax). Shape: [N, H, W]
            label: Ground truth. Each pixel has int value denoting class. Shape: [N, H, W]
        """
        assert prediction.shape == label.shape
        assert len(label.shape) == 3

        label = label.view(-1)
        prediction = prediction.view(-1)

        # Note: DO NOT pass in argument "minlength". It cause huge slowdowns on GPU.
        # (~100x, tested with Pytorch 1.6.0, with both input in .long() datatype)
        conf_mat = torch.bincount(self.num_classes * label + prediction)

        # Length of bincount depends on max value of inputs. Pad confusion matrix with zeros to get correct size.
        size_conf_mat = self.num_classes * self.num_classes
        if len(conf_mat) < size_conf_mat:
            req_padding = torch.zeros(size_conf_mat - len(conf_mat), device=prediction.device)
            conf_mat = torch.cat((conf_mat, req_padding))

        conf_mat = conf_mat.reshape((self.num_classes, self.num_classes))
        self.acc_confusion_matrix += conf_mat / self.normalize_factor

        # num_images = label.shape[0]
        num_images = 1
        self.count_samples += num_images

    def compute(self):
        """Compute the final IoU and other metrics across all samples seen"""

        """
        Total num of pixels across full dataset can easily overflow float32.
        So the acc. conf mat needs to be in .long format. But the bincount, etc do not.
        We can avoid this by normalizing the values. I.e., divide by 1k or something, so max value is limited

        We should also normalize the conf matrix at end - i.e. Divide by num of samples.
        This final norm conf matrix can be float32
        """
        # Average and de-normalize the accumulated confusion matrix
        conf_mat = self.acc_confusion_matrix / self.count_samples
        conf_mat *= self.normalize_factor

        tp = conf_mat.diagonal()
        fn = conf_mat.sum(dim=0) - tp
        fp = conf_mat.sum(dim=1) - tp
        # total_px = conf_mat.sum()

        eps = torch.tensor(1e-6, device=conf_mat.device)
        iou_per_class = (tp + eps) / (fn + fp + tp + eps)  # Use epsilon to avoid zero division errors
        mean_iou = iou_per_class.mean()

        data_r = IouMetric(iou_per_class=iou_per_class, miou=mean_iou, tp=tp, fn=fn, fp=fp)

        return data_r


# Tests
def test_iou():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create Fake label and prediction
    label = torch.zeros((1, 4, 4), dtype=torch.int, device=device)
    pred = torch.zeros((1, 4, 4), dtype=torch.int, device=device)
    label[:, :3, :3] = 1
    pred[:, -3:, -3:] = 1
    expected_iou = torch.tensor([2.0 / 12, 4.0 / 14], device=device)

    iou_train = IouSync(num_classes=2, device=device)
    iou_train(pred, label)
    metrics_r = iou_train.compute()
    iou_per_class = metrics_r.iou_per_class

    assert (iou_per_class - expected_iou).sum() < 1e-6
    print("Testing IOU: passed")


if __name__ == "__main__":
    # Run tests
    print("Running tests on metrics module..")
    test_iou()
