from dataclasses import dataclass

import torch
from pytorch_lightning import metrics


@dataclass
class IouMetric:
    iou_per_class: torch.Tensor
    miou: torch.Tensor  # Mean IoU across all classes
    accuracy: torch.Tensor
    precision: torch.Tensor
    recall: torch.Tensor
    specificity: torch.Tensor


class Iou(metrics.Metric):
    def __init__(self, num_classes: int = 11, normalize: bool = False):
        """Calculates the metrics iou, true positives and false positives/negatives for multi-class classification
        problems such as semantic segmentation.
        Because this is an expensive operation, we do not compute or sync the values per step.

        Forward accepts:

        - ``prediction`` (float or long tensor): ``(N, H, W)``
        - ``label`` (long tensor): ``(N, H, W)``

        Note:
            This metric produces a dataclass as output, so it can not be directly logged.
        """
        super().__init__(compute_on_step=False, dist_sync_on_step=False)

        self.num_classes = num_classes
        # Metric normally calculated on batch. If true, final metrics (tp, fn, etc) will reflect average values per image
        self.normalize = normalize

        self.acc_confusion_matrix = None  # The accumulated confusion matrix
        self.count_samples = None  # Number of samples seen
        # Use `add_state()` for attr to track their state and synchronize state across processes
        self.add_state(
            "acc_confusion_matrix", default=torch.zeros((self.num_classes, self.num_classes)), dist_reduce_fx="sum"
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

        num_images = int(label.shape[0])

        label = label.view(-1).long()
        prediction = prediction.view(-1).long()

        # Calculate confusion matrix
        conf_mat = torch.bincount(self.num_classes * label + prediction, minlength=self.num_classes ** 2)
        conf_mat = conf_mat.reshape((self.num_classes, self.num_classes))

        # Accumulate values
        self.acc_confusion_matrix += conf_mat
        self.count_samples += num_images

    def compute(self):
        """Compute the final IoU and other metrics across all samples seen"""
        # Normalize the accumulated confusion matrix, if needed
        conf_mat = self.acc_confusion_matrix
        if self.normalize:
            conf_mat = conf_mat / self.count_samples  # Get average per image

        # Calculate True Positive (TP), False Positive (FP), False Negative (FN) and True Negative (TN)
        tp = conf_mat.diagonal()
        fn = conf_mat.sum(dim=0) - tp
        fp = conf_mat.sum(dim=1) - tp
        total_px = conf_mat.sum()
        tn = total_px - (tp + fn + fp)

        # Calculate Intersection over Union (IoU)
        eps = 1e-6
        iou_per_class = (tp + eps) / (fn + fp + tp + eps)  # Use epsilon to avoid zero division errors
        iou_per_class[torch.isnan(iou_per_class)] = 0
        mean_iou = iou_per_class.mean()

        # Accuracy (what proportion of predictions — both Positive and Negative — were correctly classified?)
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        accuracy[torch.isnan(accuracy)] = 0

        # Precision (what proportion of predicted Positives is truly Positive?)
        precision = tp / (tp + fp)
        precision[torch.isnan(precision)] = 0

        # Recall or True Positive Rate (what proportion of actual Positives is correctly classified?)
        recall = tp / (tp + fn)
        recall[torch.isnan(recall)] = 0

        # Specificity or true negative rate
        specificity = tn / (tn + fp)
        specificity[torch.isnan(specificity)] = 0

        data_r = IouMetric(
            iou_per_class=iou_per_class,
            miou=mean_iou,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            specificity=specificity,
        )

        return data_r


# Tests
def test_iou():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create Fake label and prediction
    label = torch.zeros((1, 4, 4), dtype=torch.float32, device=device)
    pred = torch.zeros((1, 4, 4), dtype=torch.float32, device=device)
    label[:, :3, :3] = 1
    pred[:, -3:, -3:] = 1
    expected_iou = torch.tensor([2.0 / 12, 4.0 / 14], device=device)

    print("Testing IoU metrics", end="")
    iou_train = Iou(num_classes=2)
    iou_train.to(device)
    iou_train(pred, label)
    metrics_r = iou_train.compute()
    iou_per_class = metrics_r.iou_per_class
    assert (iou_per_class - expected_iou).sum() < 1e-6
    print("  passed")


if __name__ == "__main__":
    # Run tests
    print("Running tests on metrics module...\n")
    test_iou()
