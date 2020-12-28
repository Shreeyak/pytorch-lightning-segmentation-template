from pytorch_lightning.callbacks import early_stopping


class EarlyStopping(early_stopping.EarlyStopping):
    """Direct sub-class of PL's early-stopping, changing the default parameters to suit our project

    Args:
        monitor: Monitor a key validation metric (IoU). Monitoring loss is not a good idea as it is an unreliable
                 indicator of model performance. Two models might have the same loss but different
                 performance (IoU), or the loss might start increasing, even though IoU does not decrease.
        min_delta: Project-dependent - choose a value for your metric below which you'd consider the improvement
                   negligible.
                   Example: For segmentation, I do not care for improvements less than 0.05% IoU in general.
                            But in kaggle competitions, even 0.01% would matter.
        patience: Takes experimentation to figure out appropriate patience for your project. Train the model
                  once without early stopping and see how long it takes to converge on a given dataset.
                  Choose the number of epochs between when you feel it's started to converge and after you're
                  sure the model has converged. Reduce the patience if you see the model continues to train for too long.
        verbose: Minimal extra info logs about earlystopping starting
        mode: Choose between "max" and "min". If the performance is considered better when metric is higher, choose
              "max", else "min".
        strict: Whether to crash the training if monitor is not found in the validation metrics. This should always be
                True. If early stopping is not desired, disable it.
    """

    def __init__(
        self,
        monitor="Val/mIoU",
        min_delta=0.0005,
        patience=10,
        verbose=True,
        mode="max",
        strict=True,
    ):
        super(EarlyStopping, self).__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            strict=strict,
        )
