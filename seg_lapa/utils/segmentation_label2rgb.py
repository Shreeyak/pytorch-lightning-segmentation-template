import enum

import numpy as np
from PIL import Image


class Palette(enum.Enum):
    LAPA = 0


class LabelToRGB:
    # fmt: off
    palette_lapa = [
        # Color Palette for the LaPa dataset, which has 11 classes
        0, 0, 0,        # 0	background
        0, 153, 255,    # 1	skin
        102, 255, 153,  # 2	left eyebrow
        0, 204, 153,    # 3	right eyebrow
        255, 255, 102,  # 4	left eye
        255, 255, 204,  # 5	right eye
        255, 153, 0,    # 6	nose
        255, 102, 255,  # 7	upper lip
        102, 0, 51,     # 8	inner mouth
        255, 204, 255,  # 9	lower lip
        255, 0, 10,     # 10 hair
    ]
    # fmt: on

    def __init__(self):
        """Generates a color map with a unique hue for each class in label.
        The hues are uniformly sampled from the range [0, 1). If the num of classes is too high, then the difference
        between neighboring hues will become indistinguishable"""
        self.color_palettes = {Palette.LAPA: self.palette_lapa}

    def map_color_palette(self, label: np.ndarray, palette: Palette) -> np.ndarray:
        """Generates RGB visualization of label by applying a color palette
        Label should contain an uint class index per pixel.

        Args:
            Args:
            label (numpy.ndarray): Each pixel has uint value corresponding to it's class index
                                   Shape: (H, W), dtype: np.uint8, np.uint16
            palette (Palette): Which color palette to use.

        Returns:
            numpy.ndarray: RGB image, with each class mapped to a unique color.
                           Shape: (H, W, 3), dtype: np.uint8
        """
        if len(label.shape) != 2:
            raise ValueError(f"Label must have shape: (H, W). Input: {label.shape}")
        if not (label.dtype == np.uint8):
            raise ValueError(f"Label must have dtype np.uint8. Input: {label.dtype}")
        if not isinstance(palette, Palette):
            raise ValueError(f"palette must be of type {Palette}. Input: {palette}")

        color_palette = self.color_palettes[palette]

        # Check that the pallete has enough colors
        if len(color_palette) < label.max():
            raise ValueError(
                f"The chosen color palette has only {len(color_palette)} values. It does not have"
                f" enough unique colors to represent all the values in the label ({label.max()})"
            )

        # Map grayscale image's pixel values to RGB color palette
        _im = Image.fromarray(label)
        _im.putpalette(color_palette)
        _im = _im.convert(mode="RGB")
        im = np.asarray(_im)

        return im

    def colorize_batch_numpy(self, batch_label: np.ndarray) -> np.ndarray:
        """Convert a batch of numpy labels to RGB

        Args:
            batch_label (numpy.ndarray): Shape: [N, H, W], dtype=np.uint8

        Returns:
            numpy.ndarray: Colorize labels. Shape: [N, H, W, 3], dtype=np.uint8
        """
        if not isinstance(batch_label, np.ndarray):
            raise TypeError(f"`batch_label` expected to be Numpy array. Got: {type(batch_label)}")
        if len(batch_label.shape) != 3:
            raise ValueError(f"`batch_label` expected shape [N, H, W]. Got: {batch_label.shape}")
        if batch_label.dtype != np.uint8:
            raise ValueError(f"`batch_label` must be of dtype np.uint8. Got: {batch_label.dtype}")

        batch_label_rgb = [self.map_color_palette(label, Palette.LAPA) for label in batch_label]
        batch_label_rgb = np.stack(batch_label_rgb, axis=0)
        return batch_label_rgb


if __name__ == "__main__":
    # Create dummy mask
    dummy_mask = np.zeros((512, 512), dtype=np.uint8)
    num_classes = 11
    cols = dummy_mask.shape[1]
    for idx, class_id in enumerate(range(num_classes)):
        cols_slice = cols // num_classes
        dummy_mask[:, cols_slice * idx : cols_slice * (idx + 1)] = class_id

    # Colorize mask
    label2rgb = LabelToRGB()
    colorized_mask = label2rgb.map_color_palette(dummy_mask, Palette.LAPA)

    # View mask
    img = Image.fromarray(colorized_mask)
    img.show()
