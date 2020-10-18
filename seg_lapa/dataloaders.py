
import cv2
import enum
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, List

import albumentations as A
import torch
import torchvision
from torch.utils.data import Dataset


class LapaDataset(Dataset):
    """The Landmark guided face Parsing dataset (LaPa)
    Contains pixel-level annotations for face parsing.

    References:
        https://github.com/JDAI-CV/lapa-dataset
    """
    @enum.unique
    class LapaClassId(enum.IntEnum):
        BACKGROUND = 0
        SKIN = 1
        EYEBROW_LEFT = 2
        EYEBROW_RIGHT = 3
        EYE_LEFT = 4
        EYE_RIGHT = 5
        NOSE = 6
        LIP_UPPER = 7
        INNER_MOUTH = 8
        LIP_LOWER = 9
        HAIR = 10

    def __init__(self,
                 image_dir: str,
                 label_dir: str,
                 image_ext: Tuple[str] = ('*.jpg',),
                 label_ext: Tuple[str] = ('*.png',),
                 augmentations: Optional[A.Compose] = None):
        super().__init__()
        self.augmentations = augmentations

        # Create list of filenames
        self._datalist_input = []  # Variable containing list of all input images filenames in dataset
        self._datalist_label = []  # Variable containing list of all ground truth filenames in dataset
        self.image_ext = image_ext  # The file extensions of input images to search for in input dir
        self.label_ext = label_ext  # The file extensions of labels to search for in label dir
        self._create_lists_filenames(image_dir, label_dir)

    def __len__(self):
        return len(self._datalist_input)

    def __getitem__(self, index):
        # Open input rgb imgs
        image_path = self._datalist_input[index]
        img = self._read_image(image_path)

        # Open labels
        label_path = self._datalist_label[index]
        label = self._read_label(label_path)

        # Apply image augmentations and convert to Tensor
        if self.augmentations is not None:
            augmented = self.augmentations(image=img, mask=label)
            img = augmented['image']
            label = augmented['mask']

        img_tensor = torchvision.transforms.ToTensor()(img)
        label_tensor = torch.from_numpy(label.astype(np.float32))

        # data = {
        #     'image': img_tensor,
        #     'label': label_tensor
        # }
        # return data

        return img_tensor, label_tensor.long()

    @staticmethod
    def _read_label(label_path: Path) -> np.ndarray:
        mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

        if len(mask.shape) != 2:
            raise RuntimeError(f'The shape of label must be (H, W). Got: {mask.shape}')

        return mask

    @staticmethod
    def _read_image(image_path: Path) -> np.ndarray:
        mask = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if len(mask.shape) != 3:
            raise RuntimeError(f'The shape of image must be (H, W, C). Got: {mask.shape}')

        return mask

    def _create_lists_filenames(self, images_dir: Union[str, Path], labels_dir: Union[str, Path]):
        """Creates a list of filenames of images and labels in dataset

        Args:
            images_dir: Path to the dir where images are stored
            labels_dir: Path to the dir where labels are stored

        Raises:
            ValueError: Number of images and labels do not match
        """
        self._datalist_input = self._get_matching_files_in_dir(images_dir, self.image_ext)
        self._datalist_label = self._get_matching_files_in_dir(labels_dir, self.label_ext)

        num_images = len(self._datalist_input)
        num_labels = len(self._datalist_label)
        if num_images != num_labels:
            raise ValueError(f'The number of images ({num_images}) and labels ({num_labels}) do not match.'
                             f'\n  Images dir: {images_dir}\n  Labels dir:{labels_dir}')

    @staticmethod
    def _get_matching_files_in_dir(images_dir: Union[str, Path], images_ext: Tuple[str]) -> List[Path]:
        """Get filenames within a dir that match a set of patterns

        Args:
            images_dir: Directory to search within
            images_ext: List of search strings for filename ext. Eg: ['.rgb.png']

        Returns:
            list[str]: List of paths to files found
        """
        images_dir = Path(images_dir)
        if not images_dir.is_dir():
            raise ValueError(f'Could not find dir: {images_dir}')

        list_matching_files = []
        for ext in images_ext:
            list_matching_files += sorted(images_dir.glob(ext))

        if len(list_matching_files) == 0:
            raise ValueError('No matching files found in given directory.'
                             f'\n  Searched dir: {images_dir}\n  Search patterns: {images_ext}')

        return list_matching_files



"""
>>> import gdown
>>> url = "https://drive.google.com/uc?export=download&id=1EtyCtiQZt2Y5qrb-0YxRxaVLpVcgCOQV"
>>> output = 'lapa-downloaded.tar.gz'
>>> gdown.download(url, output, quiet=False, proxy=False)
"""