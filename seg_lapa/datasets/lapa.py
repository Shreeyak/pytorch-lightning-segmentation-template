
import cv2
import enum
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, List

import albumentations as A
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from seg_lapa.utils.path_check import get_path


class DatasetSplit(enum.Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2


class LapaDataset(Dataset):
    """The Landmark guided face Parsing dataset (LaPa)
    Contains pixel-level annotations for face parsing.

    References:
        https://github.com/JDAI-CV/lapa-dataset
    """
    @enum.unique
    class LapaClassId(enum.IntEnum):
        # Mapping of the classes within the lapa dataset
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

    SUBDIR_IMAGES = 'images'
    SUBDIR_LABELS = 'labels'

    SUBDIR_SPLIT = {
        DatasetSplit.TRAIN: 'train',
        DatasetSplit.VAL: 'val',
        DatasetSplit.TEST: 'test'
    }

    def __init__(self,
                 root_dir: Union[str, Path],
                 data_split: DatasetSplit,
                 image_ext: Tuple[str] = ('*.jpg',),
                 label_ext: Tuple[str] = ('*.png',),
                 augmentations: Optional[A.Compose] = None):
        super().__init__()
        self.augmentations = augmentations
        self.image_ext = image_ext  # The file extensions of input images to search for in input dir
        self.label_ext = label_ext  # The file extensions of labels to search for in label dir
        self.root_dir = self._check_dir(root_dir)

        # Get subdirs for images and labels
        self.images_dir = self._check_dir(self.root_dir / self.SUBDIR_SPLIT[data_split] / self.SUBDIR_IMAGES)
        self.labels_dir = self._check_dir(self.root_dir / self.SUBDIR_SPLIT[data_split] / self.SUBDIR_LABELS)

        # Create list of filenames
        self._datalist_input = []  # Variable containing list of all input images filenames in dataset
        self._datalist_label = []  # Variable containing list of all ground truth filenames in dataset
        self._create_lists_filenames()

    def __len__(self):
        return len(self._datalist_input)

    def __getitem__(self, index):
        # Read input rgb imgs
        image_path = self._datalist_input[index]
        img = self._read_image(image_path)

        # Read ground truth labels
        label_path = self._datalist_label[index]
        label = self._read_label(label_path)

        # Apply image augmentations
        if self.augmentations is not None:
            augmented = self.augmentations(image=img, mask=label)
            img = augmented['image']
            label = augmented['mask']

        # Convert to Tensor. RGB images are normally numpy uint8 array with shape (H, W, 3).
        # RGB tensors should be (3, H, W) with dtype float32 in range [0, 1] (may change with normalization applied)
        img_tensor = torchvision.transforms.ToTensor()(img)
        label_tensor = torch.from_numpy(label)

        # TODO: Return dict
        # data = {
        #     'image': img_tensor,
        #     'label': label_tensor
        # }
        # return data

        return img_tensor, label_tensor.long()

    @staticmethod
    def _check_dir(dir_path: Union[str, Path]) -> Path:
        return get_path(dir_path, path_type='dir', must_exist=True)

    @staticmethod
    def _read_label(label_path: Path) -> np.ndarray:
        mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

        if len(mask.shape) != 2:
            raise RuntimeError(f'The shape of label must be (H, W). Got: {mask.shape}')

        return mask.astype(np.int32)

    @staticmethod
    def _read_image(image_path: Path) -> np.ndarray:
        mask = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if len(mask.shape) != 3:
            raise RuntimeError(f'The shape of image must be (H, W, C). Got: {mask.shape}')

        return mask

    def _create_lists_filenames(self):
        """Creates a list of filenames of images and labels in dataset"""
        self._datalist_input = self._get_matching_files_in_dir(self.images_dir, self.image_ext)
        self._datalist_label = self._get_matching_files_in_dir(self.labels_dir, self.label_ext)

        num_images = len(self._datalist_input)
        num_labels = len(self._datalist_label)
        if num_images != num_labels:
            raise ValueError(f'The number of images ({num_images}) and labels ({num_labels}) do not match.'
                             f'\n  Images dir: {self.images_dir}\n  Labels dir:{self.labels_dir}')

    def _get_matching_files_in_dir(self, data_dir: Union[str, Path], wildcard_patterns: Tuple[str]) -> List[Path]:
        """Get filenames within a dir that match a set of wildcard patterns
        Will not search within subdirectories.

        Args:
            data_dir: Directory to search within
            wildcard_patterns: Tuple of wildcard patterns matching required filenames. Eg: ('*.rgb.png', '*.rgb.jpg')

        Returns:
            list[Path]: List of paths to files found
        """
        data_dir = self._check_dir(data_dir)

        list_matching_files = []
        for ext in wildcard_patterns:
            list_matching_files += sorted(data_dir.glob(ext))

        if len(list_matching_files) == 0:
            raise ValueError('No matching files found in given directory.'
                             f'\n  Directory: {data_dir}\n  Search patterns: {wildcard_patterns}')

        return list_matching_files


class LaPaDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, resize_h: int, resize_w: int):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = get_path(data_dir, path_type='dir', must_exist=True)
        self.resize_h = resize_h
        self.resize_w = resize_w

        self.lapa_train = None
        self.lapa_val = None
        self.lapa_test = None

    def prepare_data(self):
        """download dataset, tokenize, etc"""
        '''
        Downloading original data from author's google drive link: 
            >>> import gdown
            >>> url = "https://drive.google.com/uc?export=download&id=1EtyCtiQZt2Y5qrb-0YxRxaVLpVcgCOQV"
            >>> output = "lapa-downloaded.tar.gz"
            >>> gdown.download(url, output, quiet=False, proxy=False)
        '''
        pass

    def setup(self, stage=None):
        # count number of classes, perform train/val/test splits, apply transforms, etc
        augs_train = self.get_augs_train()
        augs_test = self.get_augs_test()

        self.lapa_train = LapaDataset(root_dir=self.data_dir, data_split=DatasetSplit.TRAIN, augmentations=augs_train)
        self.lapa_test = LapaDataset(root_dir=self.data_dir, data_split=DatasetSplit.VAL, augmentations=augs_test)
        self.lapa_val = LapaDataset(root_dir=self.data_dir, data_split=DatasetSplit.TEST, augmentations=augs_test)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.lapa_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=True
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.lapa_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=False
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.lapa_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=False
        )
        return test_loader

    def get_augs_test(self):
        augs_test = A.Compose(
            [
                # Geometric Augs
                A.SmallestMaxSize(max_size=self.resize_h, interpolation=0, p=1.0),
                A.CenterCrop(height=self.resize_h, width=self.resize_w, p=1.0),
            ]
        )
        return augs_test

    def get_augs_train(self):
        augs_train = A.Compose(
            [
                # Geometric Augs
                A.SmallestMaxSize(max_size=self.resize_h, interpolation=0, p=1.0),
                A.CenterCrop(height=self.resize_h, width=self.resize_w, p=1.0),
            ]
        )
        return augs_train
