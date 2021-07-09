import glob
import torch
import numpy as np
import pandas as pd
import os

import cv2
try : 
    from config import GAMMA_CORRECTION
except :
    GAMMA_CORRECTION = 1.2
import imageio
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
import albumentations.augmentations.functional as F

from albumentations import Compose, Resize, RandomSizedCrop
from albumentations import ElasticTransform, RandomScale, RandomGamma
from albumentations import OneOf, Rotate, GaussianBlur, CLAHE, Lambda
from albumentations import VerticalFlip, HorizontalFlip, Resize, Normalize
from albumentations.pytorch import ToTensorV2 as ToTensor

import json

## Define the data augmentation pipeline

MAX_SIZE = 512

def _make_train_transform(mean=0, std=1):
    _train = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(90, p=.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        OneOf([
            RandomSizedCrop((MAX_SIZE, MAX_SIZE), 400, 400, p=.8),
            Resize(400, 400, p=.2),
        ], p=1),
        # ElasticTransform(sigma=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=.1),
        GaussianBlur(blur_limit=3, p=.2),
        CLAHE(always_apply=True),
        Normalize(mean, std, always_apply=True),
        ToTensor(always_apply=True)
    ])
    return _train


def get_transforms():

    mean=[0.43008124828338623,
            0.1420290619134903,
            0.054625432938337326],
    std=[0.3244638442993164,
            0.11674251407384872,
            0.04332775995135307]

    train_transform = _make_train_transform(
        mean=mean,
        std=std
        )

    val_transform = Compose([
        Resize(400, 400),
        CLAHE(always_apply=True),
        Normalize(mean=mean,
                std=std),
        ToTensor(),
    ])

    return train_transform, val_transform    


class DriveDataset(VisionDataset):
    """DRIVE vessel segmentation dataset.
    
    We handle the mask/segmentation mask using the albumentations API, inspired by
    https://github.com/choosehappy/PytorchDigitalPathology/blob/master/segmentation_epistroma_unet/train_unet_albumentations.py
    
    Parameters
    ----------
    transforms
        Applies to both image, mask and target segmentation mask (when available).
    subset : slice
        Subset of indices of the dataset we want to use.
    green_only : bool
        Only use the green channel (idx 1).
    """
    
    def __init__(self, root: str, transforms=None, train: bool=False, subset: slice=None, return_mask=False, green_only=True):
        """
        Parameters
        ----------
        subset : slice
            Slice of data files on which to train.
        """
        super().__init__(root, transforms=transforms)
        self.train = train
        self.return_mask = return_mask
        self.green_only = green_only
        self.images = sorted(glob.glob(os.path.join(root, "images/*.tif")))
        self.masks = sorted(glob.glob(os.path.join(root, "mask/*.gif")))
        if subset is not None:
            if isinstance(subset, slice):
                self.images = self.images[subset]
                self.masks = self.masks[subset]
            else:
                self.images = [self.images[i] for i in subset]
                self.masks = [self.masks[i] for i in subset]
        self.targets = None
        if train:
            self.targets = sorted(glob.glob(os.path.join(root, "1st_manual/*.gif")))
            if subset is not None:
                if isinstance(subset, slice):
                    self.targets = self.targets[subset]
                else:
                    self.targets = [self.targets[i] for i in subset]
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = F.gamma_transform(img, GAMMA_CORRECTION)
        mask = imageio.imread(mask_path)
        if self.train:
            tgt_path = self.targets[index]
            target = imageio.imread(tgt_path)
            if self.transforms is not None:
                augmented = self.transforms(image=img, masks=[mask, target])
                img = augmented['image']
                if self.green_only:
                    img = img[[1]]
                mask, target = augmented['masks']
                # if isinstance(img, np.ndarray):
                #     img[mask == 0] = 0
                # else:
                #     img[:, mask == 0] = 0
                target = target.astype(int) / 255
            if self.return_mask:
                return img, torch.from_numpy(mask).long(), torch.from_numpy(target).long()
            else:
                return img, torch.from_numpy(target).long()
        else:
            if self.transforms is not None:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented['image']
                if self.green_only:
                    img = img[[1]]
                mask = augmented['mask']
                # if isinstance(np.ndarray, img):
                #     img[mask == 0] = 0
                # else:
                #     img[:, mask == 0] = 0
            if self.return_mask:
                return img, torch.from_numpy(mask).long()
            return img
