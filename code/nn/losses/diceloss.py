import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

import matplotlib.pyplot as plt

EPSILON = np.finfo(float).eps
class DiceLoss(nn.Module):
    """
    
    Attributes
    ----------
    soft : bool
        Variant of the Dice loss to use.
    """
    
    def __init__(self, apply_softmax=True, variant=None):
        super().__init__()
        self.apply_softmax = apply_softmax
        self.variant = str(variant).lower()
    
    def dice_loss(self, input: torch.Tensor, target: torch.Tensor, softmax=True, smooth=EPSILON) -> torch.Tensor:
        """Regular Dice loss.

        Parameters
        ----------
        input: Tensor
        (N, K, H, W) Predicted classes for each pixel.
        target: LongTensor
        (N, K, H, W) Tensor of pixel labels where `K` is the no. of classes.
        softmax: bool
        Whether to apply `F.softmax` to input to get class probabilities.
        """
        target = F.one_hot(target.to(torch.int64)).permute(0, 3, 1, 2)
        dims = (1, 2, 3)  # sum over C, H, W
        if softmax:
            input = F.softmax(input, dim=1)
        intersect = torch.sum(input * target, dim=dims)
        denominator = torch.sum(input + target, dim=dims)
        loss = 1 - (2 * intersect + smooth) / (denominator + smooth)
        return loss.mean()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return self.dice_loss(input, target, self.apply_softmax)