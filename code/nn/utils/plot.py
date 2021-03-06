"""Plotting utilities."""
import torch
import torch.nn.functional as F
import numpy as np


from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from typing import Tuple

def denormalize(image: torch.Tensor, normalizer=None, mean=0, std=1):
    """Convert normalized image Tensor to Numpy image array."""
    image = np.moveaxis(image.numpy(), 0, -1)
    if normalizer is not None:
        mean = normalizer.mean
        std = normalizer.std
    image = (image * std + mean).clip(0, 1)
    return image

def plot_prediction(img: torch.Tensor, pred_mask: torch.Tensor, target: torch.Tensor, mean, std, writer: SummaryWriter = None, apply_softmax=True):
    """Plot the original image, heatmap of predicted class probabilities, and target mask.
    
    Parameters
    ----------
    We expect the inputs to be 4D mini-batch `Tensor`s of shape (B x C x H x W) (except for target which can be B x H x W and is handled in that case).
    The `img` image tensor is expected to be in cv2 `(B,G,R)` format. `pred_mask` is expected to be pre-Softmax unless `apply_softmax` is True.
    """
    batch_size = min(12, img.shape[0])  # never plot more than 2 images
    ncol = batch_size // 4
    img = make_grid(img, 4)
    # put on CPU, denormalize
    # GREEN MODE
    mean = mean[1]
    std = std[1]
    img = denormalize(img.data.cpu(), mean=mean, std=std)
    if target is not None:
        num_plots = 3
        if target.ndim == 3:
            # put in format (B, C, H, W) i.e. add the channel dimension
            target = target.unsqueeze(1)
        target = make_grid(target, 4)
        target = target.detach().cpu().numpy()
        # collapse useless dimension
        target = target[0]
    else:
        num_plots = 2
    
    if apply_softmax:
        pred_mask = F.softmax(pred_mask.data.cpu(), dim=1)
    pred_mask = make_grid(pred_mask, 4).numpy()
    pred_mask = pred_mask[1]  # class 1
    
    norm = colors.PowerNorm(0.5, vmin=0., vmax=1., clip=True)
    splt_nums = (1, num_plots)
    fig, axes = plt.subplots(*splt_nums, figsize=(6*num_plots, 5), dpi=70)
    fig: plt.Figure
    (ax1, ax2, ax3) = axes

    ax1.imshow(img)
    ax1.set_title("Base image")
    ax1.axis('off')
    
    ax2.imshow(pred_mask, norm=norm)
    ax2.set_title("Mask probability map")
    ax2.axis('off')
    
    ax3.imshow(target, cmap="gray")
    ax3.set_title("Real mask")
    ax3.axis('off')

    fig.tight_layout()
    return fig

def make_overlay(proba_map, max_alpha=.8):
    """Make transparent overlay image from probability map to
    superimpose over input for plotting.
    
    Parameters
    ----------
    max_alpha : float
        Maximum alpha for the overlay map.
    """
    prob_overlay = np.ones(proba_map.shape + (4,))
    prob_overlay[..., [0, 1]] = 0
    prob_overlay[..., 3] = proba_map * max_alpha
    return prob_overlay


def plot_with_overlay(img, probas_, fig: plt.Figure=None, figsize=None, **kwargs):
    if fig is None:
        fig: plt.Figure = plt.figure(figsize=figsize)

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img)
    ax.set_title("Initial image")
    ax.axis('off')

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(probas_)
    ax.set_title("Proba map")
    ax.axis('off')

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(img, extent=[0, 1, 0, 1])
    prob_overlay = make_overlay(probas_)
    ax.imshow(prob_overlay, extent=[0, 1, 0, 1])
    ax.set_title("Overlay")
    ax.axis('off')
    fig.tight_layout()