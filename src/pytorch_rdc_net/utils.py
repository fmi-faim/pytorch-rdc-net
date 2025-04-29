from torch import nn as nn
from torch.nn import functional as F

from pytorch_rdc_net.embedding_loss import (
    InstanceEmbeddingLoss2d,
    InstanceEmbeddingLoss3d,
)
from pytorch_rdc_net.lovasz_losses import lovasz_softmax_2d, lovasz_softmax_3d


def get_dimensional_modules(dim: int):
    """
    Get the appropriate dimensional modules based on the specified dimension.

    Parameters:
        dim (int): The dimension (2 or 3) for which to get the modules.

    Returns:
        tuple: A tuple containing the appropriate Conv, ConvTranspose, Dropout,
        lovasz_softmax, and InstanceEmbeddingLoss classes.
    """
    if dim == 2:
        return (
            nn.Conv2d,
            nn.ConvTranspose2d,
            nn.Dropout2d,
            lovasz_softmax_2d,
            InstanceEmbeddingLoss2d,
            median_filter_2d,
        )
    elif dim == 3:
        return (
            nn.Conv3d,
            nn.ConvTranspose3d,
            nn.Dropout3d,
            lovasz_softmax_3d,
            InstanceEmbeddingLoss3d,
            median_filter_3d,
        )
    else:
        raise ValueError("Invalid dimension. Only 2D and 3D are supported.")


def median_filter_2d(input_tensor):
    patches = (
        F.pad(input_tensor, (1, 1, 1, 1), mode="constant")
        .unfold(0, 3, 1)
        .unfold(1, 3, 1)
        .contiguous()
        .view(-1, 9)
    )
    return patches.median(dim=1)[0].view(input_tensor.shape)


def median_filter_3d(input_tensor):
    patches = (
        F.pad(input_tensor, (1, 1, 1, 1, 1, 1), mode="constant")
        .unfold(0, 3, 1)
        .unfold(1, 3, 1)
        .unfold(2, 3, 1)
        .contiguous()
        .view(-1, 27)
    )
    return patches.median(dim=1)[0].view(input_tensor.shape)
