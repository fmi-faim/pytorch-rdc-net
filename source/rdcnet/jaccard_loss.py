import torch
import torch.nn as nn

from ignite.utils import to_onehot


class JaccardLoss(nn.Module):
    def __init__(self, eps=1e-6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.eps = eps

    def forward(self, y_pred, y_true):
        mask = (y_true >= 0).type(torch.float32)
        gt_one_hot = to_onehot((y_true[:, 0] > 0).type(y_true.dtype), num_classes=2)
        y_pred = y_pred * mask

        intersection = torch.sum(gt_one_hot * y_pred, dim=(2, 3))
        union = torch.sum(gt_one_hot + y_pred, dim=(2, 3)) - intersection

        jaccard = 1.0 - (intersection + self.eps) / (union + self.eps)
        return torch.mean(jaccard)
