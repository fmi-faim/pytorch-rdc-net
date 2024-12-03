import numpy as np
import torch
import torch.nn as nn

from ignite.utils import to_onehot


class InstanceEmbeddingLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(InstanceEmbeddingLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-6

    def forward(self, y_pred, y_true):
        losses = []

        for y_patch, gt_patch in zip(y_pred, y_true):
            if torch.any(gt_patch > 0):
                gt_one_hot = to_onehot(gt_patch, num_classes=gt_patch.max() + 1)[0, 1:]
                counts = torch.sum(gt_one_hot, dim=(1, 2), keepdim=True)
                centers = (
                    torch.sum(
                        (gt_one_hot.unsqueeze(0) * y_patch.unsqueeze(1)),
                        dim=(2, 3),
                        keepdim=True,
                    )
                    / counts
                )

                center_dist = torch.norm(centers - y_patch.unsqueeze(1), dim=0)

                sigma = self.margin * (-2 * np.log(0.5)) ** -0.5
                probs = torch.exp(-0.5 * (center_dist / sigma) ** 2)

                intersection = torch.sum(gt_one_hot * probs, dim=(1, 2))
                union = torch.sum(gt_one_hot + probs, dim=(1, 2)) - intersection

                jaccard = 1.0 - (intersection + self.eps) / (union + self.eps)
                losses.append(torch.mean(jaccard))

        if len(losses) > 0:
            return torch.mean(torch.stack(losses))
        else:
            return torch.tensor(0.0)
