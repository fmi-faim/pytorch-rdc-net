import numpy as np
import torch
import torch.nn as nn

from ignite.utils import to_onehot

from source.rdcnet.lovasz_losses import lovasz_hinge


class InstanceEmbeddingLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(InstanceEmbeddingLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-6

    def forward(self, y_embeddings, y_true):
        losses = []

        for y_emb, gt_patch in zip(y_embeddings, y_true):
            if torch.any(gt_patch > 0):
                gt_one_hot = to_onehot(
                    gt_patch, num_classes=int(torch.max(gt_patch).item() + 1)
                )[0, 1:]
                counts = torch.sum(gt_one_hot, dim=(1, 2), keepdim=True)
                centers = (
                    torch.sum(
                        (
                            gt_one_hot.unsqueeze(0)
                            * y_emb.unsqueeze(1)
                        ),
                        dim=(2, 3),
                        keepdim=True,
                    )
                    / counts
                )

                center_dist = torch.norm(centers - y_emb.unsqueeze(1), dim=0)

                sigma = self.margin * (-2 * np.log(0.5)) ** -0.5
                probs = torch.exp(-0.5 * (center_dist / sigma) ** 2)

                losses.append(lovasz_hinge(probs * 2 - 1, gt_one_hot, per_image=False))

        if len(losses) > 0:
            return torch.mean(torch.stack(losses))
        else:
            return torch.tensor(0.0)
