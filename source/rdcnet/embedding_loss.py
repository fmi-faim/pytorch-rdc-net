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

    def forward(self, y_embeddings, y_sigma, y_true):
        losses = []

        for y_emb, y_sig, gt_patch in zip(y_embeddings, y_sigma, y_true):
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
                instance_sigmas = gt_one_hot.unsqueeze(0) * y_sig.unsqueeze(1)
                sigmas = torch.sum(instance_sigmas, dim=(2, 3),
                                   keepdim=True) / counts

                var_sigmas = torch.sum(
                    torch.pow((instance_sigmas - sigmas.detach()) * gt_one_hot, 2)
                ) / counts

                center_dist = torch.norm(centers - y_emb.unsqueeze(1), dim=0)

                # sigma = self.margin * (-2 * np.log(0.5)) ** -0.5
                probs = torch.exp(-0.5 * (center_dist / sigmas) ** 2)

                losses.append(
                    lovasz_hinge(probs * 2 - 1, gt_one_hot, per_image=False) +
                    torch.mean(var_sigmas)
                )

        if len(losses) > 0:
            return torch.mean(torch.stack(losses))
        else:
            return torch.tensor(0.0)
