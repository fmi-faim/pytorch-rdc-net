import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.utils import to_onehot

from source.rdcnet.lovasz_losses import lovasz_hinge


class InstanceEmbeddingLoss(nn.Module):
    def __init__(self, px_classifier: nn.Module):
        super(InstanceEmbeddingLoss, self).__init__()
        self.px_classifier = px_classifier
        self.eps = 1e-6

    def forward(self, y_embeddings, y_true):
        losses = []

        for y_emb, gt_patch in zip(y_embeddings, y_true):
            if torch.any(gt_patch > 0):
                y_emb = F.pad(y_emb, (32, 32, 32, 32), value=0)
                gt_patch = F.pad(gt_patch, (32, 32, 32, 32), value=0)
                gt_one_hot = to_onehot(
                    gt_patch, num_classes=int(torch.max(gt_patch).item() + 1)
                )[0, 1:]
                counts = torch.sum(gt_one_hot, dim=(1, 2), keepdim=True).detach() + 1e-6
                centers = (
                    torch.sum(
                        (gt_one_hot.unsqueeze(0) * y_emb.unsqueeze(1)),
                        dim=(2, 3),
                        keepdim=True,
                    )
                    / counts
                ).detach()

                sims = torch.moveaxis((centers - y_emb.unsqueeze(1)), 0, 1) / 32.0
                sims = torch.clip(sims, -2.0, 2.0)
                probs = self.px_classifier.fold(
                    self.px_classifier(self.px_classifier.flatten(sims)),
                    (1,) + sims.shape[2:],
                )

                losses.append(lovasz_hinge(probs[:, 0], gt_one_hot, per_image=False))

        if len(losses) > 0:
            return torch.mean(torch.stack(losses))
        else:
            return torch.tensor(0.0)
