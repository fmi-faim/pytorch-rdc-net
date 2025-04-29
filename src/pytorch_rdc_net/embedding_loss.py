import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.utils import to_onehot
from pytorch_rdc_net.lovasz_losses import lovasz_hinge, lovasz_hinge3d


class AbstractInstanceEmbeddingLoss(nn.Module):
    def __init__(self, dim):
        super(AbstractInstanceEmbeddingLoss, self).__init__()
        self.dim = dim
        if self.dim == 2:
            self.lovasz_loss = lovasz_hinge
        elif self.dim == 3:
            self.lovasz_loss = lovasz_hinge3d
        else:
            raise ValueError("dim must be 2 or 3")

    def get_instance_size(self, gt_one_hot):
        return torch.sum(gt_one_hot, dim=tuple(range(1, self.dim + 1)), keepdim=True)

    def get_lower_triangular_values(self, y_lt, gt_one_hot, instance_size):
        instance_lower_triangular = gt_one_hot.unsqueeze(0) * y_lt.unsqueeze(1)
        lower_triangular_instance_mean = (
            torch.sum(
                instance_lower_triangular,
                dim=tuple(range(2, self.dim + 2)),
                keepdim=True,
            )
            / instance_size
        )

        lower_triangular_instance_variance = torch.sum(
            torch.pow(
                (instance_lower_triangular - lower_triangular_instance_mean.detach())
                * gt_one_hot,
                2,
            ),
            dim=tuple(range(2, self.dim + 2)),
            keepdim=True,
        ) / instance_size.unsqueeze(0)
        return lower_triangular_instance_mean, lower_triangular_instance_variance

    def compute_probabilities(self, y_emb, grid, gt_one_hot, instance_size, cov):
        centers_mean = (
            torch.sum(
                (gt_one_hot.unsqueeze(0) * grid.unsqueeze(1)),
                dim=tuple(range(2, self.dim + 2)),
                keepdim=True,
            )
            / instance_size
        )
        similar_embs = torch.square(y_emb.unsqueeze(1) - centers_mean)
        probs = []
        for i in range(similar_embs.shape[1]):
            s_e = similar_embs[:, i].unsqueeze(1)
            s_e = torch.moveaxis(s_e.view(self.dim, 1, -1), -1, 0)
            probs.append(
                torch.exp(
                    -1
                    * (torch.transpose(s_e, 1, 2) @ cov[i] @ s_e).view(
                        1, *similar_embs.shape[-self.dim :]
                    )
                )
            )
        return torch.cat(probs, dim=0)

    def forward(self, grid, y_embeddings, y_lower_triangular, y_true):
        losses = []
        lt_vars = []
        for y_emb, y_lt, gt_patch in zip(y_embeddings, y_lower_triangular, y_true):
            if torch.any(gt_patch > 0):
                gt_one_hot = to_onehot(
                    gt_patch, num_classes=int(torch.max(gt_patch).item() + 1)
                )[0, 1:]
                instance_size = self.get_instance_size(gt_one_hot=gt_one_hot)
                lt_mean, lt_var = self.get_lower_triangular_values(
                    y_lt=y_lt, gt_one_hot=gt_one_hot, instance_size=instance_size
                )
                cov = self.get_covariance_matrix(lt_mean)

                probabilities = self.compute_probabilities(
                    y_emb=y_emb,
                    grid=grid,
                    gt_one_hot=gt_one_hot,
                    instance_size=instance_size,
                    cov=cov,
                )

                losses.append(
                    self.lovasz_loss(probabilities * 2 - 1, gt_one_hot, per_image=False)
                )
                lt_vars.append(torch.mean(lt_var))

        if len(losses) > 0:
            return torch.mean(torch.stack(losses)), torch.mean(torch.stack(lt_vars))
        else:
            return torch.tensor(0.0), torch.tensor(0.0)

    def get_covariance_matrix(self, lt_mean):
        raise NotImplementedError("This method must be implemented in subclasses.")


class InstanceEmbeddingLoss2d(AbstractInstanceEmbeddingLoss):
    def __init__(self):
        super(InstanceEmbeddingLoss2d, self).__init__(dim=2)

    def get_covariance_matrix(self, lt_mean):
        lt_matrix = torch.zeros(
            (lt_mean.shape[1], 2, 2), device=lt_mean.device, dtype=lt_mean.dtype
        )
        lt_matrix[:, 0, 0] = F.softplus(lt_mean[0, :, 0, 0])
        lt_matrix[:, 1, 1] = F.softplus(lt_mean[1, :, 0, 0])
        lt_matrix[:, 1, 0] = lt_mean[2, :, 0, 0]
        return torch.bmm(lt_matrix, lt_matrix.transpose(1, 2))


class InstanceEmbeddingLoss3d(AbstractInstanceEmbeddingLoss):
    def __init__(self):
        super(InstanceEmbeddingLoss3d, self).__init__(dim=3)

    def get_covariance_matrix(self, lt_mean):
        lt_matrix = torch.zeros(
            (lt_mean.shape[1], 3, 3), device=lt_mean.device, dtype=lt_mean.dtype
        )
        lt_matrix[:, 0, 0] = F.softplus(lt_mean[0, :, 0, 0, 0])
        lt_matrix[:, 1, 1] = F.softplus(lt_mean[1, :, 0, 0, 0])
        lt_matrix[:, 2, 2] = F.softplus(lt_mean[2, :, 0, 0, 0])
        lt_matrix[:, 1, 0] = lt_mean[3, :, 0, 0, 0]
        lt_matrix[:, 2, 0] = lt_mean[4, :, 0, 0, 0]
        lt_matrix[:, 2, 1] = lt_mean[5, :, 0, 0, 0]
        return torch.bmm(lt_matrix, lt_matrix.transpose(1, 2))
