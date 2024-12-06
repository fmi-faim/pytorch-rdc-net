import numpy as np
import pytorch_lightning as pl

import sys
from faim_ipa.utils import get_git_root
from torch import optim

from source.matching import matching
from source.rdcnet.lovasz_losses import lovasz_softmax

sys.path.append(str(get_git_root()))

from source.rdcnet.embedding_loss import InstanceEmbeddingLoss
from source.rdcnet.stacked_dilated_conv import StackedDilatedConv2d

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torchist


def calc_same_pad(size: int, k: int, s: int, d: int) -> int:
    return max((math.ceil(size / s) - 1) * s + (k - 1) * d + 1 - size, 0)


class RDCNet2d(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        down_sampling_factor: int = 6,
        down_sampling_channels: int = 8,
        spatial_dropout_p: float = 0.1,
        channels_per_group: int = 32,
        n_groups: int = 4,
        dilation_rates: list[int] = [1, 2, 4, 8, 16],
        steps: int = 6,
        margin: int = 10,
        lr: float = 0.001,
        min_votes_per_instance: int = 5,
        start_val_metrics_epoch: int = 10,
    ):
        super(RDCNet2d, self).__init__()
        self.save_hyperparameters()

        down_sampling_kernel = max(
            3,
            down_sampling_factor
            if down_sampling_factor % 2 != 0
            else down_sampling_factor + 1,
        )
        self.in_conv = nn.Conv2d(
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.down_sampling_channels,
            kernel_size=down_sampling_kernel,
            stride=self.hparams.down_sampling_factor,
            padding="valid",
        )

        self.spatial_dropout = nn.Dropout2d(p=self.hparams.spatial_dropout_p)

        self.reduce_ch_conv = nn.Conv2d(
            in_channels=self.hparams.channels_per_group * self.hparams.n_groups
            + self.hparams.down_sampling_channels,
            out_channels=self.hparams.channels_per_group * self.hparams.n_groups,
            kernel_size=1,
        )

        self.sd_conv = StackedDilatedConv2d(
            in_channels=self.hparams.channels_per_group * self.hparams.n_groups,
            out_channels=self.hparams.channels_per_group * self.hparams.n_groups,
            kernel_size=3,
            dilation_rates=self.hparams.dilation_rates,
            groups=self.hparams.n_groups,
        )

        self.out_conv = nn.ConvTranspose2d(
            in_channels=self.hparams.channels_per_group * self.hparams.n_groups,
            out_channels=4,
            kernel_size=down_sampling_kernel,
            stride=self.hparams.down_sampling_factor,
        )

        self.embedding_loss = InstanceEmbeddingLoss(margin=self.hparams.margin)
        self.semantic_loss = lovasz_softmax

        self.start_val_metrics_epoch = start_val_metrics_epoch
        self.coords = None

    def forward(self, x):
        in_shape = x.shape
        y_pad = (
            self.hparams.down_sampling_factor
            - in_shape[-2] % self.hparams.down_sampling_factor
        )
        x_pad = (
            self.hparams.down_sampling_factor
            - in_shape[-1] % self.hparams.down_sampling_factor
        )
        x = F.pad(
            x, (y_pad // 2, y_pad // 2 + y_pad % 2, x_pad // 2, x_pad // 2 + x_pad % 2)
        )
        x = self.in_conv(x)

        state = torch.zeros(
            x.shape[0],
            self.hparams.channels_per_group * self.hparams.n_groups,
            x.shape[2],
            x.shape[3],
            dtype=x.dtype,
            device=x.device,
        )

        for _ in range(self.hparams.steps):
            delta = torch.cat([x, state], dim=1)
            delta = self.spatial_dropout(delta)
            delta = F.leaky_relu(delta)
            delta = self.reduce_ch_conv(delta)
            delta = F.leaky_relu(delta)
            delta = self.sd_conv(delta)
            state += delta

        state = F.leaky_relu(state)

        output = self.out_conv(state)
        output = output[:, :, : in_shape[-2], : in_shape[-1]]
        embeddings = output[:, :2]
        semantic_classes = F.softmax(output[:, 2:], dim=1)

        return embeddings, semantic_classes

    def predict_instances(self, x):
        self.eval()
        instance_segmentations = []
        with torch.no_grad():
            for patch in x:
                embeddings, semantic = self(patch.unsqueeze(0).to(self.device))
                label_img = self.get_instance_segmentations(
                    embeddings,
                    semantic,
                )
                instance_segmentations.append(label_img.cpu().numpy()[np.newaxis])

        return np.stack(instance_segmentations)

    def get_instance_segmentations(self, embeddings, semantic):
        with torch.no_grad():
            embeddings = embeddings.detach()
            semantic = semantic.detach()

            # pad to catch instances with centers outside the image.
            padding = self.hparams.margin * 2
            embeddings = F.pad(embeddings, (padding, padding, padding, padding))
            semantic = F.pad(semantic, (padding, padding, padding, padding))

            shape = embeddings.shape[-2:]

            fg_mask = torch.argmax(semantic[0], dim=0).type(torch.bool)

            grid = self._get_coordinate_grid(embeddings)
            embeddings = (embeddings + grid)[0]
            fg_embeddings = torch.round(embeddings[:, fg_mask])

            votes = torchist.histogramdd(
                fg_embeddings.moveaxis(0, -1),
                bins=(shape[0], shape[1]),
                low=(0, 0),
                upp=shape,
            )
            votes[votes < self.hparams.min_votes_per_instance] = 0
            votes[fg_mask == 0] = 0

            if votes.max() > 0:
                # local maxima suppression
                max_filtered = F.max_pool2d(
                    votes.unsqueeze(0).type(torch.float32),
                    kernel_size=2 * self.hparams.margin + 1,
                    stride=1,
                    padding=self.hparams.margin,
                )[0]
                votes[votes > 0] += 1

                # select embeddings which are less than margin away from any center
                centers = torch.clip(votes - max_filtered, 0, 1).type(torch.bool)
                center_coords = grid[0, :, centers]
                dists = embeddings.unsqueeze(1) - center_coords.unsqueeze(-1).unsqueeze(
                    -1
                )
                dists = torch.norm(dists, dim=0, p=None)
                dists = dists < self.hparams.margin

                # Convert to instance labels and apply foreground mask
                label_img = torch.concat([torch.zeros_like(dists[:1]), dists]).type(
                    torch.int32
                )
                label_img = torch.argmax(label_img, dim=0) * fg_mask
            else:
                label_img = torch.zeros(
                    shape, dtype=torch.int32, device=embeddings.device
                )

            # Remove padding
            label_img = label_img[padding:-padding, padding:-padding]

            return label_img.cpu()

    def _get_coordinate_grid(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Get the coordinate grid for the given shape.
        """
        if self.coords is None or self.coords.shape[-2:] != pred.shape[-2:]:
            grid = torch.meshgrid(
                [torch.arange(size) for size in pred.shape[-2:]],
                indexing="ij",
            )
            grid = torch.stack(grid, dim=0)
            grid = grid.unsqueeze(0)

            grid = grid.type(torch.float32)
            self.coords = grid.to(pred.device)

        return self.coords

    def training_step(self, batch, batch_idx):
        x, gt_labels = batch
        embeddings, semantic_classes = self(x)
        embeddings += self._get_coordinate_grid(embeddings)

        embedding_loss = self.embedding_loss(embeddings, gt_labels)
        semantic_loss = self.semantic_loss(
            semantic_classes, gt_labels > 0, per_image=False
        )
        train_loss = embedding_loss + semantic_loss
        self.log(
            "semantic_loss",
            semantic_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "embedding_loss",
            embedding_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_loss",
            train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, gt_labels = batch
        gt = gt_labels.cpu().numpy()[0, 0]
        if self.trainer.current_epoch >= self.start_val_metrics_epoch:
            embeddings, semantic_classes = self(x)

            instance_seg = self.get_instance_segmentations(embeddings, semantic_classes)

            metrics = matching(
                y_true=gt,
                y_pred=instance_seg.cpu().numpy(),
                criterion="iou",
            )
        else:
            metrics = matching(
                y_true=gt,
                y_pred=np.zeros_like(gt),
                criterion="iou",
            )

        self.log(
            "precision",
            metrics.precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "recall",
            metrics.recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "f1", metrics.f1, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "mean_matched_score",
            metrics.mean_matched_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "mean_true_score",
            metrics.mean_true_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "panoptic_quality",
            metrics.panoptic_quality,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return metrics.mean_true_score

    def test_step(self, batch, batch_idx):
        x, gt_labels = batch
        embeddings, semantic_classes = self(x)

        instance_seg = self.get_instance_segmentations(embeddings, semantic_classes)

        metrics = matching(
            y_true=gt_labels.cpu().numpy()[0, 0],
            y_pred=instance_seg.cpu().numpy(),
            criterion="iou",
        )

        self.log(
            "precision",
            metrics.precision,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "recall",
            metrics.recall,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "f1", metrics.f1, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "mean_matched_score",
            metrics.mean_matched_score,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "mean_true_score",
            metrics.mean_true_score,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "panoptic_quality",
            metrics.panoptic_quality,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return metrics.mean_true_score

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.trainer.max_epochs, eta_min=1e-5
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
