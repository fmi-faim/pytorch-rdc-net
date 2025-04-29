from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

import torchist
from torch import optim, nn as nn

from pytorch_rdc_net.matching import matching
from pytorch_rdc_net.stacked_dilated_conv import StackedDilatedConv
from pytorch_rdc_net.utils import get_dimensional_modules


class RDCNet(pl.LightningModule):
    def __init__(
        self,
        dim: int,
        in_channels: int = 1,
        down_sampling_factor: tuple = (6, 6),
        down_sampling_channels: int = 8,
        spatial_dropout_p: float = 0.1,
        channels_per_group: int = 32,
        n_groups: int = 4,
        dilation_rates: list[int] = [1, 2, 4, 8, 16],
        steps: int = 6,
        lr: float = 0.001,
        instance_size: tuple = (20.0, 20.0),
    ):
        super(RDCNet, self).__init__()
        self.save_hyperparameters()
        self.dim = dim

        (
            Conv,
            ConvTranspose,
            Dropout,
            lovasz_softmax,
            InstanceEmbeddingLoss,
            self.median_filter,
        ) = get_dimensional_modules(dim)

        down_sampling_kernel = tuple(
            [max(3, dsf if dsf % 2 != 0 else dsf + 1) for dsf in down_sampling_factor]
        )
        self.in_conv = Conv(
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.down_sampling_channels,
            kernel_size=down_sampling_kernel,
            stride=self.hparams.down_sampling_factor,
            padding=tuple([k // 2 for k in down_sampling_kernel]),
        )

        self.spatial_dropout = Dropout(p=self.hparams.spatial_dropout_p)

        self.reduce_ch_conv = Conv(
            in_channels=self.hparams.channels_per_group * self.hparams.n_groups
            + self.hparams.down_sampling_channels,
            out_channels=self.hparams.channels_per_group * self.hparams.n_groups,
            kernel_size=1,
        )

        self.sd_conv = StackedDilatedConv(
            in_channels=self.hparams.channels_per_group * self.hparams.n_groups,
            out_channels=self.hparams.channels_per_group * self.hparams.n_groups,
            kernel_size=3,
            dilation_rates=self.hparams.dilation_rates,
            groups=self.hparams.n_groups,
            dim=self.dim,
        )

        up_sampling_kernel = tuple(
            [2 * dsk if dsk > 3 else 3 for dsk in down_sampling_kernel]
        )
        up_padding = tuple(
            [
                usk // 2 if dsf <= 3 else dsf // 2 + 1
                for dsf, usk in zip(down_sampling_factor, down_sampling_kernel)
            ]
        )
        output_padding = tuple(
            [
                -(-1 * s - 2 * p + (k - 1) + 1) % s
                for k, p, s in zip(
                    up_sampling_kernel, up_padding, self.hparams.down_sampling_factor
                )
            ]
        )
        self.out_conv = ConvTranspose(
            in_channels=self.hparams.channels_per_group * self.hparams.n_groups,
            out_channels=7 if dim == 2 else 11,
            kernel_size=up_sampling_kernel,
            stride=self.hparams.down_sampling_factor,
            padding=up_padding,
            output_padding=output_padding,
        )

        self.embedding_loss = InstanceEmbeddingLoss()
        self.semantic_loss = lovasz_softmax

        self.receptive_field_size = None
        self.coords = None

    def forward(self, x):
        assert all(
            x.shape[i + 2] % self.hparams.down_sampling_factor[i] == 0
            for i in range(self.dim)
        )
        x = self.in_conv(x)

        state = torch.zeros(
            x.shape[0],
            self.hparams.channels_per_group * self.hparams.n_groups,
            *x.shape[2:],
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
        state = self.out_conv(state)
        embeddings = state[:, : self.dim]
        lower_triangular = state[
            :, self.dim : self.dim + (self.dim * (self.dim + 1)) // 2
        ]
        semantic_classes = F.softmax(
            state[:, self.dim + (self.dim * (self.dim + 1)) // 2 :], dim=1
        )

        return embeddings, lower_triangular, semantic_classes

    def predict_instances(self, x, vote_masking_threshold=0.01):
        """
        Predict instance segmentations for the input batch.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, *spatial_dims).
            vote_masking_threshold (float): Threshold for vote masking.

        Returns:
            np.ndarray: Instance segmentations for the input batch.
        """
        self.eval()
        instance_segmentations = []
        with torch.no_grad():
            for patch in x:
                # Forward pass
                embeddings, lower_triangular, semantic = self(
                    patch.unsqueeze(0).to(self.device)
                )

                # Get instance segmentations
                label_img = self.get_instance_segmentations(
                    embeddings[0],
                    lower_triangular[0],
                    semantic[0],
                    vote_masking_threshold=vote_masking_threshold,
                )
                instance_segmentations.append(label_img.cpu().numpy()[np.newaxis])

        return np.stack(instance_segmentations)

    def get_instance_segmentations(
        self,
        embeddings,
        lower_triangular,
        semantic,
        vote_masking_threshold=0.01,
        max_instances: int = None,
    ):
        """
        Generate instance segmentations from embeddings, lower triangular matrix, and semantic predictions.

        Args:
            embeddings (torch.Tensor): Embedding tensor of shape (*spatial_dims, dim).
            lower_triangular (torch.Tensor): Lower triangular matrix tensor.
            semantic (torch.Tensor): Semantic predictions tensor.
            vote_masking_threshold (float): Threshold for vote masking.
            max_instances (int, optional): Maximum number of instances to predict. Defaults to None, which computes all instances.

        Returns:
            torch.Tensor: Instance segmentation map.
        """
        if max_instances is None:
            max_instances = torch.inf
        with torch.no_grad():
            embeddings = embeddings.detach()
            lower_triangular = lower_triangular.detach()
            semantic = semantic.detach()

            embeddings, lower_triangular, padding, semantic = self.pad_inputs(
                embeddings, lower_triangular, semantic
            )

            shape = embeddings.shape[-self.dim :]

            fg_mask = torch.argmax(semantic, dim=0).type(torch.bool)

            # Add coordinate grid to embeddings
            grid = self._get_coordinate_grid(embeddings)[0]
            embeddings = embeddings + grid

            # Histogram-based voting
            fg_embeddings = embeddings[:, fg_mask]
            bins = tuple(shape[i] for i in range(self.dim))
            low = tuple(0 for _ in range(self.dim))
            upp = tuple(
                shape[i] / self.hparams.instance_size[i] for i in range(self.dim)
            )
            votes = torchist.histogramdd(
                fg_embeddings.moveaxis(0, -1), bins=bins, low=low, upp=upp
            )
            votes[fg_mask == 0] = 0

            label_img = torch.zeros_like(votes, dtype=torch.int32, device=votes.device)
            label_id = 0

            while votes.max() > 0 and label_id < max_instances:
                center = torch.unravel_index(torch.argmax(votes), votes.shape)
                cov = self.construct_cov_matrix(center, lower_triangular)
                instance = self.compute_instance_probabilities(
                    center, cov, embeddings, grid, shape
                )

                instance[label_img > 0] = 0

                if torch.sum(instance) > 0:
                    label_id += 1
                    label_img[self.median_filter(instance) >= 0.5] = label_id

                fg_mask[*center] = 0
                fg_mask[instance >= vote_masking_threshold] = 0
                fg_embeddings = embeddings[:, fg_mask]

                votes = torchist.histogramdd(
                    fg_embeddings.moveaxis(0, -1), bins=bins, low=low, upp=upp
                )
                votes[fg_mask == 0] = 0

            # Remove padding
            slices = tuple(
                slice(padding[i * 2], -padding[i * 2 + 1])
                for i in reversed(range(self.dim))
            )
            return label_img[slices].detach().cpu()

    def pad_inputs(self, embeddings, lower_triangular, semantic):
        padding = [int(self.hparams.instance_size[i]) + 1 for i in range(self.dim)]
        padding = tuple(reversed([p for p in padding for _ in range(2)]))
        embeddings = F.pad(embeddings, padding)
        lower_triangular = F.pad(lower_triangular, padding)
        semantic = F.pad(semantic, padding)
        return embeddings, lower_triangular, padding, semantic

    def compute_instance_probabilities(self, center, cov, embeddings, grid, shape):
        target_centroid = grid[:, *center]
        for _ in range(self.dim):
            target_centroid = target_centroid.unsqueeze(-1)
        c_e = torch.square(embeddings - target_centroid)
        for _ in range(self.dim - 2):
            c_e = c_e.unsqueeze(-1)
        c_e = torch.moveaxis(c_e.view(self.dim, 1, -1), -1, 0)
        instance = torch.exp(-1 * (torch.transpose(c_e, 1, 2) @ cov @ c_e).view(*shape))
        return instance

    def construct_cov_matrix(self, center, lower_triangular):
        lt_matrix = torch.zeros(
            (self.dim, self.dim),
            device=lower_triangular.device,
            dtype=lower_triangular.dtype,
        )
        idx = 0
        for i in range(self.dim):
            lt_matrix[i, i] = F.softplus(lower_triangular[idx, *center])
            idx += 1
            for j in range(i + 1, self.dim):
                lt_matrix[j, i] = lower_triangular[idx, *center]
                idx += 1
        cov = lt_matrix @ lt_matrix.transpose(0, 1)
        return cov

    def _get_coordinate_grid(self, pred: torch.Tensor) -> torch.Tensor:
        if (
            self.coords is None
            or self.coords.shape[-self.dim :] != pred.shape[-self.dim :]
        ):
            grid = torch.meshgrid(
                [torch.arange(size) for size in pred.shape[-self.dim :]],
                indexing="ij",
            )
            grid = torch.stack(grid, dim=0).type(torch.float32)
            for i in range(self.dim):
                grid[i] = grid[i] / self.hparams.instance_size[i]
            grid = grid.unsqueeze(0)

            grid.requires_grad = False
            self.coords = grid.to(pred.device)

        return self.coords

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (tuple): A tuple containing input data and ground truth labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The total training loss for the batch.
        """
        x, gt_labels = batch
        embeddings, lower_triangular, semantic_classes = self(x)
        grid = self._get_coordinate_grid(embeddings)

        # Compute embedding loss
        embedding_loss, var_lt = self.embedding_loss(
            grid[0], embeddings + grid, lower_triangular, gt_labels
        )

        # Compute semantic loss
        semantic_loss = self.semantic_loss(
            semantic_classes, gt_labels > 0, per_image=True
        )

        # Total training loss
        train_loss = embedding_loss + semantic_loss + var_lt

        # Log metrics
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
            "var_lt", var_lt, on_step=False, on_epoch=True, prog_bar=True, logger=True
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
        """
        Perform a single validation step.

        Args:
            batch (tuple): A tuple containing input data and ground truth labels.
            batch_idx (int): Index of the batch.

        Returns:
            float: Mean true score for the batch.
        """
        x, gt_labels = batch
        gt = gt_labels.cpu().numpy()[0, 0]

        embeddings, lower_triangular, semantic_classes = self(x)

        # Generate instance segmentations
        instance_seg = self.get_instance_segmentations(
            embeddings[0],
            lower_triangular[0],
            semantic_classes[0],
            vote_masking_threshold=0.01,
            max_instances=2 * torch.max(gt_labels).item(),
        )

        # Compute metrics
        metrics = matching(
            y_true=gt,
            y_pred=instance_seg.cpu().numpy(),
            criterion="iou",
        )

        # Log metrics
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
        """
        Perform a single test step.

        Args:
            batch (tuple): A tuple containing input data and ground truth labels.
            batch_idx (int): Index of the batch.

        Returns:
            float: Mean true score for the batch.
        """
        x, gt_labels = batch
        gt = gt_labels.cpu().numpy()[0, 0]

        # Forward pass
        embeddings, lower_triangular, semantic_classes = self(x)

        # Generate instance segmentations
        instance_seg = self.get_instance_segmentations(
            embeddings[0],
            lower_triangular[0],
            semantic_classes[0],
            vote_masking_threshold=0.01,
        )

        # Compute metrics
        metrics = matching(
            y_true=gt,
            y_pred=instance_seg.cpu().numpy(),
            criterion="iou",
        )

        # Log metrics
        self.log(
            "test_precision",
            metrics.precision,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_recall",
            metrics.recall,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_f1",
            metrics.f1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_mean_matched_score",
            metrics.mean_matched_score,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_mean_true_score",
            metrics.mean_true_score,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_panoptic_quality",
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

    def get_receptive_field(self):
        """
        Calculate the receptive field of the network.

        Returns:
            tuple: Receptive field and its dimensions.
        """
        if self.receptive_field_size is None:
            rf = compute_receptive_field(
                dim=self.hparams.dim,
                down_sampling_factor=self.hparams.down_sampling_factor,
                dilation_rates=self.hparams.dilation_rates,
                steps=self.hparams.steps,
            )
            if self.dim == 2:
                h = np.where(rf.max((0, 2)))
                w = np.where(rf.max((0, 1)))
                self.receptive_field_size = (
                    max(h[0][-1] - h[0][0], w[0][-1] - w[0][0]),
                ) * 2
            elif self.dim == 3:
                d = np.where(rf.max((0, 2, 3)))
                h = np.where(rf.max((0, 1, 3)))
                w = np.where(rf.max((0, 1, 2)))
                self.receptive_field_size = (d[0][-1] - d[0][0],) + (
                    max(h[0][-1] - h[0][0], w[0][-1] - w[0][0]),
                ) * 2

        return self.receptive_field_size


def compute_receptive_field(
    dim: int,
    down_sampling_factor: Union[tuple[int, int], tuple[int, int, int]],
    dilation_rates: list[int],
    steps: int,
):
    rf_net = RDCNet(
        dim=dim,
        in_channels=1,
        down_sampling_factor=down_sampling_factor,
        down_sampling_channels=1,
        spatial_dropout_p=0,
        channels_per_group=1,
        n_groups=1,
        dilation_rates=dilation_rates,
        steps=steps,
    )
    for param in rf_net.parameters():
        param.requires_grad = False
        param.data = nn.parameter.Parameter(torch.ones_like(param))

    rf_net.eval()
    with torch.no_grad():
        shape = tuple(
            [4 * dsf * max(dilation_rates) * steps for dsf in down_sampling_factor]
        )
        # Create input tensor with a single activated point
        x = torch.zeros((1, 1, *shape), dtype=torch.float32)
        center = tuple(s // 2 for s in shape)
        x[(0, 0, *center)] = 1
        p_0 = rf_net(x)[0].detach()
        x[(0, 0, *center)] = 10
        p_1 = rf_net(x)[0].detach()

    # Calculate receptive field
    return (torch.abs(p_1 - p_0) > 0).sum(0).detach().cpu().numpy()
