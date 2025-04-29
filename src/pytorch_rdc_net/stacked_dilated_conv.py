import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedDilatedConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation_rates, groups, dim=2
    ):
        """
        A unified class for Stacked Dilated Convolutions supporting both 2D and 3D.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel.
            dilation_rates (list): List of dilation rates for the stacked convolutions.
            groups (int): Number of groups for grouped convolution.
            dim (int): Dimensionality of the convolution (2 for 2D, 3 for 3D).
        """
        super(StackedDilatedConv, self).__init__()
        assert in_channels == out_channels, "in_channels must be equal to out_channels"
        assert dim in [2, 3], "dim must be 2 (2D) or 3 (3D)"
        self.dim = dim
        self.dilation_rates = dilation_rates

        # Choose the appropriate convolutional layer based on the dimension
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        self.base_conv = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
        )
        self.reduction_conv = Conv(
            in_channels=len(dilation_rates) * out_channels,
            out_channels=out_channels,
            kernel_size=1,
            groups=groups,
        )

    def forward(self, x):
        outputs = []
        # Choose the appropriate functional convolution based on the dimension
        conv_func = F.conv2d if self.dim == 2 else F.conv3d

        for dilation_rate in self.dilation_rates:
            outputs.append(
                conv_func(
                    x,
                    self.base_conv.weight,
                    self.base_conv.bias,
                    padding=dilation_rate,
                    dilation=dilation_rate,
                    groups=self.base_conv.groups,
                )
            )

        outputs = [
            # Split the output into groups
            torch.split(out, out.shape[1] // self.base_conv.groups, dim=1)
            for out in outputs
        ]

        outputs = list(itertools.chain(*zip(*outputs)))
        outputs = torch.concat(outputs, dim=1)
        outputs = F.leaky_relu(outputs)

        return self.reduction_conv(outputs)
