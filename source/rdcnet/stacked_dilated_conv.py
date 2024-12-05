import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedDilatedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates, groups):
        assert in_channels == out_channels, "in_channels must be equal to out_channels"
        super(StackedDilatedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, groups=groups
        )
        self.dilation_rates = dilation_rates

        self.reduction_conv = nn.Conv2d(
            in_channels=len(dilation_rates) * out_channels,
            out_channels=out_channels,
            kernel_size=1,
            groups=groups,
        )

    def forward(self, x):
        outputs = []
        for dilation_rate in self.dilation_rates:
            outputs.append(
                F.conv2d(
                    x,
                    self.weight,
                    self.bias,
                    padding=dilation_rate,
                    dilation=dilation_rate,
                    groups=self.groups,
                )
            )

        outputs = [
            # split the output into groups
            torch.split(out, out.shape[1] // self.groups, dim=1)
            for out in outputs
        ]

        outputs = list(itertools.chain(*zip(*outputs)))
        outputs = torch.concat(outputs, dim=1)
        outputs = F.leaky_relu(outputs)

        return self.reduction_conv(outputs)
