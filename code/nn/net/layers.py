import torch
import torch.nn as nn

class _downblock(nn.Module):

    def __init__(self, in_channels, out_channels, n_convs=2, antialias=False):
        super().__init__()
        layers = [
            ConvBlock(in_channels, out_channels)
        ] + [
            ConvBlock(out_channels, out_channels)
            for _ in range(n_convs-1)
        ]
        # maxpooling over patches of size 2
        if not antialias:
            self.mp = nn.MaxPool2d(kernel_size=2)  # stride is equal to kernel_size (2)
        else:
            self.mp = custom_layers.MaxBlurPool2d(kernel_size=2, channels=in_channels)
            # self.mp = MaxBlurPool2d(kernel_size=2)  # ceil mode for consistency
            # self.mp.padding = (1, 1)  # fix some odd problem with dimensionality
        self.conv = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        x = self.mp(input)
        x = self.conv(x)
        return x


class _upblock(nn.Module):

    def __init__(self, in_channels, out_channels, n_convs=2, n_connect=2, antialias=False):
        """
        
        Parameters
        ----------
        n_connect : int
            Multiplicator for the number of input for the 1st convblock after
            the upsampling convolution (useful for skip connections).
        """
        super().__init__()
        self.antialias = antialias
        layers = [
            # expects multiple of channels
            ConvBlock(n_connect * in_channels, in_channels)
        ] + [
            ConvBlock(in_channels, in_channels)
            for _ in range(n_convs-1)
        ]
        self.conv = nn.Sequential(*layers)
        # counts as one convolution
        if not self.antialias:
            self.upconv = nn.ConvTranspose2d(in_channels, out_channels,
                                            kernel_size=2, stride=2)
        else:
            self.upconv = custom_layers.BlurConvTranspose(in_channels, out_channels,
                                                          kernel_size=2, stride=2)

    def forward(self, x: Tensor, skip: Tensor=None) -> Tensor:
        if skip is not None:
            z = torch.cat((skip, x), dim=1)
        else:
            z = x
        z = self.conv(z)
        out = self.upconv(z)  # deconvolve
        return out
