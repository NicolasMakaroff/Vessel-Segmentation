import torch
import torch.nn as nn

import pytorch_lightning as pl
#import sys
#sys.path.append(r"C:\path\to\your\project")
from losses.acmloss import *
from losses.diceloss import *
from losses.topoloss import *
from layers.layers import _downblock, _upblock, ConvBlock

class UNet(pl.LightningModule):
    """The U-Net architecture.
    
    See https://arxiv.org/pdf/1505.04597.pdf 
    """

    def __init__(self, num_channels: int=3, num_classes: int=2, antialias=False,
                 antialias_down_only=True):
        """Initialize a U-Net.
        
        Parameters
        ----------
        num_channels : int
            Number of input channels.
        num_classes : int
            Number of output classes.
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.antialias = antialias
        # whether to use blur conv transpose on the upsampling path
        self.antialias_up = self.antialias and not(antialias_down_only)

        self.in_conv = nn.Sequential(
            ConvBlock(num_channels, 64),
            ConvBlock(64, 64)
        )

        self.down1 = _downblock(64, 128, antialias=self.antialias)
        self.down2 = _downblock(128, 256, antialias=self.antialias)
        self.down3 = _downblock(256, 512, antialias=self.antialias)

        self.center = nn.Sequential(
            _downblock(512, 1024, antialias=self.antialias),
            nn.ConvTranspose2d(1024, 512, 2, stride=2)  # upscale
        )

        # reminder: convolves then upsamples
        self.up1 = _upblock(512, 256, antialias=self.antialias_up)
        self.up2 = _upblock(256, 128, antialias=self.antialias_up)
        self.up3 = _upblock(128, 64, antialias=self.antialias_up)

        self.out_conv = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor):
        x1 = self.in_conv(x)  # 64 * 1. * 1. ie 224
        x2 = self.down1(x1)  # 128 * 1/2 * 1/2
        x3 = self.down2(x2)  # 256 * 1/4 * 1/4
        x4 = self.down3(x3)  # 512 * 1/8 * 1/8
        x = self.center(x4)  # 512 * 1/8 * 1/8 ie 28
        x = self.up1(x, x4)  # 256 * 1/4 * 1/4 56
        x = self.up2(x, x3)  # 128 * 1/2 * 1/2 112
        x = self.up3(x, x2)
        z = torch.cat((x1, x), dim=1)
        out = self.out_conv(z)
        return out

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    def training_step(self, batch, batch_idx):

        x, y_true = batch
        y_pred = self(x)
        loss = ACMLoss(y_pred, y_true)
        loss_info = DiceLoss(y_pred,y_true)
        logs={"train_loss": loss}
        batch_dictionary={
            #REQUIRED: It is required for us to return "loss"
            "loss": loss,
            "dice": loss_info,
            #optional for batch logging purposes
            "log": logs,
        }
        self.log_dict(batch_dictionary, prog_bar = True,on_step=False,on_epoch=True)

        return batch_dictionary
 
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = ACMLoss(y_hat, y)
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss

