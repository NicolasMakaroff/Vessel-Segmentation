from code.nn.utils.datasets import DriveDataset
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data.dataloader import DataLoader

from code.nn.unet import UNet
from code.nn.utils.datasets import DriveDataset, get_transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--loss", type=str, default="crossentropy")#choices=list(LOSSES_DICT.keys()),
                    
parser.add_argument("--weights", type=str,
                    help="Pre-trained model weights. This will resume training.")
parser.add_argument("--validate-every", "-ve", default=4, type=int,
                    help="Validate every X epochs (default %(default)d)")
parser.add_argument("--epochs", "-E", default=40, type=int)
parser.add_argument("--batch-size", "-B", default=1, type=int)
parser.add_argument("--lr", "-lr", default=2e-5, type=float)

DRIVE_SUBSET_TRAIN = slice(0, 15)
DRIVE_SUBSET_VAL = slice(15, 20)

def datasets():
    train_transforms, val_transforms = get_transforms()
    dict = {'train': DriveDataset('./data/training', transforms=train_transforms, green_only=True, train=True, subset=DRIVE_SUBSET_TRAIN),
            'validation': DriveDataset('./data/training', transforms=val_transforms, green_only=True, train=True, subset=DRIVE_SUBSET_VAL),
            'test':DriveDataset('./data/test', transforms=val_transforms, green_only=True, train=False)}
    return dict

def main(args,seed):

    dataset  = datasets()
    train_dataset, val_dataset, test_dataset = dataset['train'], dataset['validation'], dataset['test']

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

    model = UNet()
    tb_logger = pl_loggers.TensorBoardLogger('logs_acm_dice/')
    trainer = Trainer(
        #gpus=[0],
        num_nodes=1,
        #accelerator='ddp',
        #plugins=DDPPlugin(find_unused_parameters=False),
        logger = tb_logger,
        progress_bar_refresh_rate=1,
        max_epochs=args.epochs,
        benchmark=True,
        check_val_every_n_epoch=args.validate_every
    )

    trainer.fit(model,train_loader,val_loader)
    trainer.test(test_loader)

if __name__ == '__main__':

    args =parser.parse_args()

    main(args,seed=42)