import os
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace

class PLModule(pl.LightningModule):

    def __init__(self, model, args, train_dataset, eval_dataset):
        super().__init__()
        self.model = model
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = model.configure_optimizers(args)

    def forward(self, x, y):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self(x, y)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True)
        return result
        
    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits, loss = self(x)
    #     result = pl.EvalResult(checkpoint_on=loss)
    #     result.log('val_loss', loss)
    #     return result

    def configure_optimizers(self):
        return self.optimizer

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    # def val_dataloader(self):
    #     val_loader = torch.utils.data.DataLoader(
    #         dataset=self.eval_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.workers,
    #     )
    #     return val_loader

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('-b', '--batch-size', default=64, type=int,
                            metavar='N',
                            help='mini-batch size (default: 64), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        return parser

class TrainerConfig:
    # optimization parameters
    seed = None
    distributed_backend= None
    gpus=None
    num_workers=None
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, args, train_dataset, eval_dataset):
        if args.seed is not None:
            pl.seed_everything(args.seed)

        if args.distributed_backend == 'ddp':
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / max(1, args.gpus))
            args.workers = int(args.workers / max(1, args.gpus))

        self.module = PLModule(model, args, train_dataset, eval_dataset)
        self.trainer = pl.Trainer.from_argparse_args(args)

    def train(self):
        self.trainer.fit(self.module)
