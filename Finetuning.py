# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import sys
import math

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn.functional as F

import torchvision
import torchvision.models as models

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load

import torchmetrics

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback


# -

# Import model by load checkpoint

class ExperiementModel(pl.LightningModule):

    def __init__(
        self,
        arch: str = "mobilenet_v2", 
        num_classes: int = 10, 
        config = None,
        pre_trained: bool = False,
    ):
        super(ExperiementModel, self).__init__()
      
        if config == None:
            config = {
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 4e-5,
            }

        self.lr = config["lr"]
        self.momentum = config["momentum"]
        self.weight_decay = config["weight_decay"]

        self.arch = arch
        self.num_classes = num_classes
        self.pre_trained = pre_trained
           
        if arch == "mobilenet_v2":
            cfg = [(1,  16, 1, 1),
                   (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
                   (6,  32, 3, 2),
                   (6,  64, 4, 2),
                   (6,  96, 3, 1),
                   (6, 160, 3, 2),
                   (6, 320, 1, 1)]

            self.model = models.mobilenet_v2(pretrained=self.pre_trained, num_classes=self.num_classes, inverted_residual_setting=cfg)
        
        else:
            self.model = models.__dict__[self.arch](pretrained=self.pre_trained, num_classes=self.num_classes)

        self.train_accuracy_top1 = torchmetrics.Accuracy(top_k=1)
        self.train_accuracy_top5 = torchmetrics.Accuracy(top_k=5)
        self.val_accuracy_top1 = torchmetrics.Accuracy(top_k=1)
        self.val_accuracy_top5 = torchmetrics.Accuracy(top_k=5)

        # self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150,250], gamma=0.1)

        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        pred = F.softmax(logits, dim = 1)

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy_top1", self.train_accuracy_top1(pred, y))
        self.log("ptl/train_accuracy_top5", self.train_accuracy_top5(pred, y))
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        pred = F.softmax(logits, dim = 1)

        self.log("ptl/val_loss", loss)
        self.log("ptl/val_accuracy_top1", self.val_accuracy_top1(pred, y))
        self.log("ptl/val_accuracy_top5", self.val_accuracy_top5(pred, y))


    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)
