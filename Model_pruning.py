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
from pytorch_lightning.callbacks import ModelPruning

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load

import torchmetrics

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

from utils import measure_global_sparsity


# -
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
    
    def training_epoch_end(self,outputs):
        num_zeros, num_elements, sparsity = measure_global_sparsity(self.model, 
                                                                    threshold=0, 
                                                                    weight=True, 
                                                                    bias=False, 
                                                                    use_mask=True)
        self.log("num_zeros", num_zeros)
        self.log("num_elements", num_elements)
        self.log("sparsity", sparsity)
        
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                self.logger.experiment.add_histogram(name, module.weight, self.current_epoch)

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


# +
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    cifar10_normalization(),
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    cifar10_normalization(),
])

cifar10_dm = CIFAR10DataModule(
    data_dir="~/data",
    batch_size=256,
    num_workers=8,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)


# -

def training(config, num_epochs=10, num_gpus=0):
    # data_dir = os.path.expanduser("./data")
    model = ExperiementModel(config=config)
    model.datamodule = cifar10_dm
            
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            ModelPruning(
                pruning_fn='l1_unstructured',
                parameter_names=["weight"],
                amount = lambda epoch: 0.1 if (epoch > 100 and epoch % 50 == 0) else 0,
                use_global_unstructured=True,
                verbose=2,
            ),
            TuneReportCallback(
                {
                    "loss": "ptl/val_loss",
                    "mean_accuracy": "ptl/val_accuracy_top1"
                },
                on="validation_end")
        ])
    trainer.fit(model)


def tune_asha(num_samples=10, num_epochs=10, gpus_per_trial=0):
    config = {
#         "lr": tune.loguniform(1e-4, 1e-1),
        "lr" : 0.01,
        "momentum": 0.9,
        "weight_decay": 4e-5,
        # "batch_size": tune.choice([32, 64, 128]),
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=50,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["lr"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            training,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": 2,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="pruning")

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__== "__main__":
    tune_asha(num_samples=4, num_epochs=800, gpus_per_trial=1)
