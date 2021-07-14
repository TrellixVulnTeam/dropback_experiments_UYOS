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
from ray.tune import CLIReporter, JupyterNotebookReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback


# +
class ExperimentModel(pl.LightningModule):

    def __init__(
        self,
        arch: str = "mobilenet_v2", 
        num_classes: int = 10, 
        config = None,
        pre_trained: bool = False,
    ):
        super(ExperimentModel, self).__init__()

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

            self.model = models.mobilenet_v2(pretrained=self.pre_trained, 
                                             num_classes=self.num_classes, 
                                             inverted_residual_setting=cfg)
        
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
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 300], gamma=0.1)
#         scheduler = lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='min', factor=0.1, patience=20, threshold=1e-1, threshold_mode='abs', 
#             min_lr=0.001, verbose=True)
        
#         return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "ptl/val_loss"}
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
        # SDG
#         self.log("current_lr", self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[0])
        optimizer = self.trainer.optimizers[0]
        self.log("current_lr", optimizer.param_groups[0]["lr"])
        
    def training_epoch_end(self,outputs):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)   

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
    batch_size=128,
    num_workers=8,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)

import torchvision.datasets as datasets

trainset = datasets.CIFAR10(root='~/data', train=True,
                                        download=False, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=16)

testset = datasets.CIFAR10(root='~/data', train=False,
                                       download=False, transform=test_transforms)
val_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=16)


# +
def training(config, num_epochs=10, num_gpus=0):

#     seed_everything(42, workers=True)
    
    model = ExperimentModel(config=config)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
#         deterministic=True,
        callbacks=
        [
            ModelCheckpoint(
                monitor='ptl/val_loss',
                filename='epoch{epoch:02d}-top1_accuracy{val_accuracy_top1:.2f}',
                save_top_k=3,
                mode='min',
            ),
            TuneReportCallback(
                metrics = {
                    "loss": "ptl/val_loss",
                    "mean_accuracy": "ptl/val_accuracy_top1",
                    "current_lr": "current_lr",
                },
                on="validation_end")
        ]
    )
    
    trainer.fit(
        model, 
#         datamodule=cifar10_dm,
        train_dataloader=train_loader, 
        val_dataloaders=val_loader,
    ) 


def tune_asha(
    num_samples=10, 
    num_epochs=10, 
    gpus_per_trial=0,
    ):
    config = {
#         "lr": tune.loguniform(1e-4, 1e-1),
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 4e-5,
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=50,
        reduction_factor=2)

    reporter = JupyterNotebookReporter(
        overwrite=False,
        parameter_columns=["lr", "momentum"],
        metric_columns=["loss", "mean_accuracy", "training_iteration", "current_lr"]
    )
#     reporter = CLIReporter(
#         parameter_columns=["lr", "momentum"],
#         metric_columns=["loss", "mean_accuracy", "training_iteration", "current_lr"])

    analysis = tune.run(
        tune.with_parameters(
            training,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial,
        ),
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
        name="baseline")

    print("Best hyperparameters found were: ", analysis.best_config)
# -



# +
def training_w_checkpoint(config, checkpoint_dir = None, num_epochs=10, num_gpus=0):
    
    seed_everything(42, workers=True)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        deterministic=True,
        callbacks=
        [
            ModelCheckpoint(
                monitor='ptl/val_loss',
                filename='epoch{epoch:02d}-top1_accuracy{ptl/val_accuracy_top1:.2f}',
                save_top_k=3,
                mode='min',
            ),
            TuneReportCheckpointCallback(
                metrics = {
                    "loss": "ptl/val_loss",
                    "mean_accuracy": "ptl/val_accuracy_top1",
                    "current_lr": "current_lr",
                },
                filename="checkpoint",
                on="validation_end")
        ]
    )
    
    if checkpoint_dir:
        # Currently, this leads to errors:
        # model = LightningMNISTClassifier.load_from_checkpoint(
        #     os.path.join(checkpoint, "checkpoint"))
        # Workaround:
        ckpt = pl_load(
            os.path.join(checkpoint_dir, "checkpoint"),
            map_location=lambda storage, loc: storage)
        model = ExperimentModel._load_model_state(
            ckpt, config=config)
        trainer.current_epoch = ckpt["epoch"]
    else:
        model = ExperimentModel(config=config)
        
    trainer.fit(model, datamodule=cifar10_dm) 


def tune_pbt(
    num_samples=10, 
    num_epochs=10, 
    gpus_per_trial=0, 
    ):
    config = {
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 4e-5,
    }
    
    scheduler = PopulationBasedTraining(
        perturbation_interval=4,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-4, 1e-1),
        }
    )

    reporter = JupyterNotebookReporter(
        overwrite=False,
        parameter_columns=["lr", "momentum"],
        metric_columns=["loss", "mean_accuracy", "training_iteration", "current_lr"]
    )
#     reporter = CLIReporter(
#         parameter_columns=["lr", "momentum"],
#         metric_columns=["loss", "mean_accuracy", "training_iteration", "current_lr"])

    analysis = tune.run(
        tune.with_parameters(
            training_w_checkpoint,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial,
        ),
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
        name="baseline_pbt")

    print("Best hyperparameters found were: ", analysis.best_config)


# +
def fine_tuning_test():
    
    num_epochs=10
    num_gpus=1
#     seed_everything(42, workers=True)
    
    model = ExperimentModel()

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=[6],
        logger=TensorBoardLogger(
            save_dir="./log"),
        progress_bar_refresh_rate=30,
#         deterministic=True,
        callbacks=[
            ModelCheckpoint(
                monitor='ptl/val_loss',
                filename='{epoch:02d}-{val_accuracy_top1:.2f}',
                save_top_k=3,
                mode='min',
            ),
        ]
    )
    
    from pprint import pprint
    pprint(trainer.__dict__)
    
#     trainer.fit(model, datamodule=cifar10_dm) 

fine_tuning_test()
# -

if __name__== "__main__":
    tune_asha(num_samples=1, num_epochs=350, gpus_per_trial=1)
#     tune_pbt(num_samples=4, num_epochs=350, gpus_per_trial=1)
#     fine_tuning_test()
