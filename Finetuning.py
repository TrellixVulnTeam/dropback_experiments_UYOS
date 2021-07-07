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

from pl_bolts.datamodules import CIFAR10DataModule, FashionMNISTDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from ray import tune
from ray.tune import CLIReporter, JupyterNotebookReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

# +
model = models.mobilenet_v2()

print(model)


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
        self.test_accuracy_top1 = torchmetrics.Accuracy(top_k=1)

        # self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200, 250], gamma=0.1)
        
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
        self.log("current_lr", self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[0])
        
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        pred = F.softmax(logits, dim = 1)
        pred_label = torch.argmax(pred, dim=1)
        accuracy = torch.eq(pred_label, y).sum().item() / (len(y)*1.0)
        
        self.log_dict({'test_loss': loss, 'test_acc': accuracy})
        
    def training_epoch_end(self,outputs):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                          f"required shape: {model_state_dict[k].shape}, "
                          f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)


# +
from torchvision.datasets import CIFAR100
import torchvision.transforms as transform_lib
from torch.utils.data import DataLoader, random_split


class cifar100_datamodule(pl.LightningDataModule):
    
    def __init__(self, 
                 data_dir:str = "~/data",
                 num_workers:int = 16,
                 batch_size: int = 256,
                 val_split = 5000,
                 seed: int = 42,
                 *args,
                 **kwargs,
                ):
        super().__init__(*args, **kwargs)
        self.dims = (3, 32, 32)
        self.DATASET = CIFAR100
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        self.num_samples = 60000 - val_split
        
    @property
    def num_classes(self):
        return 100
    
    def prepare_data(self):
        self.DATASET(root=self.data_dir, train=True, download=True, transform=transform_lib.ToTensor())
        self.DATASET(root=self.data_dir, train=False, download=True, transform=transform_lib.ToTensor())
        
    def train_dataloader(self):
        transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms

        dataset = self.DATASET(self.data_dir, train=True, download=False, transform=transforms)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):
        transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = self.DATASET(self.data_dir, train=True, download=False, transform=transforms)
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return loader

    def test_dataloader(self):
        transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms

        dataset = self.DATASET(self.data_dir, train=False, download=False, transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader        

    def default_transforms(self):
        cifar100_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            cifar10_normalization()
        ])
        return cifar100_transforms
    


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
#     val_split=0,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)

import torchvision.datasets as datasets

testset = datasets.CIFAR10(root='~/data', train=False, download=False, transform=test_transforms)
val_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=16)

# -



# +
def fine_tuning(config, checkpoint_dir = None, num_epochs=10, num_gpus=0):
    
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
    
    ckpt_path = "/data/sunxd/ray_results/baseline/training_fcb72_00000_0_2021-07-05_14-35-58/checkpoints/epoch=152-step=24020.ckpt"
#     model = ExperiementModel.load_from_checkpoint(ckpt_path, config=config, num_classes=100)
    model = ExperiementModel.load_from_checkpoint(ckpt_path, config=config)
#     trainer.fit(model, datamodule=cifar10_dm) 
    trainer.test(model, datamodule=cifar10_dm)
    
#     cifar100_dm = cifar100_datamodule()
#     trainer.fit(model, datamodule=cifar100_dm) 
    
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
        grace_period=1,
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
            fine_tuning,
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
        name="finetune")

    print("Best hyperparameters found were: ", analysis.best_config)


# -

if __name__== "__main__":
    tune_asha(num_samples=1, num_epochs=350, gpus_per_trial=1)
