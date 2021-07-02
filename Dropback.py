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
import math
from pytorch_lightning.core.datamodule import LightningDataModule
from pathlib import Path

import numpy as np
import torch, torchvision
import torchmetrics
import pytorch_lightning as pl
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets.utils import download_and_extract_archive
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

# +
model = models.mobilenet_v2(num_classes=10)

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

print((1-0.1)**26)
print(num_parameters * ((1-0.5)**4))


# -

class Dropback(torch.optim.SGD):
    '''
    Dropback only support SGD and SGD with momentum
    Does not currently support Nesterov
    '''
    def __init__(self, params, lr, track_size=0, init_decay=1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, named_params=[]):
        
        super(Dropback, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                                       weight_decay=weight_decay, nesterov=nesterov)
        # TODO: check if input values are valid

        self.named_params = named_params
        self.dump_path= './'
        self.dump_inited= False
        self.dump_flag=False

        for group in self.param_groups:
            init_params = []
            for p in group['params']:
                init_params.append(p.clone().detach())
            group['init_params'] = init_params
            group['track_size'] = track_size
            group['first_iter'] = True
            group['init_decay'] = init_decay

        # save init weights to check?

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            is_init_decay = group['init_decay'] < 1
            # decay init weights
            if not group['first_iter'] and is_init_decay:
                for init_p in group['init_params']:
                    init_p *= group['init_decay']

            if group['first_iter']:
                group['first_iter'] = False

        super(Dropback, self).step(closure)
        # think and make sure it is a way that can be done in HW
        # evaluate and sort accumulated gradients (as an metric of importance)
        # mask off the non important weights back to initial weights
        for group in self.param_groups:
            abs_accumulated_all = []  # absolute value of accumulated gradients of the entire network
            for p, init_p in zip(group['params'], group['init_params']):
                if p.grad is None:
                    continue
                abs_accumulated_all.append(torch.abs(p - init_p).flatten().clone().detach())
            abs_accumulated_flatten = torch.cat(abs_accumulated_all)
            _, ind = torch.topk(abs_accumulated_flatten, group['track_size'])
            # create a mask that selects topk values
            flattened_mask = torch.zeros_like(abs_accumulated_flatten, dtype=torch.bool)
            flattened_mask.scatter_(0, ind, 1.)

            start = 0
            layer_id=0
            total_non_zero=0
            total_size= sum([param.nelement() for param in group['params']])
            for p, init_p, (n_p,p_p) in zip(group['params'], group['init_params'], self.named_params):
                if p.grad is None:
                    continue
                end = start + p.data.numel()
                mask = flattened_mask[start:end].view(p.size())
                p.data[~mask] = init_p.data[~mask]
                start = end
                if self.dump_flag:
                    mask_to_dump = np.array(mask.cpu().detach())
                    mask_nonZero = np.sum(mask_to_dump)
                    total_non_zero += mask_nonZero
                    mask_sparsity = mask_nonZero / mask_to_dump.size # how many nonZero elements
                    mask_portion = mask_to_dump.size / total_size # What portion of total weights are these layer's Ws
                    layer_id_name=str(layer_id)+n_p.replace('module','').replace('.','_')
                    #print(layer_id_name, '\t,#nz', mask_nonZero)
                    self.dump_array(mask_to_dump ,layer_id_name)
                    self.dump_summary_sparsity(mask_sparsity , mask_portion, layer_id_name)

                layer_id = layer_id + 1
            #print('Total Nonzeros,', total_non_zero)
    def dump_summary_sparsity(self, sp=0 , mp=0, layer_name='x_def'):
            f = open(self.dump_path+'_summary_sparsity.txt', 'a+')
            f.write(layer_name+', '+ str(sp)+', '+ str(mp)+'\n')

    def dump_array(self, w_arr, layer_name='x_def'):
        if w_arr is None:
            print(layer_name, 'of layer', layer_name, 'is None!')
        else:
            np.save(self.dump_path+'_'+layer_name+'_W_mask', w_arr)
    def dump_init(self, dump_path):
        if not self.dump_inited:
            self.dump_path = dump_path
            self.dump_inited = True
            print("Weights masks are under:", dump_path)
            f = open(self.dump_path+'_summary_sparsity.txt', 'w+')
            f.write('layer_name, nz_portion, w_portion\n')
    def enable_dumping(self):
        self.dump_flag = True
    def disable_dumping(self):
        self.dump_flag = False


# +
def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
#                 num_zeros += torch.sum(param == 0).item()
                num_zeros += torch.sum(torch.abs(param) < 0.1).item()

                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
#                 num_zeros += torch.sum(param == 0).item()
                num_zeros += torch.sum(torch.abs(param) < 0.1).item()
                num_elements += param.nelement()
                
    if num_elements == 0:
        return 0,0,0

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def measure_global_sparsity(model,
                            weight=True,
                            bias=False,
                            use_mask=True):

    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():
        module_num_zeros, module_num_elements, _ = measure_module_sparsity(
            module, weight=weight, bias=bias, use_mask=use_mask)
        num_zeros += module_num_zeros
        num_elements += module_num_elements

    if num_elements == 0:
        return 0,0,0
    
    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


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
                "track_size": 500000,
                "init_decay": 0.9,
            }

        self.lr = config["lr"]
        self.momentum = config["momentum"]
        self.weight_decay = config["weight_decay"]
        self.track_size = config["track_size"]
        self.init_decay = config["init_decay"]

        self.arch = arch
        self.num_classes = num_classes
        self.pre_trained = pre_trained
        
        
        cfg = [(1,  16, 1, 1),
               (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
               (6,  32, 3, 2),
               (6,  64, 4, 2),
               (6,  96, 3, 1),
               (6, 160, 3, 2),
               (6, 320, 1, 1)]

        # inverted_residual_setting is mobilenet_v2 specific
        self.model = models.__dict__[self.arch](pretrained=self.pre_trained, num_classes=self.num_classes, inverted_residual_setting=cfg)

        self.train_accuracy_top1 = torchmetrics.Accuracy(top_k=1)
        self.train_accuracy_top5 = torchmetrics.Accuracy(top_k=5)
        self.val_accuracy_top1 = torchmetrics.Accuracy(top_k=1)
        self.val_accuracy_top5 = torchmetrics.Accuracy(top_k=5)

        # self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    
    def configure_optimizers(self):
        optimizer = Dropback(self.parameters(), 
                             lr=self.lr, 
                             momentum=self.momentum, 
                             weight_decay=self.weight_decay, 
                             track_size = self.track_size, 
                             init_decay = self.init_decay)
#         scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1**(epoch // 30))
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
        self.log("lr", self.lr)
        
        return loss
    
    def training_epoch_end(self, outputs):
        num_zeros, num_elements, sparsity = measure_global_sparsity(self.model, weight=True, bias=False, use_mask=False)
        self.log("num_zeros", num_zeros)
        self.log("num_elements", num_elements)
        self.log("sparsity", sparsity)
        
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)

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
    data_dir='~/data',
    batch_size=256,
    num_workers=8,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)


# -

def train_debug(config, num_epochs=10, num_gpus=0):
    # data_dir = os.path.expanduser("./data")
    model = ExperiementModel(config=config)
    model.datamodule = cifar10_dm
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(save_dir="./log", name="", version="."),
        progress_bar_refresh_rate=0)
    trainer.fit(model)


def train_tune(config, num_epochs=10, num_gpus=0):
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
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 4e-5,
        # "batch_size": tune.choice([32, 64, 128]),
        "track_size": 140000,
        "init_decay": 0.1,
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["lr"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_tune,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="dropback")

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__== "__main__":
    tune_asha(num_samples=4, num_epochs=300, gpus_per_trial=1)
#     config = {
# #         "lr": tune.loguniform(1e-4, 1e-1),
#         "lr": 0.01,
#         "momentum": 0.99,
#         "weight_decay": 4e-5,
#         # "batch_size": tune.choice([32, 64, 128]),
#         "track_size": 500000,
#         "init_decay": 0.1,
#     }
#     train_debug(config=config)
