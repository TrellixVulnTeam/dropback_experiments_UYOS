import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

import torchvision.models as models

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

import torchmetrics

from utils import measure_global_sparsity

from Dropback import Dropback

class ExperimentModel(pl.LightningModule):

    def __init__(
        self,
        arch: str = "mobilenet_v2", 
        num_classes: int = 10, 
        config = {
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 4e-5,
        },
        pre_trained: bool = False,
        experiment: str = "baseline"
    ):
        super(ExperimentModel, self).__init__()

        self.lr = config["lr"]
        self.momentum = config["momentum"]
        self.weight_decay = config["weight_decay"]
        
        self.arch = arch
        self.num_classes = num_classes
        self.pre_trained = pre_trained
        self.experiment = experiment
        
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

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        
        use_ReduceLROnPlateau = False
        if use_ReduceLROnPlateau:
            scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=20, threshold=1e-1, threshold_mode='abs', 
            min_lr=0.00001, verbose=True)
        
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "ptl/val_loss"}
        
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=0.1)
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
        self.log("current_lr", self.trainer.optimizers[0].param_groups[0]["lr"])

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        pred = F.softmax(logits, dim = 1)
        pred_label = torch.argmax(pred, dim=1)
        accuracy = torch.eq(pred_label, y).sum().item() / (len(y)*1.0)
        
        self.log_dict({'test_loss': loss, 'test_acc': accuracy})

    def training_epoch_end(self,outputs):
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params,self.current_epoch)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # To avoid size mismatch when loading the checkpoint
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    rank_zero_info(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                rank_zero_info(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

class DBModel(ExperimentModel):
    
    def __init__(
        self,
        arch: str = "mobilenet_v2", 
        num_classes: int = 10, 
        config = {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 4e-5,
            "track_size": 111835,
            "init_decay": 0.1,
        },
        pre_trained: bool = False,
    ):
        super().__init__(arch=arch, num_classes=num_classes, config=config, pre_trained=pre_trained)

        self.track_size = config["track_size"]
        self.init_decay = config["init_decay"]

    def configure_optimizers(self):
        optimizer = Dropback(
            self.parameters(), 
            lr=self.lr, 
            momentum=self.momentum, 
            weight_decay=self.weight_decay, 
            track_size = self.track_size, 
            init_decay = self.init_decay
        )
        
        use_ReduceLROnPlateau = False
        if use_ReduceLROnPlateau:
            scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=20, threshold=1e-1, threshold_mode='abs', 
            min_lr=0.001, verbose=True)
        
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "ptl/val_loss"}
        
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
        return [optimizer], [scheduler]
    
    def training_epoch_end(self,outputs):
        num_zeros, num_elements, sparsity = measure_global_sparsity(self.model, threshold=1e-3, weight=True, bias=True, use_mask=False)
        self.log("num_zeros", num_zeros)
        self.log("num_elements", num_elements)
        self.log("sparsity", sparsity)
        
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

class PruneModel(ExperimentModel):
    def __init__(
        self,
        arch: str = "mobilenet_v2", 
        num_classes: int = 10, 
        config = {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 4e-5,
        },
        pre_trained: bool = False,
    ):
        super().__init__(arch=arch, num_classes=num_classes, config=config, pre_trained=pre_trained)

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )

        optimizer = optim.SGD(trainable_parameters, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        
        use_ReduceLROnPlateau = False
        if use_ReduceLROnPlateau:
            scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=20, threshold=1e-1, threshold_mode='abs', 
            min_lr=0.001, verbose=True)
        
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "ptl/val_loss"}
        
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
        return [optimizer], [scheduler]

    def training_epoch_end(self,outputs):

        num_zeros, num_elements, sparsity = measure_global_sparsity(self.model, threshold=0, weight=True, bias=True, use_mask=True)
        self.log("num_zeros", num_zeros)
        self.log("num_elements", num_elements)
        self.log("sparsity", sparsity)

        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                self.logger.experiment.add_histogram(name, module.weight, self.current_epoch)
            if hasattr(module, 'bias') and module.bias is not None:
                self.logger.experiment.add_histogram(name, module.bias, self.current_epoch)

