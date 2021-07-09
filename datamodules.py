import torch
from torch.functional import split
from torch.utils.data import DataLoader, random_split

import torchvision
from torchvision.datasets import CIFAR10, CIFAR100

import pytorch_lightning as pl

from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


class cifar_datamodule(pl.LightningDataModule):
    
    def __init__(
        self,
        dataset: str = "cifar10",
        data_dir:str = "~/data",
        num_workers:int = 16,
        batch_size: int = 256,
        shuffle: bool = False,
        split: bool = False,
        val_split = 5000,
        seed: int = 42,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args,
        **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.dims = (3, 32, 32)

        if dataset == "cifar10":
            self.DATASET = CIFAR10
        elif dataset == "cifar100":
            self.DATASET = CIFAR100
        else:
            raise ValueError('Only \'cifar10\' and \'cifar100\' are supported.')
        
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        self.shuffle = shuffle
        self.split = split
        self.num_samples = 60000 - val_split
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        
    @property
    def num_classes(self):
        return 100
    
    def prepare_data(self):
        self.DATASET(root=self.data_dir, train=True, download=True, transform=torchvision.transforms.ToTensor())
        self.DATASET(root=self.data_dir, train=False, download=True, transform=torchvision.transforms.ToTensor())
        
    def train_dataloader(self):
        transforms = self.default_transforms()[0] if self.train_transforms is None else self.train_transforms

        dataset = self.DATASET(self.data_dir, train=True, download=False, transform=transforms)
        if self.split:
            train_length = len(dataset)
            dataset_train, _ = random_split(
                dataset,
                [train_length - self.val_split, self.val_split],
                generator=torch.Generator().manual_seed(self.seed)
            )
        else:
            dataset_train = dataset
        
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def val_dataloader(self):
        transforms = self.default_transforms()[1] if self.val_transforms is None else self.val_transforms

        dataset = self.DATASET(self.data_dir, train=True, download=False, transform=transforms)

        if self.split:
            train_length = len(dataset)
            _, dataset_val = random_split(
                dataset,
                [train_length - self.val_split, self.val_split],
                generator=torch.Generator().manual_seed(self.seed)
            )
        else:
            dataset_val = dataset
        
        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )
        return loader

    def test_dataloader(self):
        transforms = self.default_transforms()[1] if self.test_transforms is None else self.test_transforms

        dataset = self.DATASET(self.data_dir, train=False, download=False, transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader        

    def default_transforms(self):
        default_train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ])

        default_val_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ])
        return [default_train_transforms, default_val_transform]
    