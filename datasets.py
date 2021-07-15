import logging
import os
import urllib.request
from urllib.error import HTTPError

from torch import Tensor
from torch.utils.data import Dataset

import pickle
import tarfile
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import Tensor

from PIL import Image

class CIFAR100():
    """
    Customized CIFAR100 dataset.
    Adapted from pl_bolts CIFAR10 implementation.
    Args:
        data_dir: Root directory of dataset
        train: If ``True``, creates dataset from training dataset, otherwise from 
        testing dataset.
        download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    data: Tensor
    targets: Tensor
    normalize: tuple
    dir_path: str
    BASE_URL = "https://www.cs.toronto.edu/~kriz/"
    FILE_NAME = 'cifar-100-python.tar.gz'
    cache_folder_name = 'complete'
    TRAIN_FILE_NAME = 'training.pt'
    TEST_FILE_NAME = 'test.pt'
    DATASET_NAME = 'CIFAR100'
    labels = set(range(100))
    relabel = False

    def __init__(
        self, 
        data_dir: str = '.', 
        train: bool = True, 
        transform: Optional[Callable] = None, 
        download: bool = True, 
        relabel: bool = False
    ):
        super().__init__()
        self.dir_path = data_dir
        self.train = train  # training set or test set
        self.transform = transform
        self.relabel = relabel
       
        os.makedirs(self.cached_folder_path, exist_ok=True)
        self.prepare_data(download)

        if not self._check_exists(self.cached_folder_path, (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME)):
            raise RuntimeError('Dataset not found.')

        data_file = self.TRAIN_FILE_NAME if self.train else self.TEST_FILE_NAME
        self.data, self.targets, self.targets_coarse = torch.load(os.path.join(self.cached_folder_path, data_file))

    def _download_from_url(self, base_url: str, data_folder: str, file_name: str):
        url = os.path.join(base_url, file_name)
        logging.info(f'Downloading {url}')
        fpath = os.path.join(data_folder, file_name)
        try:
            urllib.request.urlretrieve(url, fpath)
        except HTTPError as err:
            raise RuntimeError(f'Failed download from {url}') from err

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = self.data[idx].reshape(3, 32, 32)
        target = int(self.targets[idx])
        targets_coarse = int(self.targets_coarse[idx])

        if self.transform is not None:
            img = img.numpy().transpose((1, 2, 0))  # convert to HWC
            img = self.transform(Image.fromarray(img))

        if self.relabel:
            target = list(self.labels).index(target)

        # return img, target, targets_coarse
        return img, target

    @classmethod
    def _check_exists(cls, data_folder: str, file_names: Sequence[str]) -> bool:
        if isinstance(file_names, str):
            file_names = [file_names]
        return all(os.path.isfile(os.path.join(data_folder, fname)) for fname in file_names)

    def _unpickle(self, path_folder: str, file_name: str) -> Tuple[Tensor, Tensor]:
        with open(os.path.join(path_folder, file_name), 'rb') as fo:
            pkl = pickle.load(fo, encoding='bytes')
        return torch.tensor(pkl[b'data']), torch.tensor(pkl[b'fine_labels']), torch.tensor(pkl[b'coarse_labels'])

    def _extract_archive_save_torch(self, download_path):
        # extract achieve
        with tarfile.open(os.path.join(download_path, self.FILE_NAME), 'r:gz') as tar:
            tar.extractall(path=download_path)
        # this is internal path in the archive
        path_content = os.path.join(download_path, 'cifar-100-python')

        # load Test and save as PT
        torch.save(
            self._unpickle(path_content, 'test'), os.path.join(self.cached_folder_path, self.TEST_FILE_NAME)
        )
        # load Train and save as PT
        torch.save(
            self._unpickle(path_content, 'train'), os.path.join(self.cached_folder_path, self.TRAIN_FILE_NAME))

    def prepare_data(self, download: bool):
        if self._check_exists(self.cached_folder_path, (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME)):
            return

        base_path = os.path.join(self.dir_path, self.DATASET_NAME)
        if download:
            self.download(base_path)
        self._extract_archive_save_torch(base_path)

    def download(self, data_folder: str) -> None:
        """Download the data if it doesn't exist in cached_folder_path already."""
        if self._check_exists(data_folder, self.FILE_NAME):
            return
        self._download_from_url(self.BASE_URL, data_folder, self.FILE_NAME)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def cached_folder_path(self) -> str:
        return os.path.join(self.dir_path, self.DATASET_NAME, self.cache_folder_name)

    @staticmethod
    def _prepare_subset(
        full_data: Tensor,
        full_targets: Tensor,
        full_targets_coarse: Tensor,
        labels: Sequence,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Prepare a subset of a common dataset."""
        classes = {d: 0 for d in labels}
        indexes = []
        for idx, target in enumerate(full_targets):
            label = target.item()
            if classes.get(label, float('inf')) >= 5000:
                continue
            indexes.append(idx)
            classes[label] += 1
            if all(classes[k] >= 5000 for k in classes):
                break
        data = full_data[indexes]
        targets = full_targets[indexes]
        targets_coarse = full_targets_coarse[indexes]
        return data, targets, targets_coarse

class TrialCifar100(CIFAR100):
    """
    Create a subset of CIFAR100 given a list a labels.
    """

    def __init__(
        self,
        data_dir: str = '.',
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
        labels: Optional[Sequence] = (1, 5, 8),
        relabel:bool = False
    ):
        self.labels = labels if labels else list(range(100))

        self.cache_folder_name = f'labels-{"-".join(str(d) for d in sorted(self.labels))}'

        super().__init__(data_dir=data_dir, train=train, transform=transform, download=download, relabel=relabel)

    def prepare_data(self, download: bool) -> None:
        super().prepare_data(download)
                
        for fname in (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME):
            path_fname = os.path.join(os.path.join(self.dir_path, self.DATASET_NAME, self.cache_folder_name), fname)
            assert os.path.isfile(path_fname), 'Missing cached file: %s' % path_fname
            data, targets, targets_coarse = torch.load(path_fname)
            if len(self.labels) < 100:
                data, targets, targets_coarse = self._prepare_subset(data, targets, targets_coarse, self.labels)
            torch.save((data, targets, targets_coarse), os.path.join(self.cached_folder_path, fname))