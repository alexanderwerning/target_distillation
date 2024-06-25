from dataclasses import dataclass
import dataclasses

import numpy as np
from torch.utils.data import DataLoader as TorchLoader, RandomSampler

from target_distillation.data.utils import Augmentation, Mixup
from torch.utils.data import IterableDataset
import lazy_dataset

class IterableDatasetWrapper(IterableDataset):
    def __init__(self, ds):
        self.ds = ds
    
    def __iter__(self):
        return iter(self.ds)


@dataclass
class LogitsDataloader:
    """Loader: batch and split to workers, apply augmentation and mixing"""
    database: 'Dataset' = None
    num_workers: int = 0
    batch_size: int = 32
    seed: int = 0
    augmentation: Augmentation = None
    mixup: Mixup = None
    train_drop_last: bool = False
    persistent_workers: bool = True

    def _get_train_set(self, balance=False):
        ds = self.database.get_train_set()

        if self.augmentation is not None:
            ds = ds.map(self.augmentation)

        if self.mixup is not None:
            mixup_ds = self.database.get_train_set()
            if self.augmentation is not None:
                ds = ds.map(self.augmentation)
            # set dataset-specific attributes
            new_mixup = dataclasses.replace(self.mixup)
            new_mixup.mixup_ds = mixup_ds

            ds = ds.map(new_mixup)
        return ds

    def get_train_set(self, balance=False, replacement=False, num_samples=None, generator=None):
        ds = self._get_train_set(balance=balance)
        # pytorch lightning will add distributed sampler here automatically
        # https://pytorch-lightning.readthedocs.io/en/0.9.0/multi_gpu.html#remove-samplers
        ds = IterableDatasetWrapper(ds)
        if num_samples is not None:
            sampler = RandomSampler(ds,
                                    replacement=replacement,
                                    num_samples=num_samples,
                                    generator=generator)
        else:
            sampler = None
            # fix ddp
            # ds = ds.repeat(2).set_length(self.database.num_train_samples/num_processes/num_workers)
        dl = TorchLoader(
            dataset=ds,
            sampler=sampler,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=self.train_drop_last,
            persistent_workers=self.num_workers>0 and self.persistent_workers
        )
        return dl
    
    def get_validate_set(self):
        ds = self.database.get_validate_set()
        ds = IterableDatasetWrapper(ds)
        dl = TorchLoader(
            dataset=ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=False,
            persistent_workers=self.num_workers>0 and self.persistent_workers,  # keep processes
        )
        return dl
    
    def get_dataset(self, dataset_name):
        ds = self.database.get_dataset(dataset_name)
        ds = IterableDatasetWrapper(ds)
        dl = TorchLoader(
            dataset=ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=False,
            persistent_workers=self.num_workers>0 and self.persistent_workers,  # keep processes
        )
        return dl