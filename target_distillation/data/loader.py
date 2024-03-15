from dataclasses import dataclass
import dataclasses

import numpy as np
from torch.utils.data import DataLoader as TorchLoader
from torch.utils.data import RandomSampler

from target_distillation.data.utils import Augmentation, Mixup, worker_init_fn


@dataclass
class LogitsDataloader:
    """Loader: batch and split to workers, apply augmentation and mixing"""
    database: 'Dataset' = None
    num_workers: int = 12
    batch_size: int = 32
    seed: int = 0
    augmentation: Augmentation = None
    mixup: Mixup = None

    def __post_init__(self):
        self.rng = np.random.RandomState(self.seed)

    def _get_train_set(self, balance=False):
        ds = self.database.get_train_set(balance=balance)

        if self.augmentation is not None:
            ds = ds.map(self.augmentation)

        if self.mixup is not None:
            mixup_ds = ds.copy()
            # set dataset-specific attributes
            new_mixup = dataclasses.replace(self.mixup)
            new_mixup.mixup_ds = mixup_ds
            new_mixup.rng = self.rng

            ds = ds.map(new_mixup)
        return ds

    @staticmethod
    def finalize_example(ex):
        return {
            "example_id": ex["example_id"],
            "weak_targets": np.asarray(ex["weak_targets"], dtype=np.float32),
            "audio_data": np.asarray(ex["audio_data"], dtype=np.float32),
            "logits": ex["logits"],
            "target_idx": np.argmax(ex["weak_targets"]).flatten().item(),
        }

    def get_train_set(self, balance=False, replacement=False, num_samples=None, generator=None):
        ds = self._get_train_set(balance=balance)

        ds = ds.map(LogitsDataloader.finalize_example)
        sampler = RandomSampler(ds,
                                replacement=replacement,
                                num_samples=num_samples,
                                generator=generator)
        dl = TorchLoader(
            dataset=ds,
            sampler=sampler,
            worker_init_fn=worker_init_fn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
        )
        return dl
    
    def get_validate_set(self):
        ds = self.database.get_validate_set()
        dl = TorchLoader(
            dataset=ds,
            # sampler=sampler,
            worker_init_fn=worker_init_fn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=False,
            persistent_workers=True,  # keep processes
        )
        return dl