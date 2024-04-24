from dataclasses import dataclass
from typing import Optional

import lazy_dataset
from distillation.data.linked_json import LinkedJsonDatabase
from nt_paths import db_root
from streaming_dataset.dataset import PickleDatabase


class Esc50PickleDatabase(PickleDatabase):
    def get_dataset(self, dataset_name):
        ds = self.prepare_data(dataset_name)
        return ds

    def prepare_data(self, dataset_name, prefetch_workers=0):
        ds = self.db.get_dataset(dataset_name)
        ds = ds.map(self.reader, num_workers=self.num_workers)
        ds = ds.map(self.fix_audio)
        ds = ds.unbatch()
        ds = list(ds)
        return ds


@dataclass
class Esc50Dataset:
    """Small dataset, is always cached."""
    root_path: str = db_root + "/esc50_32khz_wav_sd"
    validation_fold: int = 0
    folds: tuple = (0,1,2,3,4)

    def __post_init__(self):
        self.db = Esc50PickleDatabase(
            json_path=self.root_path + "/database.json", root_dir=self.root_path
        )
    
    def _fold_to_name(self, fold_idx):
        # folds: ["fold01", "fold02", "fold03", "fold04", "fold05"]
        return f"fold0{fold_idx}"
    
    def get_dataset_names(self):
        return [self._fold_to_name(i) for i in self.folds]

    def get_dataset(self, fold):
        ds_name = self._fold_to_name(fold)
        ds = self.db.get_dataset(
            ds_name,
        )
        return ds

    def get_train_set(self, balance=False):
        train_folds = [f for f in self.folds if f != self.validation_fold]
        ds = []
        for train_fold in train_folds:
            fold = self.get_dataset(train_fold)
            ds.extend(fold)

        ds = lazy_dataset.from_list(list(ds), immutable_warranty="wu")
        return ds

    def get_validate_set(self):
        ds = self.get_dataset(self.validation_fold)
        ds = lazy_dataset.from_list(list(ds), immutable_warranty="wu")
        return ds


@dataclass
class Esc50LogitsDataset(Esc50Dataset):
    logit_file_esc50: Optional[str] = (
        db_root + "/logits/esc50_ensemble_logits_full/database.json"
    )

    def __post_init__(self):
        super().__post_init__()
        self.logit_db = LinkedJsonDatabase(self.logit_file_esc50)

    def get_dataset(self, fold):
        ds = super().get_dataset(fold)
        data = []
        fold_name = f"fold0{fold}"
        logit_fold_ds = self.logit_db.get_dataset(fold_name)
        for ex in ds:
            ex["logits"] = logit_fold_ds[ex["example_id"]]["logits"]
            data.append(ex)
        return data