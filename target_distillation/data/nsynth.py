from dataclasses import dataclass
from typing import List, Optional

import lazy_dataset
import numpy as np
import paderbox as pb
from distillation.data.linked_json import LinkedJsonDatabase
from nt_paths import db_root
from pathlib import Path
import soundfile as sf

import torch


nsynth_classes = [
    i for i in range(21, 109)
]

@dataclass
class NsynthDatabase:
    json_path: str = db_root+"/nsynth_pitch-v2.2.3-5h/database.json"
    num_workers: int = 0


    def __post_init__(self, *args, **kwargs):
        self.class_map = {c: i for i, c in enumerate(nsynth_classes)}
        self.db = lazy_dataset.database.JsonDatabase(self.json_path)

    def get_dataset(self, dataset_name):
        ds = self.prepare_data(dataset_name)
        return ds
    
    def reader(self, example):
        audio, sr = sf.read(db_root+"/"+example["audio_path"], dtype='float32')
        audio = audio.reshape(1,-1)
        example['audio_data'] = audio
        example.pop("audio_path")
        return example

    def add_target(self, ex):
        # if not isinstance(ex['label'], str):
        #     print(ex)
        ex["target"] = np.asarray(self.class_map[ex["label"]])
        ex["weak_targets"] = np.zeros(len(self.class_map))
        ex["weak_targets"][ex['target']] = 1.0
        return ex

    def prepare_data(self, dataset_name, shuffle=True, prefetch_workers=0):
        ds = self.db.get_dataset(dataset_name)
        ds = ds.shuffle(shuffle)
        ds = ds.map(self.reader, num_workers=self.num_workers)
        ds = ds.map(self.add_target)
        return ds


@dataclass
class NsynthDataset:
    root_path: str = db_root + "/nsynth_pitch-v2.2.3-5h"
    # root_path: str = db_root + "/HEAR_benchmark/tasks/nsynth_pitch-v2.2.3-5h"
    validation_set: str = "valid"
    eval_set: str = "test"
    train_set: str = "train"
    cache: bool = True
    dataset_names: List[str] = ("train", "test", "valid")

    def __post_init__(self):
        self.db = NsynthDatabase(
            json_path=self.root_path + "/database.json"
        )
    
    def get_dataset_names(self):
        return self.dataset_names

    def get_dataset(self, ds_name):
        ds = self.db.get_dataset(
            ds_name,
        )
        if self.cache:
            ds = lazy_dataset.from_list(list(ds), immutable_warranty="wu")
        return ds

    def get_train_set(self, balance=False):
        ds = self.get_dataset(self.train_set)

        if self.cache:
            ds = lazy_dataset.from_list(list(ds), immutable_warranty="wu")
        return ds

    def get_validate_set(self):
        ds = self.get_dataset(self.validation_set)
        if self.cache:
            ds = lazy_dataset.from_list(list(ds), immutable_warranty="wu")
        return ds
    
    def get_eval_set(self):
        ds = self.get_dataset(self.eval_set)
        if self.cache:
            ds = lazy_dataset.from_list(list(ds), immutable_warranty="wu")
        return ds



@dataclass
class NsynthLogitsDataset(NsynthDataset):
    logit_file: Optional[str] = (
        db_root + "/logits/nsynth_ensemble_logits/database.json"
    )

    def __post_init__(self):
        super().__post_init__()
        self.logit_db = LinkedJsonDatabase(self.logit_file)

    def get_dataset(self, ds_name):
        ds = super().get_dataset(ds_name)
        logit_ds = self.logit_db.get_dataset(ds_name)
        def add_logits(ex):
            ex["logits"] = logit_ds[ex["example_id"]]["logits"]
            return ex
        ds = ds.map(add_logits)

        return ds