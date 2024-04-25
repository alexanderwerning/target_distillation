import math

from dataclasses import dataclass
from typing import List, Optional

import lazy_dataset
import numpy as np
import paderbox as pb
from distillation.data.linked_json import LinkedJsonDatabase
from nt_paths import db_root
from streaming_dataset.dataset import PickleDatabase
from pathlib import Path
import webdataset as wds

import torch

# tau2022_classes = [
#     "airport",
#     "shopping_mall",
#     "metro_station",
#     "street_pedestrian",
#     "public_square",
#     "street_traffic",
#     "tram",
#     "bus",
#     "metro",
#     "park"
# ]
tau2022_classes = [
    "airport",
    "bus",
    "metro",
    "metro_station",
    "park",
    "public_square",
    "shopping_mall",
    "street_pedestrian",
    "street_traffic",
    "tram"
]

from torch.utils.data import IterableDataset
import lazy_dataset

class IterableDatasetWrapper(IterableDataset):
    def __init__(self, ds):
        self.ds = ds
    
    def __iter__(self):
        return iter(self.ds)

class Tau2022PickleDatabase(PickleDatabase):
    # class_file: str = "events/tau2022_events.json"

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        # self.class_map = {c: i for i, c in enumerate(pb.io.load(self.class_file))}
        self.class_map = {c: i for i, c in enumerate(tau2022_classes)}

    def get_dataset(self, dataset_name):
        ds = self.prepare_data(dataset_name)
        return ds
    
    def add_target(self, examples):
        for ex in examples:
            # np array as hint for batching
            if not isinstance(ex['label'], str):
                print(ex)
            ex["target"] = np.asarray(self.class_map[ex["label"]])
        return examples

    # 6k examples per pickle file
    def prepare_data(self, dataset_name, shuffle=True, prefetch_workers=0):
        ds = self.db.get_dataset(dataset_name)
        ds = ds.shuffle(shuffle)
        ds = ds.map(self.reader, num_workers=self.num_workers)
        # ds = ds.map(self.fix_audio)
        ds = ds.map(self.add_target)
        ds = ds.unbatch()
        ds = ds.shuffle(shuffle, buffer_size=512)
        # ds = list(ds)
        return ds


@dataclass
class Tau2022Dataset:
    root_path: str = db_root + "/tau2022_32khz_sd"
    validation_set: str = "test"
    train_set: str = "train_100"
    cache: bool = False
    dataset_names: List[str] = ("train_5", "train_10", "train_25", "train_50", "train_100", "test")  # , "evaluate"

    def __post_init__(self):
        self.db = Tau2022PickleDatabase(
            json_path=self.root_path + "/database.json", root_dir=self.root_path
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


@dataclass
class Tau2022LogitsDataset(Tau2022Dataset):
    logit_file: Optional[str] = (
        db_root + "/logits/tau2022_ensemble_logits_full/database.json"
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

class Preprocessing:
    def __init__(self, class_mapping, num_classes, logit_dict, target_key="target"):
        self.class_mapping = class_mapping
        self.num_classes = num_classes
        self.logit_dict = logit_dict
        self.target_key = target_key
    
    def __call__(self, x):
        example_id, (audio_data, sample_rate), meta = x
        target = np.asarray(self.class_mapping[meta["label"]])
        
        logits = torch.as_tensor(self.logit_dict[example_id]['logits'])

        example = {
            "__key__": example_id,  # map adds __key__ anyways, avoid None value
            "example_id": example_id,
            "logits": logits,
            self.target_key: target,
            "audio_data": audio_data
        }
        return example


@dataclass
class Tau2022LogitsWdsDataset:
    root_path: str = db_root + "/wds/tau2022_32khz"
    logit_file: str = db_root + "/logits/tau2022_ensemble_logits_full/database.json"
    validation_set: str = "test"
    train_set: str = "train_100"
    cache: bool = False

    def __post_init__(self):
        self.num_classes = len(tau2022_classes)
        self.class_mapping = {cls: i for i, cls in enumerate(tau2022_classes)}
        self.inverse_class_mapping = dict(enumerate(tau2022_classes))

        self.db = LinkedJsonDatabase(self.logit_file).data["datasets"]

    def get_train_shards(self):
        train_path = Path(self.root_path) / self.train_set
        return [str(p) for p in train_path.glob("*.tar")]

    def get_validate_shards(self):
        eval_path = Path(self.root_path) / self.validation_set
        return [str(p) for p in eval_path.glob("*.tar")]

    def get_dataset(self, input_shards):
        logit_dict = {k: v for ds in self.db.values() for k, v in ds.items()}

        ds = (
            wds.WebDataset(input_shards, nodesplitter=wds.split_by_node)
            .shuffle(100)
            .decode(wds.torch_audio)
            .to_tuple("__key__", "flac", "json")
        )
        if self.cache:
            ds = ds.mcached()
        ds = ds.map(
            Preprocessing(
                class_mapping=self.class_mapping,
                num_classes=self.num_classes,
                logit_dict=logit_dict,
            )
        )

        return ds

    def get_train_set(self):
        return self.get_dataset(self.get_train_shards())

    def get_validate_set(self):
        return self.get_dataset(self.get_validate_shards())
