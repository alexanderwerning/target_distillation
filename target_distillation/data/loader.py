from distillation.data.linked_json import LinkedJsonDatabase
from nt_paths import db_root
from pb_sed.data_preparation.provider import DataProvider
from dataclasses import dataclass
from typing import Optional
from distillation.data.esc50_32khz_sd import Esc50Provider
from pathlib import Path
import numpy as np
import lazy_dataset
import h5py
import io
from EfficientAT.datasets.audioset import read_mp3, pad_or_truncate, pydub_augment
import torch
from torch.utils.data import (
    DataLoader as TorchLoader,
    RandomSampler,
    WeightedRandomSampler,
)
from distillation.data.loader import DataLoader
from EfficientAT.helpers.init import worker_init_fn
from distillation.data.mix_dataset import Augmentation
import soundfile as sf
from padertorch.contrib.je.data.transforms import Collate
from lazy_dataset.database import JsonDatabase
import pickle as pkl
import wids
import webdataset as wds
import paderbox as pb
from pathlib import Path
import random
import warnings

# @dataclass
# class AudioReader:
#     audioset_path: str = db_root + "/audioset_32khz_flac_hdf"
#     clip_length: float = 5.0
#     sample_rate: int = 32000
#     add_class_labels: bool = False

#     def __post_init__(self):
#         audioset_root = Path(self.audioset_path)
#         self.dset_files = [
#             audioset_root / "balanced_train.hdf",
#             audioset_root / "unbalanced_train.hdf",
#             audioset_root / "eval.hdf",
#         ]

#     def __call__(self, example):
#         example = example.copy()
#         dset_file_idx = example["hdf_file_idx"]
#         idx = example["hdf_idx"]

#         dset_file = self.dset_files[dset_file_idx]
#         assert "start" in example and "end" in example, example.keys()
#         start_s = example["start"]
#         start = int(start_s * self.sample_rate)
#         example.pop("start")
#         end_s = example["end"]
#         end = int(end_s * self.sample_rate)
#         example.pop("end")
#         for _ in range(5):
#             try:
#                 with h5py.File(dset_file, "r") as file:
#                     bytestream = file["wav"][idx]
#                 waveform = sf.read(io.BytesIO(bytestream.tobytes()))[0][start:end]
#                 break
#             except sf.LibsndfileError as e:
#                 with open("error_bytes.bytes", "wb") as f:
#                     f.write(bytestream.tobytes())
#                 with h5py.File(dset_file, "r") as file:
#                     print(file["audio_name"][idx].decode())
#                 print(f"dset_file_idx: {dset_file_idx}, example_idx: {idx}")
#                 print(e)
#         else:
#             raise RuntimeError(
#                 f"Failed to read audio file [{dset_file_idx}] {str(dset_file)}, example {idx}"
#             )
#         waveform = pad_or_truncate(waveform, int(self.clip_length * self.sample_rate))
#         example["audio_data"] = waveform
#         if self.add_class_labels:
#             example["labeled"] = True
#         else:
#             example["labeled"] = False
#         return example

class AudioReader:
    def __call__(self, example: dict):
        raise NotImplementedError()


@dataclass
class AudioReaderWids(AudioReader):
    database_path: str = db_root + "/wds/audioset"
    clip_length: float = 5.0
    sample_rate: int = 32000
    add_class_labels: bool = False

    def __post_init__(self):
        self.database_path = Path(self.database_path)
        # file index: example_id -> wds index
        file_index_path = self.database_path / "file_index.json"
        assert file_index_path.exists(), f"file index path does not exits: {file_index_path}"
        ds = JsonDatabase(json_path=self.database_path / "file_index.json")
        self.example_index = {id_:idx for v in ds.data.values() for id_, idx in v.items()}
        self.dset_names = [
            "balanced_train",
            "unbalanced_train",
            "eval",
        ]
        shards = []
        for dset_name in self.dset_names:
            shard_path = self.database_path/dset_name/"shardindex.json"
            if shard_path.exists():
                shardlist = pb.io.load(shard_path)['shardlist']
                for shardinfo in shardlist:
                    shardinfo['url'] = dset_name+"/"+shardinfo['url'] 
                shards.extend(shardlist)
            else:
                warnings.warn(f"shard path {shard_path} does not exist")

        self.ds = wids.ShardListDataset(shards)

    def __call__(self, example):
        index = self.example_index[example['example_id']]
        example = self.ds[index]
        example = example.copy()
        example = wds.decode(example, handler=wds.torch_audio)
        example_id, audio_data, meta = wds.to_tuple(example, "__key__", "flac", "json")

        if "start" in example and "stop" in example:
            raise NotImplementedError()
        # start_s = example["start"]
        # start = int(start_s * self.sample_rate)
        # example.pop("start")
        # end_s = example["end"]
        # end = int(end_s * self.sample_rate)
        # example.pop("end")
        # [start:end]

        audio_data = pad_or_truncate(audio_data, int(self.clip_length * self.sample_rate))
        example = {
            "audio_data": audio_data,
            "example_id": example_id,
            **meta
        }
        return example


@dataclass
class AudiosetLogitsDataset:
    """Dataset is filtered by examples in given json file."""

    # logits are included in this file
    json_path: str = db_root + "/audioset_logits_db/database.json"
    audio_reader: AudioReader = None
    balance_classes: bool = False
    # load_weak_targets_from_file: bool = True
    cache: bool = False

    def __post_init__(self):
        self.db = LinkedJsonDatabase(self.json_path)
        assert self.balance_classes is False, "Do not balance classes for now"

    def get_raw_dataset(self, name):
        ds = self.db.get_dataset(name)
        return ds

    @staticmethod
    def _build_repetition_groups(dataset, label_repetitions):
        # new datasets of examples with same repetitions
        example_reps_list = [
            max(
                [
                    l_rep
                    for i, l_rep in enumerate(label_repetitions)
                    if i in ex["event_ids"]
                ]
            )
            for ex in dataset
        ]
        example_reps = np.array(example_reps_list)
        rep_groups = {}
        # group by number of repetitions
        for n_reps in set(example_reps_list):
            rep_groups[n_reps] = np.argwhere(example_reps == n_reps).flatten().tolist()
        # new datasets of examples with same repetitions
        datasets_reps = []
        for n_reps, indices in sorted(rep_groups.items()):
            dataset_subset = [dataset[i] for i in sorted(indices)]
            datasets_reps.append(
                (
                    lazy_dataset.from_list(dataset_subset, immutable_warranty="wu"),
                    n_reps,
                )
            )
        return datasets_reps

    def _get_train_set(self, balance=True):
        unbalanced_train = self.db.get_dataset("unbalanced_train")
        balanced_train = self.db.get_dataset("balanced_train")
        dict_ds = {**unbalanced_train, **balanced_train}
        assert len(dict_ds) == len(unbalanced_train) + len(balanced_train)
        ds = []
        for ex_idx, ex in dict_ds.items():
            ex["example_id"] = ex_idx
            ds.append(ex)
        
        # if not self.load_weak_targets_from_file:
        #     for ex in ds:
        #         tgt = np.zeros(527)
        #         for event_id in ex["event_ids"]:
        #             tgt[event_id] = 1
        #         ex["weak_targets"] = tgt

        if self.balance_classes and balance:
            # count labels
            label_count_arr = np.zeros_like(ds[0]['weak_targets'])  # 527 classes
            print(f"Balancing {label_count_arr.shape[0]} classes")
            for ex in ds:
                label_count_arr += ex['weak_targets']
            if len(label_count_arr) == 50:
                label_count_arr = np.zeros(50)  # TODO: assumes if targets are given, they are only for esc50
            label_counts_dict = {
                i: label_count_arr[i] for i in range(len(label_count_arr))
            }
            label_reps_dict = DataProvider._compute_label_repetitions(
                label_counts_dict, min_counts=1 / 100
            )
            label_reps = np.zeros_like(label_count_arr)
            for i, lc in label_reps_dict.items():
                label_reps[i] = lc
            # # apply label count information to datasets with loaded audio
            repetition_groups = AudiosetLogitsDataset._build_repetition_groups(
                ds, label_reps
            )
            if self.cache:
                ds = ds.map(self.audio_reader)
                ds = ds.cache()
            ds = lazy_dataset.intersperse(
                *[ds.tile(reps) for ds, reps in repetition_groups]
            )
            if not self.cache:
                ds = ds.map(self.audio_reader)
        else:
            ds = lazy_dataset.from_list(ds, immutable_warranty="wu")
            ds = ds.map(self.audio_reader)
            if self.cache:
                ds = ds.cache()
        return ds

    def _get_validate_set(self):
        raise NotImplementedError()


# @dataclass
# class Esc50LogitsDataset:
#     logit_file_esc50: Optional[str] = (
#         db_root + "/logits/esc50_ensemble_logits_full/database.json"
#     )
#     esc50: Esc50Provider = None
#     add_labeled_flag: bool = True

#     def __post_init__(self):
#         self.logit_db = LinkedJsonDatabase(self.logit_file_esc50)

#     @property
#     def validation_fold(self):
#         return self.esc50.validation_fold

#     def _get_dataset_fold(self, fold):
#         data = []
#         ds = self.esc50.get_dataset_fold(fold, no_augment=True)
#         fold_name = f"fold0{fold}"
#         logit_fold_ds = self.logit_db.get_dataset(fold_name)
#         for ex in ds:
#             ex["db"] = f"esc50_fold0{fold}"
#             ex["logits"] = logit_fold_ds[ex["example_id"]]["logits"]
#             if self.add_labeled_flag:
#                 ex["labeled"] = True
#             data.append(ex)
#         return data

#     def _get_train_set(self):
#         train_folds = [f for f in range(5) if f != self.validation_fold]
#         data = []
#         for train_fold in train_folds:
#             data.extend(list(self._get_dataset_fold(train_fold)))
#         ds = lazy_dataset.from_list(data, immutable_warranty="wu")
#         return ds

#     def _get_validate_set(self):
#         ds = self._get_dataset_fold(self.validation_fold)
#         ds = lazy_dataset.from_list(list(ds), immutable_warranty="wu")
#         return ds

#     def get_validate_set(self):
#         raise NotImplementedError()

#     @classmethod
#     def finalize_dogmatic_config(cls, config):
#         config["esc50"] = {
#             "factory": Esc50Provider,
#             "root_path": db_root + "/esc50_32khz_wav_sd",
#             "validation_fold": 0,
#         }

def collate_element(element_list):
    b = element_list[0]
    if isinstance(b, (list, tuple)):
        return element_list
    elif isinstance(b, dict):
        return element_list
    elif isinstance(b, torch.Tensor):
        return torch.stack(element_list)
    elif isinstance(b, np.ndarray):
        return np.stack(element_list)
    else:
        return element_list

def collation_fn(batch_list):
    if isinstance(batch_list[0], dict):
        batch = {}
        for k in batch_list[0].keys():
            element_list = [e[k] for e in batch_list]
            batch[k] = collate_element(element_list)
    else:
        batch = []
        for i, b in enumerate(batch_list[0]):
            element_list = [e[i] for e in batch_list]
            batch.append(collate_element(element_list))
    return batch


class Preprocessing:
    def __init__(self, device, example_len, class_mapping, num_classes, logit_dict, target_key="weak_targets"):
        self.device = device
        self.example_len = example_len
        self.class_mapping = class_mapping
        self.num_classes = num_classes
        self.logit_dict = logit_dict
        self.target_key = target_key
    
    def __call__(self, x):
        key, (audio_data, sample_rate), meta = x
        # meta: {"text": caption, "original_data": {"class_labels": [str,...], "class_names": [str,...]}}
        class_names = meta["original_data"]["class_names"]
        class_indices = [self.class_mapping[c] for c in class_names]
        targets = torch.zeros(self.num_classes, device=self.device)
        for idx in class_indices:
            targets[idx] = 1
        example_id = key.rsplit("/", maxsplit=1)[-1]
        # if split audio
        # splits = []
        # keys = [f"{example_id}_0.0", f"{example_id}_5.0"]
        # for k in keys:
        #     if k in self.logit_dict:
        #         splits.append(k)
        # assert len(splits) > 0, key
        # if len(splits) == 2:
        #     logits = self.logit_dict[random.choice(splits)]['logits']
        # else:
        #     logits = self.logit_dict[splits[0]]['logits']
        logits = self.logit_dict[example_id]['logits']


        # fix this using bucketing
        if self.example_len is not None:
            audio_data_cut_padded = torch.zeros(1, self.example_len, device=self.device)
            if audio_data.shape[1] >= self.example_len:
                audio_data_cut_padded[:] = audio_data[:, :self.example_len]
            else:
                audio_data_cut_padded[:, :audio_data.shape[1]] = audio_data
        else:
            audio_data_cut_padded = audio_data
        audio_data_cut_padded = audio_data_cut_padded.to(self.device)
        # todo: also give sequence lengths -> only compute loss for non-padded signal

        example = {
            "logits": logits,
            self.target_key: targets,
            "audio_data": audio_data_cut_padded
        }
        return example


@dataclass
class AudiosetLogitsWdsDataset:
    dataset_path: str = db_root + "/wds/audioset"
    json_path: str = db_root + "/audioset_logits_db/database.json"
    balance_classes: bool = False
    load_weak_targets_from_file: bool = True
    cache: bool = False
    num_workers: int = 4
    batch_size: int = 32
    sample_rate: int = 16_000

    device: int = 0
    clip_length: float = 5.0
    num_classes: int = 527
    
    def __post_init__(self):
        assert self.balance_classes is False, "cannot class balance"
        classes = pb.io.load("events/audioset_events.json")
        self.num_classes = len(classes)
        self.class_mapping = {cls:i for i, cls in enumerate(classes)}
        self.inverse_class_mapping = dict(enumerate(classes))

        self.db = LinkedJsonDatabase(self.json_path)
        self.example_len = int(self.clip_length * self.sample_rate)

    def get_train_shards(self):
        balanced_train_path = Path(self.dataset_path)/"balanced_train"
        unbalanced_train_path = Path(self.dataset_path)/"unbalanced_train"
        return [str(p) for p in balanced_train_path.glob("*.tar")] + [str(p) for p in unbalanced_train_path.glob("*.tar")]
    
    def get_validate_shards(self):
        eval_path = self.dataset_path/"eval"
        return [str(p) for p in eval_path.glob("*.tar")]

    def get_dataset(self, input_shards):
        logit_dict = {k: v for ds in self.db.data['datasets'].values() for k, v in ds.items()}
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(input_shards),
            wds.shuffle(100), # shuffle shard iterator
            wds.split_by_node,
            wds.split_by_worker, # split for each gpu? -> splits for rank and local worker process
            wds.tarfile_to_samples(), # iterate over examples
            wds.shuffle(1000), # shuffle examples within a shard in-memory
            wds.decode(wds.torch_audio),
            wds.to_tuple("__key__", "flac", "json"),
            wds.map(Preprocessing(device=self.device,
                                  example_len=self.example_len,
                                  class_mapping=self.class_mapping,
                                  num_classes=self.num_classes,
                                  logit_dict=logit_dict)),
            wds.batched(self.batch_size, collation_fn=collation_fn)
        )
        dataloader = wds.WebLoader(pipeline, num_workers=self.num_workers, batch_size=None)
        return dataloader
    
    def get_dataset_debug(self, input_shards, batch_size=2):
        ds = wds.WebDataset(input_shards, shardshuffle=False).decode(wds.torch_audio)
        ds = ds.to_tuple("flac", "json").map(Preprocessing(device="cpu",
                                  example_len=self.example_len,
                                  class_mapping=self.class_mapping,
                                  num_classes=self.num_classes,
                                  logit_dict={k: v for ds in self.db.data['datasets'].values() for k, v in ds.items()}))
        ds = ds.batched(batch_size, collation_fn=collation_fn)
        return ds
    
    def get_train_set(self):
        return self.get_dataset(self.get_train_shards())
    
    def get_validate_set(self):
        return self.get_dataset(self.get_validate_shards())
    
    def get_train_example(self):
        return next(iter(self.get_dataset_debug(self.get_train_shards())))
    
    def get_validate_example(self):
        return next(iter(self.get_dataset_debug(self.get_validate_shards())))


@dataclass
class AudiosetLogitsDataloader:
    audioset: AudiosetLogitsDataset = None
    num_workers: int = 12
    batch_size: int = 32
    seed: int = 0
    augmentation: Augmentation = None
    mix_interval: float = 2.0
    mix_beta: float = 2.0
    mix_targets: bool = False

    def __post_init__(self):
        self.rng = np.random.RandomState(self.seed)

    # @classmethod
    # def finalize_dogmatic_config(cls, config):
    #     config["audioset"] = {"factory": AudiosetLogitsDataset}
        
    #     config["augmentation"] = {
    #         "factory": Augmentation,
    #         "gain_augment": 0,
    #         "roll_axis": 1,
    #         "shift": None,
    #         "shift_range": 10000,
    #     }

    def _get_train_set(self, no_augment=False, balance=False):
        ds = self.audioset._get_train_set(balance=balance)

        augment = self.augmentation(
            not no_augment,
            not no_augment,
            not no_augment,
        )
        augmented_ds = ds.map(augment)

        if not no_augment and self.mix_interval is not None:
            mixup_ds = augmented_ds.copy()

            def mixup_fn(ex):
                if self.mix_interval is None or self.rng.rand() < 1 / self.mix_interval:
                    return ex
                idx = self.rng.randint(len(mixup_ds))
                ex2 = mixup_ds[idx]
                l = np.random.beta(self.mix_beta, self.mix_beta)
                l = max(l, 1.0 - l)
                ex["audio_data"] = ex["audio_data"] - ex["audio_data"].mean()
                ex2["audio_data"] = ex2["audio_data"] - ex2["audio_data"].mean()
                x = ex["audio_data"] * l + ex2["audio_data"] * (1.0 - l)
                x = x - x.mean()
                if self.mix_targets:
                    target = ex["weak_targets"] * l + ex2["weak_targets"] * (1.0 - l)
                else:
                    target = np.maximum(ex["weak_targets"], ex["weak_targets"])
                new_ex = {
                    "audio_data": x,
                    "weak_targets": target,
                    "logits": l * ex["logits"] + (1 - l) * ex2["logits"],
                    "example_id": ex["example_id"] + "+" + ex2["example_id"],
                }
                return new_ex

            augmented_ds = augmented_ds.map(mixup_fn)
        ds = augmented_ds
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

    def get_train_set(self, no_augment=False, balance=False):
        ds = self._get_train_set(balance=balance)

        ds = ds.map(AudiosetLogitsDataloader.finalize_example)
        sampler = RandomSampler(ds) if not no_augment else None
        dl = TorchLoader(
            dataset=ds,
            sampler=sampler,
            worker_init_fn=worker_init_fn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
        )
        return dl
