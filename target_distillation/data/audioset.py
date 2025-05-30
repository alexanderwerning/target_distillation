import pickle as pkl
import warnings
from dataclasses import dataclass
from pathlib import Path

import lazy_dataset
import numpy as np
import paderbox as pb
import soundfile as sf
import webdataset as wds
import wids
from distillation.data.linked_json import LinkedJsonDatabase
from lazy_dataset.database import JsonDatabase
from nt_paths import db_root, json_root

from target_distillation.data.utils import (
    Preprocessing,
    build_targets,
    collation_fn,
    get_repetition_groups,
    pad_or_truncate,
)


class AudioReader:
    def __call__(self, example: dict):
        raise NotImplementedError()

    def get_available_examples(self) -> set:
        raise NotImplementedError()


@dataclass
class AudioReaderWids(AudioReader):
    database_path: str = db_root + "/wds/audioset"
    clip_length: float = 5.0
    sample_rate: int = 32000

    def __post_init__(self):
        self.database_path = Path(self.database_path)
        # file index: example_id -> wds index
        assert (
            self.database_path.exists()
        ), f"Database path {self.database_path} does not exist"
        file_index_path = self.database_path / "file_index.json"
        assert (
            file_index_path.exists()
        ), f"file index path does not exits: {file_index_path}"
        ds = JsonDatabase(json_path=self.database_path / "file_index.json")
        self.example_index = {
            id_: idx for v in ds.data.values() for id_, idx in v.items()
        }
        self.dset_names = [
            "balanced_train",
            "unbalanced_train",
            "eval",
        ]
        shards = []
        for dset_name in self.dset_names:
            shard_path = self.database_path / dset_name / "shardindex.json"
            if shard_path.exists():
                shardlist = pb.io.load(shard_path)["shardlist"]
                for shardinfo in shardlist:
                    shardinfo["url"] = dset_name + "/" + shardinfo["url"]
                shards.extend(shardlist)
            else:
                warnings.warn(f"shard path {shard_path} does not exist")

        self.ds = wids.ShardListDataset(
            shards, localname=lambda x: str(self.database_path / x)
        )

    def __call__(self, example):
        index = self.example_index[example["example_id"]]
        data = self.ds[index]
        audio_data, sample_rate = wds.torch_audio(
            key=".flac", data=data[".flac"].read()
        )
        meta = data[".json"]

        # example = wds.decode(handler=wds.torch_audio)(example)
        # example_id, audio_data, meta = wds.to_tuple("__key__", "flac", "json")(example)
        example_id = data["__key__"].rsplit("/", maxsplit=1)[-1]

        if "start" in example and "stop" in example:
            raise NotImplementedError()

        audio_data = pad_or_truncate(
            audio_data, int(self.clip_length * self.sample_rate)
        )
        example = {
            "audio_data": audio_data,
            "example_id": example_id,
            "weak_targets": build_targets(example["event_ids"]),
            "logits": example["logits"],
            **meta,
        }
        return example

    def get_available_examples(self) -> set:
        return set(self.example_index.keys())


@dataclass
class AudioReaderSD(AudioReader):
    database_path: str = db_root + "/audioset_32khz_flac_sd"
    clip_length: float = 10.0
    sample_rate: int = 32000

    def __post_init__(self):
        self.database_path: Path = Path(self.database_path)
        assert (
            self.database_path.exists()
        ), f"Database path {self.database_path} does not exist"
        self.db = JsonDatabase(json_path=self.database_path / "file_index.json")
        self.dset_names = [
            "balanced_train",
            "unbalanced_train",
            "eval",
        ]

    def __call__(self, example):
        example = example.copy()
        dset_idx = example["hdf_file_idx"]
        idx = example["hdf_idx"]

        dset_name = self.dset_names[dset_idx]
        example_id = example["example_id"]  # .rsplit("_", 1)[0]  # split _5.0 or _10.0
        meta_example = self.db.data["datasets"][dset_name][example_id]
        pickle_filename = meta_example["file"]
        pickle_pos = meta_example["pos"]
        # assert "start" in example and "end" in example, example.keys()
        # start_s = example["start"]
        # start = int(start_s * self.sample_rate)
        # example.pop("start")
        # end_s = example["end"]
        # end = int(end_s * self.sample_rate)
        # example.pop("end")
        try:
            with open(self.database_path / "data" / pickle_filename, "rb") as file:
                file.seek(pickle_pos)
                payload_example = pkl.load(file)
            waveform: np.array = payload_example["audio_data"].astype(
                np.float32
            )  # [start:end]
        except sf.LibsndfileError as e:
            print(e)
            raise RuntimeError(
                f"Failed to read audio {str(dset_name)}, example {idx}"
            ) from e

        # ensure channel dim exists
        assert (waveform.ndim == 2 and waveform.shape[0] == 1) or (waveform.ndim == 1)
        waveform = waveform.reshape(1, -1)

        waveform = pad_or_truncate(waveform, int(self.clip_length * self.sample_rate))
        example = {
            "audio_data": waveform,
            "example_id": example_id,
            "weak_targets": build_targets(example["event_ids"]),
            "logits": example["logits"],
        }
        return example

    def get_available_examples(self) -> set:
        return set([id_ for v in self.db.data["datasets"].values() for id_ in v])


# sample embeddings and load webdataset on demand
@dataclass
class AudiosetLogitsDataset:
    """Dataset is filtered by examples in given json file."""

    # logits are included in this file
    json_path: str = db_root + "/audioset_logits_db/database.json"
    audio_reader: AudioReader = None
    balance_classes: bool = False
    cache: bool = False

    def __post_init__(self):
        self.db = LinkedJsonDatabase(self.json_path)
        deleted = 0
        total = 0
        available_examples = self.audio_reader.get_available_examples()
        for name, ds in self.db.data["datasets"].items():
            delete = [ex_id for ex_id in ds if ex_id not in available_examples]
            for d in delete:
                ds.pop(d)
            deleted += len(delete)
            total += len(ds)
            print(f"Deleted {len(delete)} missing examples of {len(ds)} in {name}")
        print(f"Deleted {deleted} missing examples of {total}")
        assert self.balance_classes is False, "Do not balance classes for now"

    def get_raw_dataset(self, name):
        ds = self.db.get_dataset(name)
        return ds

    def get_train_set(self, balance=True):
        unbalanced_train = self.db.get_dataset("unbalanced_train")
        balanced_train = self.db.get_dataset("balanced_train")
        dict_ds = {**unbalanced_train, **balanced_train}
        assert len(dict_ds) == len(unbalanced_train) + len(balanced_train)
        ds = []
        for ex_idx, ex in dict_ds.items():
            ex["example_id"] = ex_idx
            ds.append(ex)

        if self.balance_classes and balance:
            # count labels
            num_classes = ds[0]["weak_targets"]
            print(f"Balancing {num_classes} classes")
            repetition_groups = get_repetition_groups(ds, num_classes)
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

    def get_validate_set(self):
        eval_set = self.db.get_dataset("eval")
        ds = []
        for ex_idx, ex in eval_set.items():
            ex["example_id"] = ex_idx
            ds.append(ex)
        ds = lazy_dataset.from_list(ds, immutable_warranty="wu")
        ds = ds.map(self.audio_reader)
        if self.cache:
            ds = ds.cache()
        return ds


@dataclass
class LogitFilter:
    logit_dict: dict = None

    def __call__(self, example):
        example_id = example[0].split("/")[-1]
        predicate = example_id in self.logit_dict
        if not predicate:
            print(f"Missing logits for: {example_id}")
        return predicate


@dataclass
class AudiosetLogitsWdsDataset:
    dataset_path: str = db_root + "/wds/audioset"
    json_path: str = db_root + "/audioset_logits_db/database.json"
    class_file: str = json_root + "/events/audioset_events.json"
    balance_classes: bool = False
    cache: bool = False
    sample_rate: int = 16_000

    # device: int = 0
    clip_length: float = 5.0
    num_classes: int = 527

    def __post_init__(self):
        assert self.balance_classes is False, "cannot class balance"
        classes = pb.io.load(self.class_file)
        self.num_classes = len(classes)
        self.class_mapping = {cls: i for i, cls in enumerate(classes)}
        self.inverse_class_mapping = dict(enumerate(classes))

        self.db = LinkedJsonDatabase(self.json_path).data["datasets"]
        self.example_len = int(self.clip_length * self.sample_rate)
        # self.num_samples_ds = {}
        # for ds_name in ["unbalanced_train", "balanced_train", "eval"]:
        #     self.num_samples_ds[ds_name] = sum(pb.io.load(self.dataset_path/ds_name/"sizes.json").keys())
        # self.num_train_samples = self.num_samples_ds["unbalanced_train"] + self.num_samples_ds["balanced_train"]
    
    def _get_subset_path(self, name):
        path = Path(self.dataset_path) / name
        if not (path/"0.tar").exists():
            raise ValueError(f"shards not found: {path/'0.tar'}")
        return path

    def get_train_shards(self):
        balanced_train_path = self._get_subset_path("balanced_train")
        unbalanced_train_path = self._get_subset_path("unbalanced_train")
        return [str(p) for p in balanced_train_path.glob("*.tar")] + [
            str(p) for p in unbalanced_train_path.glob("*.tar")
        ]

    def get_validate_shards(self):
        eval_path = self._get_subset_path("eval")
        return [str(p) for p in eval_path.glob("*.tar")]

    def get_dataset(self, name_or_shards):
        if isinstance(name_or_shards, str):
            dataset_path = self._get_subset_path(name_or_shards)
            shards = [str(p) for p in dataset_path.glob("*.tar")]
        else:
            shards = name_or_shards
        return self._get_dataset(shards)

    def _get_dataset(self, input_shards):
        logit_dict = {k: v for ds in self.db.values() for k, v in ds.items()}

        ds = (
            wds.WebDataset(input_shards, nodesplitter=wds.split_by_node, resampled=True)
            .shuffle(100)
            .decode(wds.torch_audio)
            .to_tuple("__key__", "flac", "json")
            .select(LogitFilter(logit_dict=logit_dict))
        )
        if self.cache:
            ds = ds.mcached()
        ds = ds.map(
            Preprocessing(
                # device=self.device,
                example_len=self.example_len,
                class_mapping=self.class_mapping,
                num_classes=self.num_classes,
                logit_dict=logit_dict,
            )
        ).with_epoch(160_000//8) # desired num examples: 160k/8workers

        return ds
        # pipeline = wds.DataPipeline(
        #     wds.SimpleShardList(input_shards),
        #     # random seed: uses torch distributed info: worker_seed = 1000*rank+local_worker_id
        #     wds.shuffle(100), # shuffle shard iterator
        #     wds.split_by_node,
        #     wds.split_by_worker, # split for each gpu? -> splits for rank and local worker process
        #     wds.tarfile_to_samples(), # iterate over examples
        #     wds.shuffle(1000), # shuffle examples within a shard in-memory
        #     wds.decode(wds.torch_audio),
        #     wds.to_tuple("__key__", "flac", "json"),
        #     wds.map(Preprocessing(device=self.device,
        #                           example_len=self.example_len,
        #                           class_mapping=self.class_mapping,
        #                           num_classes=self.num_classes,
        #                           logit_dict=logit_dict)),
        #     # wds.batched(self.batch_size, collation_fn=collation_fn)
        # )
        # return pipeline
        # # dataloader = wds.WebLoader(pipeline, num_workers=self.num_workers, batch_size=None)

    def get_dataset_debug(self, input_shards, batch_size=2):
        ds = wds.WebDataset(input_shards, shardshuffle=False).decode(wds.torch_audio)
        ds = ds.to_tuple("flac", "json").map(
            Preprocessing(
                # device="cpu",
                example_len=self.example_len,
                class_mapping=self.class_mapping,
                num_classes=self.num_classes,
                logit_dict={k: v for ds in self.db.values() for k, v in ds.items()},
            )
        )
        return ds
    
    def get_dataset_names(self):
        return ["balanced_train", "unbalanced_train", "eval"]

    def get_train_set(self):
        return self.get_dataset(self.get_train_shards())

    def get_validate_set(self):
        return self.get_dataset(self.get_validate_shards())

    def get_train_example(self):
        return next(iter(self.get_dataset_debug(self.get_train_shards())))

    def get_validate_example(self):
        return next(iter(self.get_dataset_debug(self.get_validate_shards())))
