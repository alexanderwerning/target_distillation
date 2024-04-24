import math
import torch
import numpy as np
from typing import Dict, Union, List, Tuple, Optional
import lazy_dataset
from dataclasses import dataclass
from dcase2024_task1_baseline.helpers.utils import mixstyle

## batching/loading

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

## example handling/processing

def build_targets(event_ids, num_events=527):
    targets = torch.zeros(num_events, dtype=int)
    for ev_id in event_ids:
        targets[ev_id] = 1
    return targets

class Preprocessing:
    def __init__(self, example_len, class_mapping, num_classes, logit_dict, target_key="weak_targets"):
        # self.device = device
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
        targets = torch.zeros(self.num_classes)
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
        logits = torch.as_tensor(self.logit_dict[example_id]['logits'])

        # fix this using bucketing
        if self.example_len is not None:
            audio_data_cut_padded = torch.zeros(1, self.example_len)
            if audio_data.shape[1] >= self.example_len:
                audio_data_cut_padded[:] = audio_data[:, :self.example_len]
            else:
                audio_data_cut_padded[:, :audio_data.shape[1]] = audio_data
        else:
            audio_data_cut_padded = audio_data
        # todo: also give sequence lengths -> only compute loss for non-padded signal

        example = {
            "__key__": example_id,  # map adds __key__ anyways, avoid None value
            "example_id": example_id,
            "logits": logits,
            self.target_key: targets,
            "audio_data": audio_data_cut_padded
        }
        return example

@dataclass
class Augmentation:
    gain_augment: int = 0
    roll_axis: int = 1
    shift: Optional[int] = None
    shift_range: int = 10000
    mixstyle_p: bool = 0
    mixstyle_alpha: bool = 0

    def roll_fn(self, x):
        # roll waveform (over time)
        if self.shift is None:
            if self.shift_range == 0:
                return x
            sf = torch.randint(-self.shift_range, self.shift_range, (1,)).item()
        else:
            sf = self.shift
        if isinstance(x, torch.Tensor):
            return torch.roll(x, sf, self.roll_axis)
        else:
            return np.roll(x, sf, self.roll_axis)

    def preprocess(self, ex, gain_augment, roll):
        if self.mixstyle_p > 0:
            ex["audio_data"] = mixstyle(ex["audio_data"], self.mixstyle_p, self.mixstyle_alpha)

        if gain_augment and self.gain_augment > 0:
            gain = torch.rand(1).item() * self.gain_augment * 2 - self.gain_augment
            amp = 10 ** (gain / 20)
            ex["audio_data"] = ex["audio_data"] * amp

        if roll:
            ex["audio_data"] = self.roll_fn(ex["audio_data"])
        return ex

    def __call__(self, ex, gain_augment=True, roll=True):
        return self.preprocess(ex, gain_augment=gain_augment, roll=roll)
    

def pad_or_truncate(audio_data, length):
    audio_len = audio_data.shape[-1]
    if audio_len < length:
        # pad
        return torch.pad(audio_data, ((0, 0), (0, length-audio_len)))
    else:
        # truncate
        return audio_data[:, :length]

@dataclass
class Mixup:
    mix_interval: float = 2.0
    mix_beta: float = 2.0
    mixup_ds: lazy_dataset.Dataset = None

    def __post_init__(self):
        self.ds_iter = None
        self._dist = torch.distributions.beta.Beta(self.mix_beta, self.mix_beta)
    
    def _get_next(self):
        if self.ds_iter is None:
            self.ds_iter = iter(self.mixup_ds)
        try:
            return next(self.ds_iter)
        except StopIteration:
            self.ds_iter = iter(self.mixup_ds)
            return next(self.ds_iter)

    def mixup_fn(self, ex):
        if self.mix_interval is None or torch.rand(1).item() < 1 / self.mix_interval:
            return ex
        ex2 = self._get_next()
        l = self._dist.sample()
        l = max(l, 1.0 - l)
        ex["audio_data"] = ex["audio_data"] - ex["audio_data"].mean()
        ex2["audio_data"] = ex2["audio_data"] - ex2["audio_data"].mean()
        x = ex["audio_data"] * l + ex2["audio_data"] * (1.0 - l)
        x = x - x.mean()
        target = torch.maximum(ex["weak_targets"], ex["weak_targets"])
        new_ex = {
            "audio_data": x,
            "weak_targets": target,
            "logits": l * ex["logits"] + (1 - l) * ex2["logits"],
            "example_id": ex["example_id"] + "+" + ex2["example_id"],
        }
        return new_ex
    
    def __call__(self, ex):
        return self.mixup_fn(ex)


## dataset mixing

# pb_sed.data_preparation.provider.DataProvider._compute_label_repetitions
def compute_label_repetitions(label_counts: Dict[str, int],
    min_counts: Union[int, float]) -> Dict[str, int]:
        """Compute label repetitions for a dataset.

        Args:
            label_counts: A dictionary mapping labels to their counts.
            min_counts: The minimum number of repetitions for each label.
                If a float in (0, 1), it is interpreted as a fraction of the
                maximum label count.
        
        Returns:
            A dictionary mapping labels to their repetitions.
        """
        max_count = max(label_counts.values())
        if isinstance(min_counts, float):
            assert 0. < min_counts < 1., min_counts
            min_counts = math.ceil(max_count * min_counts)
        assert isinstance(min_counts, int) and min_counts > 1, min_counts
        assert min_counts - 1 <= 0.9 * max_count, (f"The minimum number of label repetitions should be "
                                                  f"less than 90 percent of the dataset length {(min_counts, max_count)}")
        
        base_rep = 1 // (1 - (min_counts-1)/max_count)
        min_counts *= base_rep
        label_repetitions = {
            label: math.ceil(min_counts / count)
            for label, count in label_counts.items()
        }
        return label_repetitions

def build_repetition_groups(dataset: "lazy_dataset", label_repetitions: Dict[str, int]):
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

def get_repetition_groups(ds: "lazy_dataset", num_classes: int) -> List[Tuple["lazy_dataset", int]]:
    label_count_arr = np.zeros_like(num_classes)  # 527 classes
    for ex in ds:
        label_count_arr += ex['weak_targets']
    label_counts_dict = {
        i: label_count_arr[i] for i in range(len(label_count_arr))
    }
    label_reps_dict = compute_label_repetitions(
        label_counts_dict, min_counts=1 / 100
    )
    label_reps = np.zeros_like(label_count_arr)
    for i, lc in label_reps_dict.items():
        label_reps[i] = lc
    # # apply label count information to datasets with loaded audio
    repetition_groups = build_repetition_groups(
        ds, label_reps
    )
    return repetition_groups