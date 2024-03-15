import random
import math
import torch
import numpy as np
from typing import Dict, Union, List, Tuple, Optional
import lazy_dataset
from dataclasses import dataclass

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
    targets = np.zeros(num_events)
    for ev_id in event_ids:
        targets[ev_id] = 1
    return targets

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
class Augmentation:
    gain_augment: int = 0
    roll_axis: int = 1
    shift: Optional[int] = None
    shift_range: int = 10000
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.RandomState(self.seed)   

    def roll_fn(self, x):
        # roll waveform (over time)
        x = np.asarray(x)
        if self.shift is None:
            sf = int(np.random.random_integers(-self.shift_range, self.shift_range))
        else:
            sf = self.shift
        return np.roll(x, sf, self.roll_axis)

    def preprocess(self, ex, gain_augment, roll):
        ex["audio_data"] = ex["audio_data"].reshape(1, -1)
        ex["weak_targets"] = ex["weak_targets"].reshape(-1)
        if gain_augment and self.gain_augment > 0:
            gain = self.rng.randint(0, self.gain_augment * 2) - self.gain_augment
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
        return np.pad(audio_data, ((0, 0), (0, length-audio_len)))
    else:
        # truncate
        return audio_data[:, :length]

@dataclass
class Mixup:
    mix_interval: float = 2.0
    mix_beta: float = 2.0
    rng: np.random.RandomState = None
    mixup_ds: lazy_dataset.Dataset = None

    def mixup_fn(self, ex):
        if self.mix_interval is None or self.rng.rand() < 1 / self.mix_interval:
            return ex
        idx = self.rng.randint(len(self.mixup_ds))
        ex2 = self.mixup_ds[idx]
        l = np.random.beta(self.mix_beta, self.mix_beta)
        l = max(l, 1.0 - l)
        ex["audio_data"] = ex["audio_data"] - ex["audio_data"].mean()
        ex2["audio_data"] = ex2["audio_data"] - ex2["audio_data"].mean()
        x = ex["audio_data"] * l + ex2["audio_data"] * (1.0 - l)
        x = x - x.mean()
        target = np.maximum(ex["weak_targets"], ex["weak_targets"])
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


## data loader


# EfficientAT.helpers.init.spawn_get
def spawn_get(seedseq, n_entropy, dtype):
    child = seedseq.spawn(1)[0]
    state = child.generate_state(n_entropy, dtype=np.uint32)

    if dtype == np.ndarray:
        return state
    elif dtype == int:
        state_as_int = 0
        for shift, s in enumerate(state):
            state_as_int = state_as_int + int((2 ** (32 * shift) * s))
        return state_as_int
    else:
        raise ValueError(f'not a valid dtype "{dtype}"')


# EfficientAT.helpers.init.worker_init_fn
def worker_init_fn(wid):
    seed_sequence = np.random.SeedSequence(
        [torch.initial_seed(), wid]
    )

    to_seed = spawn_get(seed_sequence, 2, dtype=int)
    torch.random.manual_seed(to_seed)

    np_seed = spawn_get(seed_sequence, 2, dtype=np.ndarray)
    np.random.seed(np_seed)

    py_seed = spawn_get(seed_sequence, 2, dtype=int)
    random.seed(py_seed)