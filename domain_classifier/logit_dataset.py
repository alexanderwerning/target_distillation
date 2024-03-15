import numpy as np
import random
from dataclasses import dataclass
from distillation.data.linked_json import LinkedJsonDatabase
import paderbox as pb
from nt_paths import db_root, json_root
import torch
from distillation.data.linked_json import LinkedJsonDatabase
from paderbox.utils.nested import nested_merge
from padertorch.contrib.je.data.transforms import Collate


def encode_target(ds, num_classes=50):
    for k, v in ds.items():
        if "event_ids" in v:
            tgt = np.zeros(num_classes)
            for ev_id in v["event_ids"]:
                tgt[int(ev_id)] = 1
            v["weak_targets"] = tgt
    return ds

def get_dataset(ds_name, db, emb_db):
    ds = db.get_dataset(ds_name)
    fixed_ds = {k.replace(".wav", ""): v for k, v in ds.items()}
    encode_target(fixed_ds)
    merged_ds = nested_merge(fixed_ds, emb_db.get_dataset(ds_name))
    for ex_id, ex in merged_ds.items():
        ex["example_id"] = ex_id
    return merged_ds

def prepare_datasets(validation_fold, db_path, embedding_db_path):
    """
    [
    example_id: {'example_id':example_id, 'weak_targets': one-hot, 'logits': logits}
    ]"""
    train_ds_names = [f"fold0{i}" for i in [1,2,3,4,5] if i != validation_fold]
    emb_db = LinkedJsonDatabase(embedding_db_path)
    db = LinkedJsonDatabase(db_path)
    train_ds = {}
    for ds_name in train_ds_names:
        ds = get_dataset(ds_name, db, emb_db)
        train_ds.update(ds)
    val_ds_name = f"fold0{validation_fold}"
    validate_ds = get_dataset(val_ds_name, db, emb_db)
    coll_train_ds = Collate()(train_ds.values())
    coll_validate_ds = Collate()(validate_ds.values())
    return coll_train_ds, coll_validate_ds

def get_random_samples(samples, num):
    num_samples = samples['logits'].shape[0]
    rand_indices = np.random.randint(0, num_samples, size=(num,))
    return {'logits': samples['logits'][rand_indices],
            'target': samples['target'][rand_indices]}

def to_torch(x_list):
    return torch.as_tensor(np.stack(x_list), dtype=torch.float32).cuda()

@dataclass
class LogitDataset:
    source_validation_split_size: float = 1/5
    balance: bool = True
    target_db_path: str = json_root + "/esc50_32khz.json"
    target_embedding_db_path: str = db_root + "/logits/esc50_ensemble_logits_full/database.json"
    source_embedding_db_path: str = db_root + "/relabeling/audioset_full/database.json"
    validation_fold: int = 1

    def __post_init__(self):
        train_ds, validate_ds = prepare_datasets(validation_fold=self.validation_fold,
                                                db_path=self.target_db_path,
                                                embedding_db_path=self.target_embedding_db_path
                                            )
        target_train_logits = train_ds["logits"]
        target_validate_logits = validate_ds["logits"]
        self.train_positive_samples = {'logits': target_train_logits, 'target': np.ones(len(target_train_logits))}

        audioset_eval = LinkedJsonDatabase(self.source_embedding_db_path).data["datasets"]["eval"]

        source_train_logits = []
        source_validate_logits = []
        np.random.seed(13)  # TODO shuffle instead?
        for ex in audioset_eval.values():
            if np.random.random() < self.source_validation_split_size:
                source_validate_logits.append(ex["logits"])
            else:
                source_train_logits.append(ex["logits"])
        source_train_logits = np.stack(source_train_logits)
        source_validate_logits = np.stack(source_validate_logits)
        self.num_train_samples = len(target_train_logits)+len(source_train_logits)


        self.train_negative_samples = {'logits': source_train_logits, 'target': np.zeros(len(source_train_logits))}
        self.validate_set = {'logits': np.concatenate([source_validate_logits, target_validate_logits]),
                            'target': np.concatenate([np.zeros(len(source_validate_logits)), np.ones(len(target_validate_logits))])}

    def get_train_batch(self, batch_size, positive_mixup_rate=2.0):
        positive = get_random_samples(self.train_positive_samples, batch_size//2)
        negative = get_random_samples(self.train_negative_samples, batch_size//2)
        if positive_mixup_rate:
            num_mixup_examples = batch_size//2//positive_mixup_rate
            mixup_positive = get_random_samples(self.train_positive_samples, num_mixup_examples)
            for i in range(num_mixup_examples):
                positive['logits'][i] = np.maximum(positive['logits'][i], mixup_positive['logits'][i])
        batch_logits = np.concatenate([positive['logits'], negative['logits']])
        batch_target = np.concatenate([positive['target'], negative['target']])
        batch = {
            'logits': torch.as_tensor(batch_logits, dtype=torch.float32).cuda(),
            'target': torch.as_tensor(batch_target, dtype=torch.float32).cuda()
        }
        assert batch['target'].sum().item() == batch_size // 2
        assert len(batch['target']) == batch_size
        return batch
    
    def get_validate_set(self):
        return {
            'logits': torch.as_tensor(self.validate_set['logits'], dtype=torch.float32).cuda(),
            'target': torch.as_tensor(self.validate_set['target'], dtype=torch.float32).cuda()
            }
