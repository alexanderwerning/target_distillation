"""Reproduce the results of the paper "Efficient Large-Scale Audio Tagging Via Transformer-To-CNN Knowledge Distillation"""
# distill knowledge from strong teacher to efficient student
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from padertorch.contrib.aw.utils import str_nested_shape_maybe

import torch


@hydra.main(version_base=None, config_path="conf", config_name="distill")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    print("Instantiate dataset")
    db = instantiate(cfg.db.loader)
    print("Get train set")
    train_set = db.get_train_set()
    num_examples = 0
    example_ids = set()
    for ex in train_set:
        keys = ex['example_id']
        num_examples += len(keys)
        for k in keys:
            example_ids.add(k)
    print(num_examples, len(example_ids))
    data_it = iter(train_set)
    for i in range(5):
        ex = next(data_it)
        print(str_nested_shape_maybe(ex, sequence_limit=31))

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()