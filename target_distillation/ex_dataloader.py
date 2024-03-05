"""Reproduce the results of the paper "Efficient Large-Scale Audio Tagging Via Transformer-To-CNN Knowledge Distillation"""
# distill knowledge from strong teacher to efficient student
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from padertorch.contrib.aw.utils import str_nested_shape_maybe

import torch
torch.multiprocessing.set_start_method('spawn')


def get_train_set(cfg_db, cfg_loader):
    return instantiate(cfg_db.loader).get_train_set()
    

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    print("Get train set")
    train_set = get_train_set(cfg.db, cfg.loader)
    # train_set, validate_set = get_datasets()
    data_it = iter(train_set)
    for i in range(5):
        ex = next(data_it)
        print(str_nested_shape_maybe(ex))

if __name__ == "__main__":
    main()