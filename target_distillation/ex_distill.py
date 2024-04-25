"""Reproduce the results of the paper "Efficient Large-Scale Audio Tagging Via Transformer-To-CNN Knowledge Distillation"""
# distill knowledge from strong teacher to efficient student
import os
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from target_distillation.data.loader import LogitsDataloader
from target_distillation.lr_schedule import linear_warmup_linear_down, linear_warmup_cosine_down


from target_distillation.model import LightningModule

def get_schedule(num_epochs, config):
    warm_up_len = config["warm_up_len"]
    ramp_down_start = config["ramp_down_start"]
    ramp_down_len = config["ramp_down_len"]
    last_lr_value = config["last_lr_value"]

    warm_up_len = int(num_epochs * warm_up_len)
    ramp_down_start = int(num_epochs * ramp_down_start)
    ramp_down_len = int(num_epochs * ramp_down_len)

    sched_fn = linear_warmup_cosine_down(
        warm_up_len, ramp_down_start, ramp_down_len, last_lr_value
    )
    return sched_fn

@hydra.main(version_base=None, config_path="conf", config_name="distill")
def main(cfg: DictConfig):
    """Run training

    use `python ex_distill +ckpt_path="/path/to/state.ckpt"` to continue training

    cp_mobile model:
    python ex_distill.py model/arch@model.net=cp_mobile
    """

    print(OmegaConf.to_yaml(cfg))

    print("Get train set")
    db: LogitsDataloader = instantiate(cfg.db.loader)
    train_set = db.get_train_set()
    validate_set = db.get_validate_set()

    module = instantiate(cfg.model)

    # manually instantiate optimizer to add parameters
    opt_cls = hydra.utils.get_class(cfg.optimizer._target_)
    optimizer = opt_cls(
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        params = module.parameters()
    )

    lr_schedule = get_schedule(num_epochs=cfg.epochs, config=cfg.lr_schedule)

    model = LightningModule(module,
                            optimizer,
                            lr_schedule=torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, 
                                                                          lr_lambda=lr_schedule))
    trainer = instantiate(cfg.trainer)
    if "ckpt_path" in cfg:
        print("Resuming from checkpoint")
        ckpt_path = cfg.ckpt_path
    else:
        ckpt_path = None
    trainer.fit(model,
                train_set,
                validate_set,
                ckpt_path=ckpt_path)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high') # 'medium'
    # torch.multiprocessing.set_start_method('spawn')
    main()