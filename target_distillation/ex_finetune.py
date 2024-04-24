"""Reproduce the results of the paper "Efficient Large-Scale Audio Tagging Via Transformer-To-CNN Knowledge Distillation"""
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from target_distillation.data.loader import LogitsDataloader
from target_distillation.model import LightningModule
from target_distillation.ex_distill import get_schedule


@hydra.main(version_base=None, config_path="conf", config_name="finetune")
def main(cfg):
    """
    python ex_finetune.py hydra.run.dir=<parent-dir-of-checkpoints-dir> checkpoint=<checkpoint-filename> 
    """
    print(OmegaConf.to_yaml(cfg))
    storage_dir = Path(HydraConfig.get().run.dir)
    print(storage_dir)
    print(cfg.checkpoint)
    if cfg.checkpoint is None:
        print("No checkpoint given, use ImageNet init")
    else:
        trainer_state_dict = torch.load(str(Path(storage_dir)/"checkpoints"/cfg.checkpoint))
        if "state_dict" in trainer_state_dict:
            model_state_dict = trainer_state_dict["state_dict"]
        elif "model" in trainer_state_dict:
            model_state_dict = trainer_state_dict["model"]
        else:
            print(f"Missing 'state_dict' or 'model' keys in state dict, assume raw weights: {next(iter(trainer_state_dict.keys()))}")
            model_state_dict = trainer_state_dict
        some_key = next(iter(model_state_dict))
        prefixes = ["student.model.", "net.model.",
                    "model.student.model.", "model.net.model.", ""]
        for p in prefixes:
            if some_key.startswith(p):
                student_key_prefix = p
                break
        else:
            raise ValueError(f"Key prefix unknown: {some_key}")
        student_state_dict = {k[len(student_key_prefix):]: v for k, v in model_state_dict.items() if k.startswith(student_key_prefix)}
        torch.save(student_state_dict, str(Path(storage_dir)/"checkpoints/student_ckpt_latest.pth"))
        
        cfg.model.net.model.pretrained_name = str(Path(storage_dir)/"checkpoints/student_ckpt_latest.pth")
    
    db: LogitsDataloader = instantiate(cfg.db.loader)
    num_samples = 1600
    train_set = db.get_train_set(replacement=True, num_samples=num_samples)
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
    trainer.fit(model,
                train_set,
                validate_set)


if __name__ == "__main__":
    main()