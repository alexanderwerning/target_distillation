"""Reproduce the results of the paper "Efficient Large-Scale Audio Tagging Via Transformer-To-CNN Knowledge Distillation"""
# distill knowledge from strong teacher to efficient student
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from padertorch.contrib.aw.name_generator import default_name_generator
from target_distillation.lr_schedule import apply_lr_schedule, get_total_iterations


@hydra.main(version_base=None, config_path="conf", config_name="distill")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # setup training time
    actual_batch_size = cfg.loader.batch_size  * cfg.trainer.virtual_minibatch_size
    cfg.trainer.stop_trigger[0] = int(100_000 * 200 / actual_batch_size)

    # if cfg.trainer.storage_dir is None:
        # stage 1
        # group_name = default_name_generator()
        # cfg.trainer.storage_dir = Path(cfg.system.storage_root)/cfg.experiment_name/group_name

    print(f"storage_dir: {cfg.trainer.storage_dir}")

    print("Instantiate trainer")
    trainer = instantiate(cfg.trainer)

    # if compile and hasattr(torch, "compile"):
    #     if hasattr(trainer.model, "net"):
    #         trainer.model.net = torch.compile(trainer.model.net)
    #     if hasattr(trainer.model, "student"):
    #         trainer.model.student = torch.compile(trainer.model.student)
    #     if hasattr(trainer.model, "teacher"):
    #         trainer.model.teacher = torch.compile(trainer.model.teacher)
    #     print("Compiled model")
    # else:
    #     print("Not compiling model")

    print("Get train set")
    db = instantiate(cfg.db.loader)
    train_set = db.get_train_set()
    validate_set = db.get_validate_set()

    num_iterations = get_total_iterations(cfg.epochs, cfg.num_iterations, cfg.db.num_train_samples)
    apply_lr_schedule(trainer, cfg.lr, num_iterations=num_iterations, **cfg.lr_schedule)

    # # trainer.test_run(train_set, validate_set)

    trainer.register_validation_hook(validate_set)
    resume = (
        trainer.checkpoint_dir.exists()
        and (trainer.checkpoint_dir / "ckpt_latest.pth").exists()
    )
    if resume:
        print(
            f"Resuming training from {(trainer.checkpoint_dir/'ckpt_latest.pth')}"
        )
    print("Start training")
    trainer.train(
        train_set,
        resume=resume,
        device=cfg.device,
        track_emissions=False,
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()