"""Reproduce the results of the paper "Efficient Large-Scale Audio Tagging Via Transformer-To-CNN Knowledge Distillation"""
# distill knowledge from strong teacher to efficient student
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pathlib import Path
import torch
torch.multiprocessing.set_start_method('spawn')


# trainer
from padertorch.contrib.aw.name_generator import default_name_generator
# from padertorch.contrib.aw.optimizer import AdamW
# from padertorch.contrib.aw.trainer import AutocastTrainer
# from pb_sed.paths import storage_root

# data
# from distillation.data.data_provider import (
#     get_data_provider_config,
#     get_database_metric,
# )

# models
# from hear21passt.models.passt import get_model as get_model_passt
# from distillation.model import DistillationModel
# from distillation.model_wrapper import ModelWrapper
# from EfficientAT.models.MobileNetV3 import get_model as get_mobilenet
from EfficientAT.models.preprocess import AugmentMelSTFT

# @ex.named_config
def as_labels():
    num_classes = 527
    data_provider = {
        "use_audioset_targets": True,
        "audioset": {
            "load_weak_targets_from_file": False,
            "audio_reader": {
                "add_class_labels": True
            }
        }
    }

# @ex.named_config
def esc50():
    database_name = "esc50_mp3_sd"
    # with mixup, BCE is needeed
    # loss_fn = {"factory": torch.nn.CrossEntropyLoss}
    num_classes = 50
    data_provider = {"validation_fold": 0}

# @ex.named_config
def ce_kd_loss():
    trainer_config = {'model': {'kd_loss': {'factory': torch.nn.CrossEntropyLoss}}}

# @ex.named_config
def ce_label_loss():
    trainer_config = {'model': {'label_loss': {'factory': torch.nn.CrossEntropyLoss}}}

# @ex.named_config
def jsc_run():
    device=[0,1,2,3]
    actual_batch_size = 120
    batch_size = actual_batch_size // len(device)
    num_epochs = 200 #  * 2 #  split audioset into 5s segments
    epoch_len = 100_000
    num_iterations = int(num_epochs * epoch_len / actual_batch_size)
    lr_sched_config = {
        "warm_up_len": int(8 / num_epochs * num_iterations),
        "ramp_down_start": int(80 / num_epochs * num_iterations),
        "ramp_down_len": int(95 / num_epochs * num_iterations),
        "last_lr_value": 0.01,
    }
    del actual_batch_size
    del num_epochs
    del epoch_len
    

#!/bin/bash
# python -m distillation.ex_distill with mn40 
    # jsc_run
    # database_name=audioset_logits 
    # num_workers=80 
    # data_provider.audioset.json_path=/p/project/westai-upb/awerning/db/relabeling/domain_classifier3_1/database.json
    # data_provider.audioset.audio_reader.audioset_path=$DB/audioset_32khz_flac_sd
    # compile=False
    # data_provider.use_audioset_targets=True
    # data_provider.audioset.balance_classes=False
    # data_provider.audioset.audio_reader.add_class_labels=True
    # validate=False
    # data_provider.audioset.load_weak_targets_from_file=False
# add_cmds(ex)
# distill_configs(ex)

# @ex.named_config
def short_test_run():
    num_iterations = 10
    # trainer_config = {"checkpoint_trigger": (10, "iteration"),}

def get_train_set(cfg_db, cfg_loader):
    # from data.data_provider import get_data
    # return get_data(cfg_db, cfg_loader).get_train_set()
    return instantiate(cfg_db.loader).get_train_set()
    


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    feature_extractor = instantiate(cfg.feature_extractor)
    cfg.trainer.stop_trigger[0] = int(100_000 * 200 / cfg.loader.batch_size)
    if cfg.trainer.storage_dir is None:
        group_name = default_name_generator()
        cfg.trainer.storage_dir = Path(cfg.system.storage_root)/cfg.experiment_name/group_name
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
    train_set = get_train_set(cfg.db, cfg.loader)
    # train_set, validate_set = get_datasets()

    # if lr_sched_config is not None:
    #         apply_lr_schedule(trainer, **lr_sched_config)

    #     # trainer.test_run(train_set, validate_set)
    #     if validation_hook_config is None:
    #         validation_hook_config = {}
    #     if validate:
    #         trainer.register_validation_hook(validate_set, **validation_hook_config)
    #     resume = (
    #         trainer.checkpoint_dir.exists()
    #         and (trainer.checkpoint_dir / "ckpt_latest.pth").exists()
    #     )
    #     if resume:
    #         print(
    #             f"Resuming training from {(trainer.checkpoint_dir/'ckpt_latest.pth')}"
    #         )
    resume = False
    print("Start training")
    trainer.train(
        train_set,
        resume=resume,
        device=cfg.device,
        track_emissions=False,
    )
    # storage_dir, width_mult, continue_fine_tune
    # data = get_data(cfg.db)
    # model = get_model(cfg.model)
    # trainer = get_trainer(cfg.trainer, model)
    # ex.commands["run"]()
    # if not continue_fine_tune:
    #     return
    # run fine tuning 
    # from distillation.ex_finetune import ex as fine_tune_ex
    # new_storage_dir = str(Path(storage_dir)/"fine_tune")

    # trainer_state_dict = torch.load(str(Path(storage_dir)/"checkpoints/ckpt_latest.pth"))
    # model_state_dict = trainer_state_dict["model"]
    # student_key_prefix = "student.model."
    # student_state_dict = {k[len(student_key_prefix):]: v for k, v in model_state_dict.items() if k.startswith(student_key_prefix)}
    # torch.save(student_state_dict, str(Path(storage_dir)/"checkpoints/student_ckpt_latest.pth"))

    # fine_tune_ex.run("main", config_updates={
    #     "storage_dir": new_storage_dir,
    #     "width_mult": width_mult,
    #     "pretrained_name":str(Path(storage_dir)/"checkpoints/student_ckpt_latest.pth")
    # }, named_configs=["efficientat"])

if __name__ == "__main__":
    main()