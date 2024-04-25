"""Reproduce the results of the paper "Efficient Large-Scale Audio Tagging Via Transformer-To-CNN Knowledge Distillation"""
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from target_distillation.data.loader import LogitsDataloader

import pandas as pd

import padertorch


@hydra.main(version_base=None, config_path="conf", config_name="evaluate")
def main(cfg):
    """
    python eval_model.py checkpoint=<path/to/checkpoint>
    """
    print(OmegaConf.to_yaml(cfg))
    print(cfg.checkpoint)
    if cfg.checkpoint is None:
        print("No checkpoint given, use ImageNet init")
    else:
        trainer_state_dict = torch.load(cfg.checkpoint)
        if "state_dict" in trainer_state_dict:
            model_state_dict = trainer_state_dict["state_dict"]
        elif "model" in trainer_state_dict:
            model_state_dict = trainer_state_dict["model"]
        else:
            print(f"Missing 'state_dict' or 'model' keys in state dict, assume raw weights: {next(iter(trainer_state_dict.keys()))}")
            model_state_dict = trainer_state_dict
        prefixes = ["student.model.", "net.model.",
                    "model.student.model.", "model.net.model.", "model.", ""]
        for p in prefixes:
            for k in model_state_dict:
                if k.startswith(p):
                    student_key_prefix = p
                    break
            if k.startswith(p):
                break
        else:
            raise ValueError(f"Key prefix unknown: {next(iter(model_state_dict))}")
        print(f"Found prefix {p}")
        student_state_dict = {k[len(student_key_prefix):]: v for k, v in model_state_dict.items() if k.startswith(student_key_prefix)}
        
    db: LogitsDataloader = instantiate(cfg.db.loader)
    num_samples = 1600
    validate_set = db.get_validate_set()

    module = instantiate(cfg.model)
    module.net.model.load_state_dict(student_state_dict)

    module = module.eval().cuda()

    results = []
    classes = list(db.database.db.class_map.keys())

    with torch.no_grad():
        for batch in validate_set:
            batch = padertorch.data.example_to_device(batch, 0)
            soft_scores = module.predict(batch)
            predictions = soft_scores.argmax(-1).cpu().numpy()
            example_ids = batch['example_id']
            target = batch['target'].cpu().numpy()
            city =  batch['city']
            device = batch['device']
            for p, e_id, t, c, dev in zip(predictions, example_ids, target, city, device):
                results.append([e_id, p, t, classes[t], c.split("-")[0], dev])

    df = pd.DataFrame(results, columns=["id", "prediction", "target", "label", "city", "device"])
    df["accuracy"] = 1 * (df["target"] == df["prediction"])
    # create table:
    # n_examples x 5
    # | id | target | location | device | prediction |
    by_label = df.groupby("label")["accuracy"].mean()
    by_city = df.groupby("city")["accuracy"].mean()
    by_device = df.groupby("device")["accuracy"].mean()

    print("Label")
    print(by_label.to_markdown())
    print("City")
    print(by_city.to_markdown())
    print("Location")
    print(by_device.to_markdown())
    print("total", df["accuracy"].mean())
    # create table; accuracies by target/location/device


if __name__ == "__main__":
    main()