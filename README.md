This repository contains the code used in the paper ["Target-specific Dataset Pruning for Compression of Audio Tagging Models"](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000061.pdf)

The proposed method combines model compression and domain adaptation to obtain better efficient audio tagging models.

## Installation
Install requirements manually from requirements.txt and install the package.

## Data preparation
Refer to [audio-data](https://github.com/LAION-AI/audio-dataset) to obtain a webdataset version of [AudioSet](https://research.google.com/audioset/).
ESC-50 can be downloaded [here](https://github.com/karolpiczak/ESC-50?tab=readme-ov-file#download).
Create logit datasets for both AudioSet and the target datasets using target\_distillation/create\_ensemble\_embeddings.py.


## Domain classifier training
The notebook in domain\_classifier allows to train a domain classifier model for a given dataset. For new datasets, a dataset class needs to be added in target\_distillation/data.

## Model distillation
With the trained domain classifier, AudioSet can be filtered using the create_wds.py script.
Use the ex_distill.py script to distill a model, a fine-tuning is automatically performed after the training is finished.


If there are some missing files or packages, please let me know.
