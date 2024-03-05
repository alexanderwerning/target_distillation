from distillation.data.blend_dataset import AudiosetLogitsDataloader

from omegaconf import DictConfig

def get_data(data_cfg: DictConfig, loader_cfg: DictConfig):
    if data_cfg.name == "audioset_logits":
        assert data_cfg.sample_rate == 32_000
        loader = AudiosetLogitsDataloader(batch_size=loader_cfg.batch_size,
                                          num_workers=loader_cfg.num_workers)
    else:
        raise ValueError(data_cfg.name)
    return loader

# def get_data_provider_config(
#     db_name, batch_size, num_workers, sample_rate=16_000, **kwargs
# ):
#     if db_name == "audioset":
#         config = {
#             "factory": AudioSetProvider,
#             "train_set": {
#                 "balanced_train": 1,
#                 "unbalanced_train": 1,
#             },
#             "audio_reader": {
#                 "source_sample_rate": 32000,
#                 "target_sample_rate": 32000,
#             },
#             "train_transform": {
#                 "factory": RawWaveformTransform,
#                 "label_encoder": {
#                     "factory": MultiHotAlignmentEncoder,
#                     "label_key": "events",
#                 },
#             },
#             "test_transform": {
#                 "factory": RawWaveformTransform,
#                 "label_encoder": {
#                     "factory": MultiHotAlignmentEncoder,
#                     "label_key": "events",
#                 },
#             },
#         }
#     elif db_name == "esc50_wav_sd":
#         assert sample_rate == 32_000
#         config = {
#             "factory": Esc50ProviderSd,
#             "root_path": db_root + "/esc50_32khz_wav_sd",
#             "batch_size": batch_size,
#             "num_workers": num_workers,
#         }
#     elif db_name == "audioset_logits":
#         assert sample_rate == 32_000
#         config = {
#             "factory": AudiosetLogitsDataloader,
#             "batch_size": batch_size,
#             "num_workers": num_workers,
#         }


# def get_database_metric(db_name):
#     if "esc50" in db_name:
#         return "top1acc_weak"
#     elif db_name.startswith("audioset"):
#         return "map_weak"
#     else:
#         raise ValueError(db_name)
