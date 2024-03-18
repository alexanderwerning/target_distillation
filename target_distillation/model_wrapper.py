from pathlib import Path

import numpy as np
import torch


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, feature_extractor=None, weight_path=None):
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        if weight_path is not None:
            assert Path(weight_path).exists()
            self.model.load_state_dict(torch.load(weight_path))
    
    def forward(self, batch):
        if self.feature_extractor is not None:
            assert batch["audio_data"].ndim == 3, batch["audio_data"].shape
            assert batch["audio_data"].shape[1] == 1, batch["audio_data"].shape
            float_input = batch["audio_data"].squeeze(1)
            mel_spec = self.feature_extractor(float_input)[:, None]
        else:
            assert isinstance(batch, (torch.Tensor, np.ndarray)), type(batch)
            mel_spec = batch

        x, features = self.model(mel_spec)
        return x, features
