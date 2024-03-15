from pathlib import Path

import numpy as np
import torch


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, feature_extractor=None, weight_path=None):
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        # some models return a tuple (output, hidden_state)
        if weight_path is not None:
            assert Path(weight_path).exists()
            self.model.load_state_dict(torch.load(weight_path))
    
    def _forward_wrapper(self, batch, fn):
        if self.feature_extractor is not None:
            assert batch["audio_data"].ndim == 3, batch["audio_data"].shape
            assert batch["audio_data"].shape[1] == 1, batch["audio_data"].shape
            float_input = batch["audio_data"].squeeze(1)
            mel_spec = self.feature_extractor(float_input)[:, None]
        else:
            assert isinstance(batch, (torch.Tensor, np.ndarray)), type(batch)
            mel_spec = batch

        x, features = fn(mel_spec)
        return x, features

    def forward(self, batch):
        return self._forward_wrapper(batch, self.model)
    
    # def forward_return_emb(self, batch):
    #     return self._forward_wrapper(batch, self.model.forward_return_emb)

