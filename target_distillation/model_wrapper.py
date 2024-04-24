import torch

# thin wrapper to match output parameters

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, has_features=True):
        super().__init__()
        self.model = model
        self.has_features = has_features
    
    def forward(self, mel_spec):
        if self.has_features:
            x, features = self.model(mel_spec)
        else:
            x = self.model(mel_spec)
            features = None
        return x, features
