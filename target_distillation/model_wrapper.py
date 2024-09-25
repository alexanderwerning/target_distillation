import torch
from transformers import ClapProcessor, ClapModel

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

class ClapModelWrapper(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path):
        super().__init__()
        self.processor = ClapProcessor.from_pretrained(pretrained_model_name_or_path)
        self.model = ClapModel.from_pretrained(pretrained_model_name_or_path)
    
    def encode_audio(self, audio_data):
        # cast back to numpy
        audio_data = [a[0] for a in audio_data.cpu().numpy()]
        
        inputs = self.processor(audios=audio_data, return_tensors="pt", sampling_rate=32000)
        return self.model.get_audio_features(**inputs)
    
    def forward(self, inputs):
        return self.encode_audio(inputs)

class NoFeatureExtractor(torch.nn.Module):
    def forward(self, inputs):
        return inputs