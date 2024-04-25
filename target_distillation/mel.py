import torchaudio
import torch

class LogMel(torch.nn.Module):
    def __init__(self, sr=32000, n_fft=4096, win_length=3072, hopsize=500, n_mels=256, f_min=0, f_max=None,
                  freqm=0, timem=0):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hopsize,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )

        freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)


        self.mel_augment = torch.nn.Sequential(
            freqm,
            timem
        )

    def forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: log mel spectrogram
        """
        x = self.mel(x)
        if self.training:
            x = self.mel_augment(x)
        x = (x + 1e-5).log()
        return x