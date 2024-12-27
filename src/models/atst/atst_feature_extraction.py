import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB



class MinMax():

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, input):
        min_, max_ = None, None
        if self.min is None:
            min_ = torch.min(input)
            max_ = torch.max(input)
        else:
            min_ = self.min
            max_ = self.max
        input = (input - min_) / (max_ - min_) * 2. - 1.
        return input

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ATSTNorm(nn.Module):

    def __init__(self):
        super(ATSTNorm, self).__init__()
        # Audio feature extraction
        self.amp_to_db = AmplitudeToDB(stype="power", top_db=80)
        self.scaler = MinMax(min=-79.6482, max=50.6842)  # TorchScaler("instance", "minmax", [0, 1])

    def amp2db(self, spec):
        return self.amp_to_db(spec).clamp(min=-50, max=80)

    def forward(self, spec):
        spec = self.scaler(self.amp2db(spec))
        return spec


class AtstFeatureExtractor(nn.Module):

    def __init__(self, n_mels=64, n_fft=1024, hopsize=160, win_length=1024, fmin=60, fmax=7800, sr=16000) -> None:
        super().__init__()
        self.transform = MelSpectrogram(sample_rate=sr,
                                        f_min=fmin,
                                        f_max=fmax,
                                        hop_length=hopsize,
                                        win_length=win_length,
                                        n_fft=n_fft,
                                        n_mels=n_mels)
        self.to_db = ATSTNorm()

    def forward(self, wavs: torch.Tensor):
        return self.normolize(self.wav2mel(wavs))

    def wav2mel(self, wavs: torch.Tensor) -> torch.Tensor:
        mels = self.transform(wavs)
        return mels

    def normolize(self, mels: torch.Tensor) -> torch.Tensor:
        return self.to_db(mels)
