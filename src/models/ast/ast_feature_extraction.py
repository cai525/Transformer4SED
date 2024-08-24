import torch
import torchaudio


def setmelspectrogram_ast(feature_cfg):
    return ASTFeatsExtraction(sample_frequency=feature_cfg["sr"],
                              frame_length=feature_cfg["window_ms"],
                              frame_shift=feature_cfg["hop_ms"],
                              num_mel_bins=feature_cfg["n_mels"],
                              audioset_mean=feature_cfg["scaler"]["mean"],
                              audioset_std=feature_cfg["scaler"]["std"])


class ASTFeatsExtraction:

    def __init__(self,
                 audioset_mean=-4.2677393,
                 audioset_std=4.5689974,
                 target_length=1024,
                 sample_frequency=16000,
                 num_mel_bins=128,
                 frame_shift=10,
                 frame_length=25):
        super(ASTFeatsExtraction, self).__init__()
        self.audioset_mean = audioset_mean
        self.audioset_std = audioset_std
        self.target_length = target_length
        self.sample_frequency = sample_frequency
        self.num_mel_bins = num_mel_bins
        self.frame_shift = frame_shift
        self.frame_length = frame_length

    def __call__(self, waveform):

        def get_bank(waveform):
            waveform = waveform.squeeze(0)
            waveform = waveform - torch.mean(waveform, -1)
            fbank = torchaudio.compliance.kaldi.fbank(waveform.unsqueeze(0),
                                                      htk_compat=True,
                                                      sample_frequency=self.sample_frequency,
                                                      use_energy=False,
                                                      window_type='hanning',
                                                      num_mel_bins=self.num_mel_bins,
                                                      frame_length=self.frame_length,
                                                      dither=0.0,
                                                      frame_shift=10)
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, self.target_length - fbank.shape[0]), mode='constant')
            fbank = (fbank - self.audioset_mean) / (self.audioset_std * 2)
            return fbank

        B, L = waveform.shape
        fbank_list = []
        for i in range(B):
            fbank_list.append(get_bank(waveform[i]).unsqueeze(0))
        fbank = torch.cat(fbank_list, dim=0)
        fbank = fbank.transpose(1, 2)
        return fbank
