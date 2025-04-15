import numpy as np
import soundfile as sf
import torch
import torchaudio


def waveform_modification(filepath, pad_to, encoder):
    wav, _ = sf.read(filepath)
    wav = to_mono(wav)
    wav, pad_mask = pad_wav(wav, pad_to, encoder)
    wav = torch.from_numpy(wav).float()
    wav = normalize_wav(wav)
    return wav, pad_mask


def normalize_wav(wav):
    return wav / (torch.max(torch.max(wav), -torch.min(wav)) + 1e-10)


def to_mono(wav, rand_ch=False):
    if wav.ndim > 1:
        if rand_ch:
            ch_idx = np.random.randint(0, wav.shape[-1] - 1)
            wav = wav[:, ch_idx]
        else:
            wav = np.mean(wav, axis=-1)
    return wav


def pad_wav(wav, pad_to, encoder):
    if len(wav) < pad_to:
        pad_from = len(wav)
        wav = np.pad(wav, (0, pad_to - len(wav)), mode="constant")
    else:
        wav = wav[:pad_to]
        pad_from = pad_to
    pad_idx = np.ceil(encoder._time_to_frame(pad_from / encoder.sr))
    pad_mask = torch.arange(encoder.n_frames) >= pad_idx  # size = n_frame, [0, 0, 0, 0, 0, ..., 0, 1, ..., 1]
    return wav, pad_mask


def take_log(feature):
    amp2db = torchaudio.transforms.AmplitudeToDB(stype="amplitude")
    amp2db.amin = 1e-5
    return amp2db(feature).clamp(min=-50, max=80)


def setmelspectrogram(feature_cfg):
    return torchaudio.transforms.MelSpectrogram(sample_rate=feature_cfg["sample_rate"],
                                                n_fft=feature_cfg["n_window"],
                                                win_length=feature_cfg["n_window"],
                                                hop_length=feature_cfg["hop_length"],
                                                f_min=feature_cfg["f_min"],
                                                f_max=feature_cfg["f_max"],
                                                n_mels=feature_cfg["n_mels"],
                                                window_fn=torch.hamming_window,
                                                wkwargs={"periodic": False},
                                                power=1)  # 1:energy, 2:power


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