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