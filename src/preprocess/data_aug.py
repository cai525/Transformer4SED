#Some codes are adopted from https://github.com/DCASE-REPO/DESED_task
import copy
import torch
import numpy as np
import random
from typing import Union, Tuple

import torchaudio


def frame_shift(features, label=None, net_pooling=None, max_shift_frame=90):
    if label is not None:
        batch_size, _, _ = features.shape
        shifted_feature = []
        shifted_label = []
        for idx in range(batch_size):
            shift = int(random.gauss(0, max_shift_frame))
            shifted_feature.append(torch.roll(features[idx], shift, dims=-1))
            shift = int(-abs(shift) // net_pooling if shift < 0 else shift // net_pooling)
            shifted_label.append(torch.roll(label[idx], shift, dims=-1))
        return torch.stack(shifted_feature), torch.stack(shifted_label)
    else:
        batch_size, _, _ = features.shape
        shifted_feature = []
        for idx in range(batch_size):
            shift = int(random.gauss(0, max_shift_frame))
            shifted_feature.append(torch.roll(features[idx], shift, dims=-1))
        return torch.stack(shifted_feature)


def mixup(
    features,
    label=None,
    permutation=None,
    c=None,
    alpha=0.2,
    beta=0.2,
    mixup_label_type="soft",
    power=None,
    repeat=True,
):
    """ Apply mixup algorithm to (feature, label) 
        Args:
            features: input features;
            label: label of features;
            permutation: shuffle of indexes for a new batch;
            c: mixing rate for mixup;
            alpha, beta: parameter of beta distribution;
            mixup_label_type: "soft" or "hard" ."soft" mixup gives the ratio of the mix \
            to the labels, "hard" mixup gives a 1 to every label present;
        
        Returns:
            (mixed_features, mixed_label) or mixed_features, depended on input 
    """
    with torch.no_grad():
        batch_size = features.size(0)

        if permutation is None:
            if repeat:
                permutation = torch.randperm(batch_size)
            else:
                while (True):
                    permutation = torch.randperm(batch_size)
                    combine = [(min(i, permutation[i]), max(i, permutation[i])) for i in range(batch_size)]
                    if len(set(combine)) == batch_size:
                        break

        if c is None:
            if mixup_label_type == "soft":
                c = np.random.beta(alpha, beta)
            elif mixup_label_type == "hard":
                c = np.random.beta(alpha, beta) * 0.4 + 0.3  # c in [0.3, 0.7]

        mixed_features = c * features + (1 - c) * features[permutation, :]
        if label is not None:
            if mixup_label_type == "soft":
                mixed_label = torch.clamp(c * label + (1 - c) * label[permutation, :], min=0, max=1)
                # # TODO: Test
                if power:
                    mixed_label_temp = torch.float_power(mixed_label, power)
                    mixed_label = mixed_label_temp.to(mixed_label)
            elif mixup_label_type == "hard":
                mixed_label = torch.clamp(label + label[permutation, :], min=0, max=1)
            else:
                raise NotImplementedError(f"mixup_label_type: {mixup_label_type} not implemented. choice in "
                                          f"{'soft', 'hard'}")

            return mixed_features, mixed_label
        else:
            return mixed_features


def time_mask(features, labels=None, net_pooling=None, mask_ratios=(10, 20)):
    if labels is not None:
        _, _, n_frame = labels.shape
        t_width = torch.randint(low=int(n_frame / mask_ratios[1]), high=int(n_frame / mask_ratios[0]),
                                size=(1, ))  # [low, high)
        t_low = torch.randint(low=0, high=n_frame - t_width[0], size=(1, ))
        features[:, :, int(t_low * net_pooling):min(int((t_low + t_width) * net_pooling), len(features))] = 1e-4
        labels[:, :, t_low:t_low + t_width] = 0
        return features, labels
    else:
        _, _, n_frame = features.shape
        t_width = torch.randint(low=int(n_frame / mask_ratios[1]), high=int(n_frame / mask_ratios[0]),
                                size=(1, ))  # [low, high)
        t_low = torch.randint(low=0, high=n_frame - t_width[0], size=(1, ))
        features[:, :, t_low:(t_low + t_width)] = 0
        return features


def feature_transformation(
    features,
    n_transform,
    choice,
    filter_db_range,
    filter_bands,
    filter_minimum_bandwidth,
    filter_type,
    freq_mask_ratio=None,
    noise_snrs=None,
    norm_std=5,
    log=False,
):
    feature_list = []
    for _ in range(n_transform):
        features_temp = copy.deepcopy(features)
        if choice[3]:
            bias = 0.03 * random.random()
            features_temp = torch.Tensor(freq_nonlinear(features_temp.detach().cpu().numpy(), bias=bias)).to(features)
        if choice[0]:
            features_temp = filt_aug(features_temp,
                                     db_range=filter_db_range,
                                     n_band=filter_bands,
                                     min_bw=filter_minimum_bandwidth,
                                     filter_type=filter_type,
                                     norm_std=norm_std,
                                     log=log)
        if choice[1]:
            freqm = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_ratio, iid_masks=True)
            features_temp = freqm(features_temp)
        if choice[2]:
            features_temp = add_noise(features_temp, snrs=noise_snrs)

        feature_list.append(features_temp)
    if n_transform == 1:
        return feature_list[0]
    return feature_list


def filt_aug(features, db_range=[-0.5, 0.5], n_band=[3, 6], min_bw=6, filter_type="step", log=False, norm_std=1):
    # this is updated FilterAugment algorithm used for ICASSP 2022
    batch_size, n_freq_bin, _ = features.shape
    n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1, )).item()  # [low, high)
    if n_freq_band > 1:
        while n_freq_bin - n_freq_band * min_bw + 1 < 0:
            min_bw -= 1
        band_bndry_freqs = torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1,
                                                    (n_freq_band - 1,)))[0] + \
                           torch.arange(1, n_freq_band) * min_bw
        band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))

        if filter_type == "step":
            band_factors = torch.rand(
                (batch_size, n_freq_band)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            band_factors = 10**(band_factors / 20)

            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] =\
                    band_factors[:,i].unsqueeze(-1).unsqueeze(-1)
        elif filter_type == "linear":
            band_factors = torch.rand(
                (batch_size, n_freq_band + 1)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                for j in range(batch_size):
                    freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = \
                        torch.linspace(band_factors[j, i], band_factors[j, i+1],
                                       band_bndry_freqs[i+1] - band_bndry_freqs[i]).unsqueeze(-1)
        else:
            raise Exception("Unkonwn filter augment type")

        if log:
            # corresponding to the process of extracting mel spectrogram
            ret = features + torch.log(freq_filt + 0.00001) / norm_std
        else:
            # ret = features * freq_filt
            raise NotImplementedError("[DEBUG] Don't support filter augumentation after log operation")
        return ret

    else:
        return features


def add_noise(features, snrs=(15, 30), dims=(1, 2)):
    if isinstance(snrs, (list, tuple)):
        snr = (snrs[0] - snrs[1]) * torch.rand(
            (features.shape[0], ), device=features.device).reshape(-1, 1, 1) + snrs[1]
    else:
        snr = snrs

    snr = 10**(snr / 20)
    sigma = torch.std(features, dim=dims, keepdim=True) / snr
    return features + torch.randn(features.shape, device=features.device) * sigma


def freq_nonlinear(mel: np.ndarray, f=1, bias=0.02):
    """
        Non-linear transformation for frequency dimension. 
        Args:
            mel: (B, C, F, T)
    """
    mel = copy.deepcopy(mel)  # important! don't delete it!
    B, F, T = mel.shape
    mel = np.reshape(np.transpose(mel, (0, 2, 1)), (B * T, F))
    trans = lambda x: x + bias * np.sin(2 * np.pi * (f * x + random.random()))
    ind = np.arange(F)
    ind_t = F * trans(ind / F)
    for i in range(B * T):
        mel[i, :] = np.interp(ind, ind_t, mel[i, :])
    mel = np.reshape(mel, (B, T, F)).transpose((0, 2, 1))
    return mel
