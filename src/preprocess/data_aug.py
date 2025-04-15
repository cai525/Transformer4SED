#Some codes are adopted from https://github.com/DCASE-REPO/DESED_task
import copy
import torch
import numpy as np
import random

import torchaudio

def frame_shift(features, label=None, net_pooling=None):
    if label is not None:
        batch_size, _, _ = features.shape
        shifted_feature = []
        shifted_label = []
        for idx in range(batch_size):
            shift = int(random.gauss(0, 90))
            shifted_feature.append(torch.roll(features[idx], shift, dims=-1))
            shift = int(-abs(shift) // net_pooling if shift < 0 else shift // net_pooling)
            shifted_label.append(torch.roll(label[idx], shift, dims=-1))
        return torch.stack(shifted_feature), torch.stack(shifted_label)
    else:
        batch_size, _, _ = features.shape
        shifted_feature = []
        for idx in range(batch_size):
            shift = int(random.gauss(0, 90))
            shifted_feature.append(torch.roll(features[idx], shift, dims=-1))
        return torch.stack(shifted_feature)



def mixup(features, label=None, permutation=None, c=None, alpha=0.2, beta=0.2, mixup_label_type="soft", power=None, repeat=True):
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
            if  repeat:
                permutation = torch.randperm(batch_size)
            else:
                while(True):
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
                raise NotImplementedError(
                    f"mixup_label_type: {mixup_label_type} not implemented. choice in "
                    f"{'soft', 'hard'}")

            return mixed_features, mixed_label
        else:
            return mixed_features


def time_mask(features, labels=None, net_pooling=None, mask_ratios=(10, 20)):
    if labels is not None:
        _, _, n_frame = labels.shape
        t_width = torch.randint(low=int(n_frame/mask_ratios[1]), high=int(n_frame/mask_ratios[0]), size=(1,))   # [low, high)
        t_low = torch.randint(low=0, high=n_frame-t_width[0], size=(1,))
        features[:, :, int(t_low * net_pooling):min(int((t_low+t_width)*net_pooling), len(features))] = 1e-4
        labels[:, :, t_low:t_low+t_width] = 0
        return features, labels
    else:
        _, _, n_frame = features.shape
        t_width = torch.randint(low=int(n_frame/mask_ratios[1]), high=int(n_frame/mask_ratios[0]), size=(1,))   # [low, high)
        t_low = torch.randint(low=0, high=n_frame-t_width[0], size=(1,))
        features[:, :, t_low:(t_low + t_width)] = 0
        return features


def feature_transformation(features, n_transform, choice, filter_db_range, filter_bands,
                           filter_minimum_bandwidth, filter_type, freq_mask_ratio, noise_snrs):
    feature_list = []
    for _ in range(n_transform):
        features_temp = features
        if choice[3]:
            B_, F_ , T_ = features_temp.shape
            assert F_ == 128
            assert T_ == 1000
            bias = 0.03*random.random()
            features_temp = torch.Tensor(freq_nonlinear(features_temp.detach().cpu().numpy(), bias=bias)).to(features)
        if choice[0]:
            features_temp = filt_aug(features_temp, db_range=filter_db_range, n_band=filter_bands,
                                        min_bw=filter_minimum_bandwidth, filter_type=filter_type)
        if choice[1]:
            freqm = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_ratio, iid_masks=True)
            features_temp = freqm(features_temp)
        if choice[2]:
            features_temp = add_noise(features_temp, snrs=noise_snrs)
            
        feature_list.append(features_temp)
    return feature_list


def filt_aug(features, db_range=[-0.5, 0.5], n_band=[3, 6], min_bw=6, filter_type="linear", if_log=True):
    # this is updated FilterAugment algorithm used for ICASSP 2022
    if not isinstance(filter_type, str):
        if torch.rand(1).item() < filter_type:
            filter_type = "step"
            n_band = [2, 5]
            min_bw = 4
        else:
            filter_type = "linear"
            n_band = [3, 6]
            min_bw = 6

    batch_size, n_freq_bin, _ = features.shape
    n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()   # [low, high)
    if n_freq_band > 1:
        while n_freq_bin - n_freq_band * min_bw + 1 < 0:
            min_bw -= 1
        band_bndry_freqs = torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1,
                                                    (n_freq_band - 1,)))[0] + \
                           torch.arange(1, n_freq_band) * min_bw
        band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))

        if filter_type == "step":
            band_factors = torch.rand((batch_size, n_freq_band)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            band_factors = 10 ** (band_factors / 20)

            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = band_factors[:, i].unsqueeze(-1).unsqueeze(-1)

        elif filter_type == "linear":
            band_factors = torch.rand((batch_size, n_freq_band + 1)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                for j in range(batch_size):
                    freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = \
                        torch.linspace(band_factors[j, i], band_factors[j, i+1],
                                       band_bndry_freqs[i+1] - band_bndry_freqs[i]).unsqueeze(-1)
            freq_filt = 10 ** (freq_filt / 20)
        if if_log:
            ret = features + torch.log(freq_filt + 0.00001)
        else:
            ret = features * freq_filt
        return ret

    else:
        return features


def freq_mask(features, mask_ratio=16):
    batch_size, n_freq_bin, _ = features.shape
    max_mask = int(n_freq_bin/mask_ratio)
    if max_mask == 1:
        f_widths = torch.ones(batch_size)
    else:
        f_widths = torch.randint(low=1, high=max_mask, size=(batch_size,))   # [low, high)

    for i in range(batch_size):
        f_width = f_widths[i]
        f_low = torch.randint(low=0, high=n_freq_bin-f_width, size=(1,))

        features[i, f_low:f_low+f_width, :] = 0
    return features


def add_noise_on_freq(features, snrs=(15, 30), dims=(1, 2)):
    if isinstance(snrs, (list, tuple)):
        snr = (snrs[0] - snrs[1]) * torch.rand((features.shape[0],), device=features.device).reshape(-1, 1, 1) + snrs[1]
    else:
        snr = snrs

    snr = 10 ** (snr / 20)
    sigma = torch.std(features, dim=dims, keepdim=True) / snr
    return features + torch.randn(features.shape, device=features.device) * sigma

def add_noise(waveform:torch.Tensor, noise:torch.Tensor, snr_db):
    waveform_rms = waveform.norm(p=2, dim=1)
    noise_rms = noise.norm(p=2, dim=1)
    snr = 10 ** (snr_db / 20)
    scale = snr * noise_rms / waveform_rms
    return (torch.unsqueeze(scale, -1) * waveform + noise) / 2


def freq_nonlinear(mel:np.ndarray, f=1, bias=0.02):
    mel = copy.deepcopy(mel)
    B, F, T = mel.shape
    mel = np.reshape(np.transpose(mel, (0, 2, 1)), (B*T, F))
    trans = lambda x: x + bias*np.sin(2*np.pi*(f*x + random.random()))
    ind = np.arange(F)
    ind_t  = F*trans(ind/F)
    for i in range(B*T):
        mel[i, :] = np.interp(ind, ind_t, mel[i, :])
    mel = np.reshape(mel, (B, T, F)).transpose((0, 2, 1))
    return mel
    
    


