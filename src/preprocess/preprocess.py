import copy
import random

import numpy as np
import torch

from src.preprocess.augmentMelSTFT import AugmentMelSTFT
from src.preprocess.data_aug import mixup, add_noise, frame_shift, feature_transformation, time_mask


class Preprocess:
    """ Preprocess to origin wave. Include 
    1. transforme to mel spectrum;
    2. data augmentation.
    """

    def __init__(self, mixup_rate=(0, 0), training=True, device=torch.device('cuda')) -> None:
        self.mel_trans = AugmentMelSTFT(n_mels=128,
                                        sr=32000,
                                        win_length=800,
                                        hopsize=320,
                                        n_fft=1024,
                                        freqm=48,
                                        timem=192,
                                        htk=False,
                                        fmin=0.0,
                                        fmax=None,
                                        norm=1,
                                        fmin_aug_range=10,
                                        fmax_aug_range=2000).to(device)
        self.training = training
        if not training:
            self.mel_trans.training == False
            self.mel_trans.eval()
        else:
            self.mel_trans.training == True
            self.mel_trans.train()
        # mixup
        self.if_mix = False
        self.mixup_rate = mixup_rate
        self.feature_transform_dict = {
            "n_transform": 2,                     # 0: no augmentation below is applied. 1: same augmentation below is applied on student/teacher model input. 2: different augmentations below is applied on student/teacher model input.
            "choice": [ 1, 0, 0 ],                # apply the chosen data augmentations: [ FilterAugment, freq_mask, add_noise ]
            "filter_db_range": [ -4.5, 6 ],       # db range of FilterAugment to be applied on each band
            "filter_bands": [ 2, 5 ],             # range of frequency band number in FilterAugment
            "filter_minimum_bandwidth": 4,
            "filter_type": "step",
            "freq_mask_ratio": 16,                # maximum ratio of freuqnecy masking range. max 1/16 of total frequnecy number will be masked
            "noise_snrs": [ 35, 40 ]              # snr of original signal wrpt the noise added.
        }

    def wav2mel(self, wav: torch.Tensor):
        """ 
        Transforme to mel spectrum;
        Args:
            wav: shape : (Batch, sample_len)
        Return:
            mel : shape : (Batch, Frequency , Frame)
        """
        mel = self.mel_trans(wav)
        return mel

    # def label_independent_aug(self, mel, mask):
    #     """ Data augmentation which is independent to label
    #     """
    #     # mixup may also be a efficient way to add noise (according to ICT)
    #     label_index = torch.logical_not(mask.unlabel)
    #     permutation = torch.randperm(mel.size(0))
    #     c_unlabel=max(np.random.beta(5, 0.5),0.75)
    #     c_label=max(np.random.beta(10, 0.5),0.75)
    #     len_label = 8
    #     mel_old = copy.deepcopy(mel)
    #     mel[label_index] = c_label * mel_old[label_index] + (1 - c_label) * mel_old[permutation[:len_label], :]
    #     mel[mask.unlabel] = c_unlabel * mel_old[mask.unlabel] + (1 - c_unlabel) * mel_old[permutation[len_label:], :]
    #     mel = add_noise_on_freq(mel)
    #     return mel

    def add_noise(self, wav: torch.Tensor, base_snr):
        noise = torch.randn(wav.shape).to(wav)
        snr_dbs = random.randrange(-3, 3) + base_snr
        return add_noise(wav, noise, snr_dbs)


def preprocess_mt(wav: torch.Tensor, label: torch.Tensor, mask, subsample=1):
    """ preprocess before  mean-teacher(mt) model   
    
    Return:
        (teacher mel feature, student mel feature, label)
    """
    util = Preprocess(mixup_rate=(0, 0.5), training=True)
    # ============ Augment to wav =============================
    # # mixup(time)
    # if util.mixup_rate[0] > torch.rand(1).item():
    #     for m in [mask.strong, mask.weak]:
    #         wav[m], label[m] = mixup(wav[m], label[m], alpha=80, beta=80, power=0.2, repeat=False)

    #     util.if_mix = True
    # else:
    util.if_mix = False
    
    # # # add noise
    # mask_nosyn = torch.logical_not(mask.strong_syn)  # don't add noise to synthetic data
    # base_snr = random.randrange(20, 50)
    # wav[mask_nosyn] = util.add_noise(wav[mask_nosyn], base_snr)

    # ============ Transform to mel respectively ==============
    mel= util.wav2mel(wav)  # mel for student
    # ============ data augmentation on spectrum ===============
    # time shift
    mel, label = frame_shift(mel, label, net_pooling=subsample)
    # mixup(mel)
    if (util.if_mix == False) and (util.mixup_rate[1] > torch.rand(1).item()):
        for m in [mask.strong, mask.weak]:
            mel[m], label[m] = mixup(mel[m], label[m], c=np.random.beta(10, 0.5))
    # time masking
    # mel[mask.strong], label[mask.strong] = time_mask(mel[mask.strong], label[mask.strong],
    #                                                 subsample,
    #                                                 mask_ratios=[5, 20])
    # Do label-independent augmentation
    stu_mel, tch_mel = feature_transformation(mel, **util.feature_transform_dict)
    return tch_mel, stu_mel, label


def preprocess_eval(wav: torch.Tensor):
    """ preprocess in evaluation period """
    util = Preprocess(training=False)
    return util.wav2mel(wav)
