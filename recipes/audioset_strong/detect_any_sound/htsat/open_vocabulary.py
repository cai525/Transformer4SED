import logging
import random

import numpy as np
import torch

from recipes.audioset_strong.detect_any_sound.passt.open_vocabulary import OV_DASM_Trainer
from recipes.audioset_strong.detect_any_sound.passt.train import pool_strong_labels

logging.getLogger('numba').setLevel(logging.WARNING)


class OV_DASM_HTSAT_Trainer(OV_DASM_Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    ################ tool functions for training process ###################
    def preprocess(self, wav, label):
        if random.random() < 0.5:
            mixup_lambda = np.random.beta(10, 0.5)
            label = (label * mixup_lambda + torch.flip(label, dims=[0]) * (1 - mixup_lambda))
        else:
            mixup_lambda = None
        extractor = self.net.get_feature_extractor(mixup_lambda)
        mel = extractor(wav)
        # weak labels
        label_weak = pool_strong_labels(label)
        return mel, label, label_weak

    def preprocess_eval(self, wav):
        extractor = self.net.get_feature_extractor(None)
        mel = extractor(wav)
        return mel
