import random

import numpy as np

from recipes.finetune.train import Trainer
from src.preprocess.data_aug import frame_shift, mixup, feature_transformation


class AtstTrainer(Trainer):

    def __init__(self, *argc, **kwarg):
        super().__init__(*argc, **kwarg)
