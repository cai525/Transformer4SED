from src.models.atst.atst_sed import AtstSED
from src.models.atst.atst_feature_extraction import AtstFeatureExtractor
from recipes.mlm.mlm_setting import *


def get_model_atst(configs):
    net = AtstSED(**configs["ATST_SED"])
    net.mel_trans = AtstFeatureExtractor(n_mels=configs["feature"]["n_mels"],
                                         n_fft=configs["feature"]["n_fft"],
                                         hopsize=configs["feature"]["hopsize"],
                                         win_length=configs["feature"]["win_length"],
                                         fmin=configs["feature"]["fmin"],
                                         fmax=configs["feature"]["fmax"],
                                         sr=configs["feature"]["sr"])
    for k, p in net.get_encoder().named_parameters():
        p.requires_grad = False
    return net
