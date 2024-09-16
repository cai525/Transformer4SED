from copy import deepcopy

from recipes.basic_setting import *
from src.models.cnn_transformer.atst_cnn import CRNN


def get_models_atst_cnn(configs):
    net = CRNN(
        unfreeze_atst_layer=0,
        **configs["ATST_CNN"]["init_kwargs"],
        atst_init="ROOT-PATH/pretrained_model/atst.ckpt",
    )
    # ema network
    ema_net = deepcopy(net)
    for param in ema_net.parameters():
        param.detach_()

    return net, ema_net
