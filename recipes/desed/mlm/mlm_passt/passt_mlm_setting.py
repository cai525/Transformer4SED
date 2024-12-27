from recipes.desed.mlm.mlm_setting import *
from src.models.passt.passt_sed import PaSST_SED


def get_model_passt(configs):
    net = PaSST_SED(**configs["PaSST_SED"])
    for k, p in net.patch_transformer.named_parameters():
        p.requires_grad = False
    return net
