import re
import logging
from copy import deepcopy

from recipes.basic_setting import *
from src.codec.encoder import Encoder
from src.models.passt.passt_sed import PaSST_SED


def get_encoder_passt(LabelDict, config):
    return Encoder(list(LabelDict.keys()),
                   audio_len=config["feature"]["audio_max_len"],
                   frame_len=config["feature"]["win_length"],
                   frame_hop=config["feature"]["hopsize"],
                   net_pooling=config["feature"]["net_subsample"],
                   sr=config["feature"]["sr"])


def get_models_passt(configs):
    net = PaSST_SED(**configs["PaSST_SED"])
    # ema network
    ema_net = deepcopy(net)
    for param in ema_net.parameters():
        param.detach_()

    return net, ema_net


def get_params(net: PaSST_SED, configs: dict, logger: logging.Logger):
    lr_dict = configs["opt"]["param_groups"]
    assert len(lr_dict) in (2, 3), "Configuration \"lr_dict\"'s length must be 2 or 3."
    passt_params = [p for p in net.patch_transformer.parameters()]
    passt_ids = [id(p) for p in passt_params]
    if not lr_dict["passt"]["step_lr"]:
        passt_lr = [{
            "params": passt_params,
            "lr": lr_dict["passt"]["lr"],
            "weight_decay": lr_dict["passt"]["weight_decay"]
        }]
    else:
        # step learning rate
        low_lr_params = []
        high_lr_params = []
        for k, p in net.patch_transformer.named_parameters():
            match = re.search(r"blocks.(\d+)", k)
            if match and (12 - int(match.group(1)) <= lr_dict["passt"]["step_lr"]):
                high_lr_params.append(p)
                logging.debug("<DEBUG> lr {0} = {1}".format(k, lr_dict["passt"]["lr"] * 2))
            elif "norm." in k:
                high_lr_params.append(p)
                logging.debug("<DEBUG> lr {0} = {1}".format(k, lr_dict["passt"]["lr"] * 2))
            else:
                low_lr_params.append(p)
                logging.debug("<DEBUG> lr {0} = {1}".format(k, lr_dict["passt"]["lr"]))
        passt_lr = [{
            "params": low_lr_params,
            "lr": lr_dict["passt"]["lr"],
            "weight_decay": lr_dict["passt"]["weight_decay"]
        }, {
            "params": high_lr_params,
            "lr": lr_dict["passt"]["lr"] * 2,
            "weight_decay": lr_dict["passt"]["weight_decay"]
        }]
    # freeze the passt model when lr <= 0
    if lr_dict["passt"]["lr"] <= 0:
        for k, p in net.patch_transformer.named_parameters():
            if "norm." not in k:  # Don't fix the last norm layer
                p.requires_grad = False
            else:
                logger.info("{0} is trainable".format(k))
    # Freeze the blocks whose id is less than the lr_dict["passt"]["freeze_layer"]
    if lr_dict["passt"]["freeze_layer"] > 0:
        for k, p in net.patch_transformer.named_parameters():
            match = re.search(r"blocks.(\d+)", k)
            if match and (int(match.group(1)) + 1 > lr_dict["passt"]["freeze_layer"]):
                p.requires_grad = True
                logging.debug("<DEBUG> unfreeze {0}".format(k))
            elif "norm." in k:
                # the norm layer above the last transformer block
                p.requires_grad = True
                logging.debug("<DEBUG> unfreeze {0}".format(k))
            else:
                p.requires_grad = False
                logging.debug("<DEBUG> freeze {0}".format(k))

    if len(lr_dict) == 2:
        backend_params = [p for p in net.parameters() if id(p) not in passt_ids]
        backend_lr = [{"params": backend_params, **lr_dict["backend"]}]
        total_lr = passt_lr + backend_lr
        logger.info("[INFO] Length of passt params is {0}, lr is {1}".format(len(passt_params), passt_lr["lr"]))
        logger.info("[INFO] Length of backend params is {0}, lr is {1}".format(len(backend_params), backend_lr["lr"]))

    elif len(lr_dict) == 3:
        decoder_params = [p for k, p in net.named_parameters() if "decoder" in k]
        decoder_ids = [id(p) for p in decoder_params]
        if lr_dict["decoder"]["lr"] <= 0:
            for p in decoder_params:
                p.requires_grad = False
        head_params = [p for p in net.parameters() if ((id(p) not in passt_ids) and (id(p) not in decoder_ids))]
        decoder_lr = [{"params": decoder_params, **lr_dict["decoder"]}]
        head_lr = [{"params": head_params, **lr_dict["head"]}]
        total_lr = passt_lr + decoder_lr + head_lr
        logger.info("[INFO] Length of passt params is {0}, lr is {1}".format(len(passt_params), passt_lr[0]["lr"]))
        logger.info("[INFO] Length of decoder params is {0}, lr is {1}".format(len(decoder_params),
                                                                               decoder_lr[0]["lr"]))
        logger.info("[INFO] Length of head params is {0}, lr is {1}".format(len(head_params), head_lr[0]["lr"]))

    return total_lr
