import re
import logging
from copy import deepcopy

from recipes.basic_setting import *
from src.models.atst.atst_feature_extraction import AtstFeatureExtractor
from src.models.atst.atst_sed import AtstSED


def get_models_atst(configs):
    if configs["generals"]["load_from_existed_path"] or configs["generals"]["finetune_mlm"]:
        configs["ATST_SED"]["init_kwargs"]["atst_path"] = None

    net = AtstSED(**configs["ATST_SED"]["init_kwargs"])
    net.mel_trans = AtstFeatureExtractor(n_mels=configs["feature"]["n_mels"],
                                         n_fft=configs["feature"]["n_fft"],
                                         hopsize=configs["feature"]["hopsize"],
                                         win_length=configs["feature"]["win_length"],
                                         fmin=configs["feature"]["fmin"],
                                         fmax=configs["feature"]["fmax"],
                                         sr=configs["feature"]["sr"])
    # ema network
    ema_net = deepcopy(net)
    for param in ema_net.parameters():
        param.detach_()

    return net, ema_net


def get_params(net: AtstSED, configs: dict, logger: logging.Logger):
    lr_dict = configs["opt"]["param_groups"]
    assert len(lr_dict) in (2, 3), "Configuration \"lr_dict\"'s length must be 2 or 3."
    encoder_params = [p for p in net.get_encoder().parameters()]
    encoder_ids = [id(p) for p in encoder_params]
    if not lr_dict["encoder"]["step_lr"]:
        encoder_lr = [{
            "params": encoder_params,
            "lr": lr_dict["encoder"]["lr"],
            "weight_decay": lr_dict["encoder"]["weight_decay"]
        }]
    else:
        # step learning rate
        low_lr_params = []
        high_lr_params = []
        for k, p in net.get_encoder().named_parameters():
            match = re.search(r"blocks.(\d+)", k)
            if match and (net.get_encoder_depth() - int(match.group(1)) <= lr_dict["encoder"]["step_lr"]):
                high_lr_params.append(p)
                logging.debug("<DEBUG> lr {0} = {1}".format(k, lr_dict["encoder"]["lr"] * 2))
            elif "norm_frame." in k:
                high_lr_params.append(p)
                logging.debug("<DEBUG> lr {0} = {1}".format(k, lr_dict["encoder"]["lr"] * 2))
            else:
                low_lr_params.append(p)
                logging.debug("<DEBUG> lr {0} = {1}".format(k, lr_dict["encoder"]["lr"]))
        encoder_lr = [{
            "params": low_lr_params,
            "lr": lr_dict["encoder"]["lr"],
            "weight_decay": lr_dict["encoder"]["weight_decay"]
        }, {
            "params": high_lr_params,
            "lr": lr_dict["encoder"]["lr"] * 2,
            "weight_decay": lr_dict["encoder"]["weight_decay"]
        }]

    for k, p in net.get_encoder().named_parameters():
        p.requires_grad = True
    # freeze the encoder model when lr <= 0
    if lr_dict["encoder"]["lr"] <= 0:
        for k, p in net.get_encoder().named_parameters():
            if "norm_frame." not in k:  # Don't fix the last norm layer
                p.requires_grad = False
            else:
                logger.info("{0} is trainable".format(k))

    # Freeze the blocks whose id is less than the lr_dict["encoder"]["freeze_layer"]
    if lr_dict["encoder"]["freeze_layer"] > 0:
        for k, p in net.get_encoder().named_parameters():
            match = re.search(r"blocks.(\d+)", k)
            if match and (int(match.group(1)) + 1 > lr_dict["encoder"]["freeze_layer"]):
                p.requires_grad = True
                logging.debug("<DEBUG> unfreeze {0}".format(k))
            elif "norm_frame." in k:
                # the norm layer above the last transformer block
                p.requires_grad = True
                logging.debug("<DEBUG> unfreeze {0}".format(k))
            else:
                p.requires_grad = False
                logging.debug("<DEBUG> freeze {0}".format(k))

    if len(lr_dict) == 2:
        backend_params = [p for p in net.parameters() if id(p) not in encoder_ids]
        backend_lr = [{"params": backend_params, **lr_dict["backend"]}]
        total_lr = encoder_lr + backend_lr
        logger.info("[INFO] Length of encoder params is {0}, lr is {1}".format(len(encoder_params), encoder_lr["lr"]))
        logger.info("[INFO] Length of backend params is {0}, lr is {1}".format(len(backend_params), backend_lr["lr"]))

    elif len(lr_dict) == 3:
        decoder_params = [p for k, p in net.named_parameters() if "decoder" in k]
        decoder_ids = [id(p) for p in decoder_params]
        if lr_dict["decoder"]["lr"] <= 0:
            for p in decoder_params:
                p.requires_grad = False
        head_params = [p for p in net.parameters() if ((id(p) not in encoder_ids) and (id(p) not in decoder_ids))]
        decoder_lr = [{"params": decoder_params, **lr_dict["decoder"]}]
        head_lr = [{"params": head_params, **lr_dict["head"]}]
        total_lr = encoder_lr + decoder_lr + head_lr
        logger.info("[INFO] Length of encoder params is {0}, lr is {1}".format(len(encoder_params),
                                                                               encoder_lr[0]["lr"]))
        logger.info("[INFO] Length of decoder params is {0}, lr is {1}".format(len(decoder_params),
                                                                               decoder_lr[0]["lr"]))
        logger.info("[INFO] Length of head params is {0}, lr is {1}".format(len(head_params), head_lr[0]["lr"]))

    return total_lr
