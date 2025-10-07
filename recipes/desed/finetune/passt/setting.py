import re
import logging
from copy import deepcopy

from src.models.passt.passt_sed import PaSST_SED


def get_models_passt(configs):
    net = PaSST_SED(**configs["PaSST_SED"]["init_kwargs"])
    # ema network
    ema_net = deepcopy(net)
    for param in ema_net.parameters():
        param.detach_()

    return net, ema_net


def check_tensor_name_decoder(tensor_name) -> bool:
    keyword_list = ["decoder", "f_pool_module", "transformer_projector"]
    status = False
    for kw in keyword_list:
        if kw in tensor_name:
            status = True
            break
    return status


def get_params(net: PaSST_SED, configs: dict, logger: logging.Logger):
    lr_dict = configs["opt"]["param_groups"]
    assert len(lr_dict) in (2, 3), "Configuration \"lr_dict\"'s length must be 2 or 3."
    passt_params = [p for p in net.backbone.parameters()]
    passt_ids = [id(p) for p in passt_params]
    if not lr_dict["encoder"]["step_lr"]:
        passt_lr = [{
            "params": passt_params,
            "lr": lr_dict["encoder"]["lr"],
            "weight_decay": lr_dict["encoder"]["weight_decay"],
        }]
    else:
        # step learning rate
        low_lr_params = []
        high_lr_params = []
        for k, p in net.backbone.named_parameters():
            match = re.search(r"blocks.(\d+)", k)
            if match and (12 - int(match.group(1)) <= lr_dict["encoder"]["step_lr"]):
                high_lr_params.append(p)
                logger.debug("<DEBUG> lr {0} = {1}".format(k, lr_dict["encoder"]["lr"] * 2))
            elif "norm." in k:
                high_lr_params.append(p)
                logger.debug("<DEBUG> lr {0} = {1}".format(k, lr_dict["encoder"]["lr"] * 2))
            else:
                low_lr_params.append(p)
                logger.debug("<DEBUG> lr {0} = {1}".format(k, lr_dict["encoder"]["lr"]))
        passt_lr = [{
            "params": low_lr_params,
            "lr": lr_dict["encoder"]["lr"],
            "weight_decay": lr_dict["encoder"]["weight_decay"]
        }, {
            "params": high_lr_params,
            "lr": lr_dict["encoder"]["lr"] * 2,
            "weight_decay": lr_dict["encoder"]["weight_decay"]
        }]
    # freeze the passt model when lr <= 0
    if lr_dict["encoder"]["lr"] <= 0:
        for k, p in net.backbone.named_parameters():
            if "norm." not in k:  # Don't fix the last norm layer
                p.requires_grad = False
            else:
                logger.info("{0} is trainable".format(k))
    # Freeze the blocks whose id is less than the lr_dict["encoder"]["freeze_layer"]
    if lr_dict["encoder"]["freeze_layer"] > 0:
        for k, p in net.backbone.named_parameters():
            match = re.search(r"blocks.(\d+)", k)
            if match and (int(match.group(1)) + 1 > lr_dict["encoder"]["freeze_layer"]):
                p.requires_grad = True
                logger.debug("<DEBUG> unfreeze {0}".format(k))
            elif "norm." in k:
                # the norm layer above the last transformer block
                p.requires_grad = True
                logger.debug("<DEBUG> unfreeze {0}".format(k))
            else:
                p.requires_grad = False
                logger.debug("<DEBUG> freeze {0}".format(k))

    # set decoder parameter
    decoder_params = [p for k, p in net.named_parameters() if check_tensor_name_decoder(k)]
    decoder_ids = [id(p) for p in decoder_params]
    if lr_dict["decoder"]["lr"] <= 0:
        for p in decoder_params:
            p.requires_grad = False
    for k, _ in net.named_parameters():
        if "decoder" in k:
            logger.debug("<DEBUG> lr {0} = {1}".format(k, lr_dict["decoder"]["lr"]))
    head_params = [p for p in net.parameters() if ((id(p) not in passt_ids) and (id(p) not in decoder_ids))]
    decoder_lr = [{"params": decoder_params, **lr_dict["decoder"]}]
    head_lr = [{"params": head_params, **lr_dict["head"]}]

    total_lr = passt_lr + decoder_lr + head_lr
    logger.info("[INFO] Length of passt params is {0}, lr is {1}".format(len(passt_params), passt_lr[0]["lr"]))
    logger.info("[INFO] Length of decoder params is {0}, lr is {1}".format(len(decoder_params), decoder_lr[0]["lr"]))
    logger.info("[INFO] Length of head params is {0}, lr is {1}".format(len(head_params), head_lr[0]["lr"]))

    return total_lr
