import logging
import re
from copy import deepcopy

from src.utils import count_parameters
from src.models.cnn_transformer.passt_cnn import PaSST_CNN


def get_models_passt_cnn(configs):
    # initialize a student model
    net = PaSST_CNN(**configs["PaSST_CNN"]["init_kwargs"])
    # ema network
    ema_net = deepcopy(net)
    for param in ema_net.parameters():
        param.detach_()

    return net, ema_net


def check_tensor_name_decoder(tensor_name) -> bool:
    keyword_list = ["decoder", "cnn_projector", "transformer_projector", "merge_weight", "f_pool_module"]
    status = False
    for kw in keyword_list:
        if kw in tensor_name:
            status = True
            break
    return status


def get_param_lr(net: PaSST_CNN, configs: dict, logger: logging.Logger):
    lr_dict = configs["opt"]["param_groups"]
    assert len(lr_dict) == 4, "Configuration \"lr_dict\"'s length must be equal to 4"
    # ========================= set PaSST parameter =========================
    passt_params = [p for p in net.backbone.parameters()]
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
        for k, p in net.backbone.named_parameters():
            match = re.search(r"blocks.(\d+)", k)
            if match and (12 - int(match.group(1)) <= lr_dict["passt"]["step_lr"]):
                high_lr_params.append(p)
                logger.debug("<DEBUG> lr {0} = {1}".format(k, lr_dict["passt"]["lr"] * 2))
            elif "norm." in k:
                high_lr_params.append(p)
                logger.debug("<DEBUG> lr {0} = {1}".format(k, lr_dict["passt"]["lr"] * 2))
            else:
                low_lr_params.append(p)
                logger.debug("<DEBUG> lr {0} = {1}".format(k, lr_dict["passt"]["lr"]))
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
        for k, p in net.backbone.named_parameters():
            if "norm." not in k:  # Don't fix the last norm layer
                p.requires_grad = False
            else:
                logger.info("{0} is trainable".format(k))
    # Freeze the blocks whose id is less than the lr_dict["passt"]["freeze_layer"]
    elif lr_dict["passt"]["freeze_layer"] > 0:
        for k, p in net.backbone.named_parameters():
            match = re.search(r"blocks.(\d+)", k)
            if match and (int(match.group(1)) + 1 > lr_dict["passt"]["freeze_layer"]):
                logger.debug("<DEBUG> param {0}:  requires_grad = {1}".format(k, p.requires_grad))
            elif "norm." in k:
                # the norm layer above the last transformer block
                p.requires_grad = True
                logger.debug("<DEBUG> unfreeze {0}".format(k))
            else:
                p.requires_grad = False
                logger.debug("<DEBUG> freeze {0}".format(k))

    # ========================= set cnn parameters =========================
    if hasattr(net, "cnn"):
        cnn_params = [p for p in net.cnn.parameters()]
        cnn_ids = [id(p) for p in cnn_params]
        if lr_dict["cnn"]["lr"] <= 0:
            for p in cnn_params:
                p.requires_grad = False
        cnn_lr = [{"params": cnn_params, "lr": lr_dict["cnn"]["lr"], "weight_decay": lr_dict["cnn"]["weight_decay"]}]
    else:
        cnn_ids = []
        cnn_lr = []

    # ========================= set decoder parameters =========================
    decoder_params = [p for k, p in net.named_parameters() if check_tensor_name_decoder(k)]
    decoder_ids = [id(p) for p in decoder_params]
    if lr_dict["decoder"]["lr"] <= 0:
        for p in decoder_params:
            p.requires_grad = False
    for k, _ in net.named_parameters():
        if check_tensor_name_decoder(k):
            logger.debug("<DEBUG> lr {0} = {1}".format(k, lr_dict["decoder"]["lr"]))

    head_params = []
    for k, p in net.named_parameters():
        if ((id(p) not in passt_ids) and (id(p) not in cnn_ids) and (id(p) not in decoder_ids)):
            head_params.append(p)
            logger.debug("<DEBUG> lr {0} = {1}".format(k, lr_dict["head"]["lr"]))

    decoder_lr = [{"params": decoder_params, **lr_dict["decoder"]}]
    head_lr = [{"params": head_params, **lr_dict["head"]}]
    total_lr = passt_lr + cnn_lr + decoder_lr + head_lr
    logger.info("[INFO] Length of passt params is {0}, lr is {1}".format(len(passt_params), passt_lr[0]["lr"]))
    if len(cnn_lr) > 0:
        logger.info("[INFO] Length of cnn params is {0}, lr is {1}".format(len(cnn_params), cnn_lr[0]["lr"]))
    logger.info("[INFO] Length of decoder params is {0}, lr is {1}".format(len(decoder_params), decoder_lr[0]["lr"]))
    logger.info("[INFO] Length of head params is {0}, lr is {1}".format(len(head_params), head_lr[0]["lr"]))
    logger.info("Total  Params: %.3f M" % (count_parameters(net, trainable_only=False) * 1e-6))
    logger.info("Total Trainable Params: %.3f M" % (count_parameters(net) * 1e-6))
    return total_lr
