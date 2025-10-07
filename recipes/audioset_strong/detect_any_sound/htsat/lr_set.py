import logging

import torch

from recipes.desed.detect_any_sound.detect_any_sound.finetune.setting import set_lr
from src.utils.statistics.model_statistic import count_parameters
from src.models.detect_any_sound.detect_any_sound_htast import DASM_HTSAT


def set_backbone_lr(net: DASM_HTSAT, lr_dict: dict):
    # ========================= set PaSST parameter =========================
    backbone_params = [p for p in net.backbone.parameters()]
    backbone_ids = [id(p) for p in backbone_params]
    backbone_lr = [{"params": backbone_params, "lr": lr_dict["lr"], "weight_decay": lr_dict["weight_decay"]}]
    # freeze the passt model when lr <= 0
    if lr_dict["lr"] <= 0:
        for k, p in net.backbone.named_parameters():
            p.requires_grad = False

    return backbone_lr, backbone_ids


def get_lr_htsat(net: DASM_HTSAT, configs: dict, logger: logging.Logger):
    lr_dict = configs["opt"]["param_groups"]

    # backbone
    backbone_lr, backbone_ids = set_backbone_lr(net, lr_dict["backbone"])
    # cnn
    if hasattr(net, "cnn"):
        cnn_params = [p for p in net.cnn.parameters()]
        cnn_lr, cnn_ids = set_lr(lr_dict["cnn"], cnn_params)
    else:
        cnn_ids = []
        cnn_lr = []

    # at_decoder
    at_decoder_params = [p for k, p in net.named_parameters() if "at_decoder" in k]
    at_decoder_lr, at_decoder_ids = set_lr(lr_dict["at_decoder"], at_decoder_params)

    # sed_decoder (with parameters of joint module)
    def check_tensor_name_decoder(tensor_name: str) -> bool:
        keyword_list = [
            "sed_decoder",
            "f_pool_module",
            "cnn_projector",
            "transformer_projector",
            "at_projector",
            "merge_weight",
            "norm_before_pool",
            "norm_after_merge",
        ]
        status = False
        for kw in keyword_list:
            if kw in tensor_name:
                status = True
                break
        return status

    sed_decoder_params = [p for k, p in net.named_parameters() if check_tensor_name_decoder(k)]
    sed_decoder_lr, sed_decoder_ids = set_lr(lr_dict["sed_decoder"], sed_decoder_params)
    # query
    query_params = [net.at_query] if isinstance(net.at_query, torch.Tensor) else [p for p in net.at_query.parameters()]
    query_lr, query_ids = set_lr(lr_dict["query"], query_params)
    # head
    head_params = []
    above_ids = backbone_ids + cnn_ids + at_decoder_ids + sed_decoder_ids + query_ids
    for k, p in net.named_parameters():
        if (id(p) not in above_ids):
            head_params.append(p)
            logger.debug("<DEBUG> lr {0} = {1}".format(k, lr_dict["head"]["lr"]))

    head_lr = [{"params": head_params, **lr_dict["head"]}]
    total_lr = backbone_lr + cnn_lr + at_decoder_lr + sed_decoder_lr + query_lr + head_lr
    logger.info("[INFO] Length of backbone params is {0}, lr is {1}".format(len(backbone_ids), backbone_lr[0]["lr"]))
    if len(cnn_ids) > 0:
        logger.info("[INFO] Length of cnn params is {0}, lr is {1}".format(len(cnn_ids), cnn_lr[0]["lr"]))
    logger.info("[INFO] Length of at decoder params is {0}, lr is {1}".format(len(at_decoder_params),
                                                                              at_decoder_lr[0]["lr"]))
    logger.info("[INFO] Length of sed decoder params is {0}, lr is {1}".format(len(sed_decoder_params),
                                                                               sed_decoder_lr[0]["lr"]))
    logger.info("[INFO] Length of query params is {0}, lr is {1}".format(len(query_ids), query_lr[0]["lr"]))
    logger.info("[INFO] Length of head params is {0}, lr is {1}".format(len(head_params), head_lr[0]["lr"]))
    logger.info("Total  Params: %.3f M" % (count_parameters(net, trainable_only=False) * 1e-6))
    logger.info("Total Trainable Params: %.3f M" % (count_parameters(net) * 1e-6))

    return total_lr
