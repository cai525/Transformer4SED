from src.models.htsat.htsat_cnn import HTSAT_CNN
from src.utils.statistics.model_statistic import count_parameters
from recipes.audioset_strong.detect_any_sound.htsat.lr_set import set_backbone_lr, set_lr


def get_param_lr(net: HTSAT_CNN, configs, logger):
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

    # sed_decoder (with parameters of joint module)
    def check_tensor_name_decoder(tensor_name: str) -> bool:
        keyword_list = [
            "sed_decoder",
            "fpn.",
            "norm_list",
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

    # head
    head_params = []
    above_ids = backbone_ids + cnn_ids + sed_decoder_ids
    for k, p in net.named_parameters():
        if (id(p) not in above_ids):
            head_params.append(p)
            logger.debug("<DEBUG> lr {0} = {1}".format(k, lr_dict["head"]["lr"]))

    head_lr = [{"params": head_params, **lr_dict["head"]}]
    total_lr = backbone_lr + cnn_lr + sed_decoder_lr + head_lr
    logger.info("[INFO] Length of passt params is {0}, lr is {1}".format(len(backbone_ids), backbone_lr[0]["lr"]))
    if len(cnn_ids) > 0:
        logger.info("[INFO] Length of cnn params is {0}, lr is {1}".format(len(cnn_ids), cnn_lr[0]["lr"]))
    logger.info("[INFO] Length of sed decoder params is {0}, lr is {1}".format(len(sed_decoder_params),
                                                                               sed_decoder_lr[0]["lr"]))
    logger.info("[INFO] Length of head params is {0}, lr is {1}".format(len(head_params), head_lr[0]["lr"]))
    logger.info("Total  Params: %.3f M" % (count_parameters(net, trainable_only=False) * 1e-6))
    logger.info("Total Trainable Params: %.3f M" % (count_parameters(net) * 1e-6))
    return total_lr
