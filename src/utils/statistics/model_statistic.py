import torch

def count_parameters(model:torch.nn.Module, trainable_only=True):
    """caculate the total parameter number of input model

    Args:
        model: the model to be calculated
        trainable_only (bool): If only count the trainable parameters. Defaults to True.

    Returns: Number of parameters(bit)
    """
    total_params = 0
    for name, parameter in model.named_parameters():
        if trainable_only and (not parameter.requires_grad):
            continue
        param = parameter.numel()
        total_params += param
    return total_params