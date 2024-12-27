import importlib

import torch
import torch.nn as nn
from torch.nn import BCELoss, MSELoss

from src.functional.loss.sup_con_loss import SupConLoss
from src.functional.loss.info_nce import InfoNCE


def buildInstance(moduleName, className, **kwargs):
    inputModule = importlib.import_module(moduleName)
    instanceCls = getattr(inputModule, className)
    instance = instanceCls(**kwargs)
    return instance


def loss_function_factory(name: str, kwargs):
    if kwargs:
        return buildInstance("src.functional.loss", name, **kwargs)
    else:
        return buildInstance("src.functional.loss", name)


class AslLoss(nn.Module):

    def __init__(self, rp, rn, margin):
        super(AslLoss, self).__init__()
        self.rp = rp
        self.rn = rn
        self.margin = margin

    def forward(self, pred, target):
        pred_m = torch.maximum(pred - self.margin, torch.zeros_like(pred))
        losses = -(((1 - pred)**self.rp) * target * torch.clamp_min(torch.log(pred), -100) + (pred_m**self.rn) *
                   (1 - target) * torch.clamp_min(torch.log(1 - pred_m), -100))
        return torch.mean(losses)


class ReweightedASL(nn.Module):

    def __init__(self, rp, rn, margin, weight: list):
        super().__init__()
        self.rp = rp
        self.rn = rn
        self.margin = margin
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """ Compute the reweighting ASL loss. Notice that the last dimension of input must be the number
        of classes
        """

        pred_m, weight = torch.maximum(pred - self.margin, torch.zeros_like(pred)), torch.tensor(self.weight).to(pred)
        losses = -weight * (((1 - pred)**self.rp) * target * torch.clamp_min(torch.log(pred), -100) +
                            (pred_m**self.rn) * (1 - target) * torch.clamp_min(torch.log(1 - pred_m), -100))
        return torch.mean(losses)


class AsymmetricalFocalLoss(nn.Module):

    def __init__(self, gamma=0, zeta=0):
        super(AsymmetricalFocalLoss, self).__init__()
        self.gamma = gamma  # balancing between classes
        self.zeta = zeta  # balancing between active/inactive frames

    def forward(self, pred, target):
        losses = -(((1 - pred)**self.gamma) * target * torch.clamp_min(torch.log(pred), -100) + (pred**self.zeta) *
                   (1 - target) * torch.clamp_min(torch.log(1 - pred), -100))
        return torch.mean(losses)
