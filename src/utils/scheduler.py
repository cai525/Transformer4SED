import numpy as np
import torch

from src.functional import ramps


class ExponentialWarmup(object):

    def __init__(self, optimizer, max_lr, rampup_length, exponent=-5.0):
        self.optimizer = optimizer
        self.rampup_length = rampup_length
        self.max_lr = max_lr
        self.step_num = 1
        self.exponent = exponent

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _get_lr(self):
        return self.max_lr * self._get_scaling_factor()

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        self._set_lr(lr)

    def _get_scaling_factor(self):
        if self.rampup_length == 0:
            return 1.0
        else:
            current = np.clip(self.step_num, 0.0, self.rampup_length)
            phase = 1.0 - current / self.rampup_length
            return float(np.exp(self.exponent * phase * phase))



class ExponentialDown(object):

    def __init__(self, optimizer:torch.optim.Optimizer, start_iter, total_iter, exponent=-0.5, warmup_iter = 0, warmup_rate=0.1):
        #exponent set bigger (such as -0.5), lr will decay lower, results not stable, sometimes get higher results
        #exponet set -5 is same as company results
        self.optimizer = optimizer
        self.total_iter = total_iter
        self.start_iter = start_iter
        self.step_num = 1
        self.exponet = exponent
        self.lr_init_list = [param_groups["lr"] for param_groups in optimizer.param_groups]
        self.warmup_iter = warmup_iter
        self.warmup_rate = warmup_rate

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _get_scale(self):
        if self.step_num < self.warmup_iter:
            phase = self.step_num / self.warmup_iter
            self.scale = (1-self.warmup_rate)*phase + self.warmup_rate
        elif self.step_num > self.start_iter:
            phase = (self.step_num - self.start_iter) / (self.total_iter - self.start_iter)
            self.scale = float(np.exp(self.exponet * phase * phase))
        else:
            self.scale = 1
        return self.scale

    def _set_lr(self, scale):
        for i, param_groups in enumerate(self.optimizer.param_groups):
            param_groups["lr"] = self.lr_init_list[i] * scale
            
    def step(self):
        self.step_num += 1
        scale = self._get_scale()
        self._set_lr(scale)


class CosineDown(object):

    def __init__(
        self,
        optimizer,
        max_lr,
        rampup_iter,
        total_iter,
    ):
        self.optimizer = optimizer
        self.total_iter = total_iter
        self.rampup_iter = rampup_iter
        self.step_num = 1
        self.max_lr = max_lr

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _get_scaling_factor(self):
        if self.step_num < self.rampup_iter:
            rampup_value = ramps.sigmoid_rampup(self.step_num, self.rampup_iter)
            rampdown_value = 1.0
        else:
            rampup_value = 1.0
            rampdown_value = ramps.cosine_rampdown((self.step_num - self.rampup_iter), self.total_iter)
        self.scale = rampup_value * rampdown_value
        return self.scale

    def _get_ramupup_value(self):
        if self.step_num < self.rampup_iter:
            rampup_value = ramps.sigmoid_rampup(self.step_num, self.rampup_iter)
        else:
            rampup_value = 1.0

        return rampup_value

    def _set_lr(self, scale):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.max_lr * scale

    def step(self):
        self.step_num += 1
        scale = self._get_scaling_factor()
        self._set_lr(scale)


def update_ema(net, ema_net, step, ema_factor):
    # update EMA model
    alpha = min(1 - 1 / step, ema_factor)
    for ema_params, params in zip(ema_net.parameters(), net.parameters()):
        ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)
    return ema_net


