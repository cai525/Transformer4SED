from abc import ABC, abstractmethod
from logging import Logger
from typing import Union

import torch

from src.codec.encoder import Encoder
from src.models.sed_model import SEDModel
from src.utils import DataParallelWrapper


class Trainer(ABC):

    def __init__(self, net: Union[SEDModel, torch.nn.parallel.DataParallel], train_loader, val_loader, config,
                 optimizer, scheduler, encoder: Encoder, logger: Logger, device) -> None:
        if isinstance(net, torch.nn.parallel.DataParallel):
            assert isinstance(net.module, SEDModel)

        self.net = DataParallelWrapper(net)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.encoder = encoder
        self.device = device
        self.logger = logger

    @abstractmethod
    def train(self):
        raise NotImplementedError("Method 'train' is not implemented in  {}".format(__class__.__name__))

    @abstractmethod
    def validation(self):
        raise NotImplementedError("Method 'validation' is not implemented in  {}".format(__class__.__name__))
