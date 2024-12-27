from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SEDModel(nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()
        pass

    @abstractmethod
    def get_feature_extractor(self):
        pass

    @abstractmethod
    def get_backbone_encoder(self):
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def get_backbone_upsample_ratio(self):
        pass
