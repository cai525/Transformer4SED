from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class FeatureExtractor(nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def wav2mel(self, wavs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def normolize(self, mels: torch.Tensor) -> torch.Tensor:
        pass


class SEDModel(nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()
        pass

    @abstractmethod
    def get_feature_extractor(self) -> FeatureExtractor:
        pass

    @abstractmethod
    def get_encoder(self):
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def get_encoder_output_dim(self):
        pass
    
    @abstractmethod
    def get_decode_ratio(self):
        pass
