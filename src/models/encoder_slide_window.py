from abc import ABC, abstractclassmethod

import torch

from src.models.sed_model import SEDModel


class EncoderSlideWindow(ABC):

    def __init__(self, net: SEDModel, win_param=[512, 31]):
        super().__init__()
        self.net = net
        self.out_dim = self.net.get_encoder_output_dim()
        self.win = win_param

    def __call__(self, input: torch.Tensor, emb_len):
        device = input.device
        batch_size, _, input_len = input.shape
        scale = emb_len / input_len

        # decode output shape :[batch, frame, channel]
        win_width, step = self.win
        embedding = torch.zeros([batch_size, emb_len, self.out_dim]).to(device)
        accumlator = torch.zeros([batch_size, emb_len, self.out_dim]).to(device)

        for w_left in range(0, input_len + step - win_width, step):
            w_right = min(w_left + win_width, input_len)
            out_left = round(w_left * scale)
            out = self.encode(input[:, :, w_left:w_right])
            out_right = int(min(emb_len, out_left + out.shape[1]))
            embedding[:, out_left:out_right, :] += out
            accumlator[:, out_left:out_right, :] += 1

        embedding /= accumlator
        embedding[torch.isnan(embedding)] = 0
        return embedding

    @abstractclassmethod
    def encode(self, input: torch.Tensor) -> torch.Tensor:
        pass
