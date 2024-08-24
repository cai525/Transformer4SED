import torch

from src.models.encoder_slide_window import EncoderSlideWindow
from src.models.atst.atst_sed import AtstSED


class AtstWithSlide(EncoderSlideWindow):

    def __init__(self, net: AtstSED, win_param=[512, 29]):
        super().__init__(net, win_param)

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.net, AtstSED)
        atst_out_dict = self.net.get_encoder()(input, n=self.net.get_encoder_depth() + 1 - min(self.net.feature_layer, self.net.at_feature_layer))
        x = atst_out_dict["feature_map_{layer}".format(layer=self.net.feature_layer)]
        # x.shape = (B, T, C)
        assert x.shape[-1] == self.net.embed_dim
        x = self.net.encoder_out_norm(x)
        # interpolate
        x = self.net.interpolate_module(x, self.net.decode_interpolate_ratio)
        return x
