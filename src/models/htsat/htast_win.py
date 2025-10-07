import torch
import torch.nn.functional as F

from src.models.encoder_slide_window import EncoderSlideWindow
from src.models.htsat.htsat_cnn import HTSAT_CNN


class HtsatSlideWindow(EncoderSlideWindow):

    def __init__(self, net: HTSAT_CNN, win_param=[512, 29]):
        super().__init__(net, win_param)

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        input = input.unsqueeze(1)
        output = self.net.get_backbone()(input)['fine_grained_embedding'].transpose(1, 2)
        if self.net.get_backbone_upsample_ratio() != 1:
            output = F.interpolate(
                output,
                scale_factor=self.net.get_backbone_upsample_ratio(),
                mode='linear',
            )
            output = output.transpose(1, 2)  # B,C,T->B,T,C
        return output
