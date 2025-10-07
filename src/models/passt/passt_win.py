import torch
import torch.nn.functional as F

from src.models.encoder_slide_window import EncoderSlideWindow
from src.models.passt.passt_sed import PaSST_SED


class PasstWithSlide(EncoderSlideWindow):

    def __init__(self, net: PaSST_SED, win_param=[512, 29]):
        super().__init__(net, win_param)

    def default_f_pool(self, passt_out_dict):
        passt_feature = passt_out_dict["layer{}_out".format(self.net.passt_feature_layer)]
        passt_feature = passt_feature.transpose(1, 2)  # N,C,P->N,P,C
        passt_feature = self.net.out_norm(passt_feature)

        B_, P_, C_ = passt_feature.shape
        passt_feature = passt_feature[:, 2:, :].reshape(B_, passt_out_dict['f_dim'], passt_out_dict['t_dim'], C_)
        frameout = torch.mean(passt_feature, dim=1)
        return frameout

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        #patch-wise context modeling
        input = input.unsqueeze(1)
        passt_out_dict = self.net.get_backbone()(input)

        # pooling along the frequency dimension
        if hasattr(self.net, "f_pool"):
            frameout = self.net.f_pool(passt_out_dict)
        else:
            frameout = self.default_f_pool(passt_out_dict)

        # interpolate to fit the decoder dimension
        if self.net.get_backbone_upsample_ratio() != 1:
            frameout = frameout.transpose(1, 2)  # B,T,C->B,C,T
            frameout = F.interpolate(frameout,
                                     scale_factor=self.net.get_backbone_upsample_ratio(),
                                     mode=self.net.interpolate_module.mode)
            frameout = frameout.transpose(1, 2)  # B,C,T->B,T,C
        return frameout
