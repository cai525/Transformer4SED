import torch
import torch.nn as nn
import torch.nn.functional as f

from src.models.passt.passt_sed import PaSST_SED
from src.models.cnn import CNN, ResNet, FDY_CNN


class PaSST_CNN(PaSST_SED):

    def __init__(self, passt_sed_param, cnn_param) -> None:
        super().__init__(**passt_sed_param)
        if cnn_param is not None:
            self.init_cnn(cnn_param)
            self.cnn_feat_dim = cnn_param["nb_filters"][-1] if "cnn_1d_dict" not in cnn_param else cnn_param[
                "cnn_1d_dict"]["filters"][-1]
            self.cnn_projector = torch.nn.Linear(self.cnn_feat_dim, self.decoder_dim)
            self.merge_weight = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=self.mlm)
        self.transformer_projector = torch.nn.Linear(self.embed_dim, self.decoder_dim)

    def init_cnn(self, cnn_param: dict):
        self.cnn_name = cnn_param.pop('cnn_name', "base")
        if self.cnn_name == "base":
            self.cnn = CNN(**cnn_param)
        elif self.cnn_name == "resnet":
            self.cnn = ResNet(**cnn_param)
        elif self.cnn_name == "FDY-CNN":
            self.cnn = FDY_CNN(**cnn_param)
        else:
            raise NotImplementedError("Unknown cnn encoder name")

    def forward(self, input, encoder_win=False, mix_rate=0.5, win_param=[512, 49], temp_w=1, pad_mask=None):
        # input shape B,F,T i.e. 10x128x1000
        other_dict = {}
        x = input.unsqueeze(1)

        #patch-wise context modeling
        passt_out_dict = self.patch_transformer(x)
        # pooling
        x = self.f_pool(passt_out_dict)
        x = torch.cat((x, x[:, -1, :].unsqueeze(1)), dim=1)
        x = self.interpolate_module(x, self.decode_ratio)
        if encoder_win:
            from src.models.passt.passt_win import PasstWithSlide
            # merge the global feature map and the local feature map
            slide_window_model = PasstWithSlide(net=self, win_param=win_param)
            x_local = self.slide_window_layer(slide_window_model(input, emb_len=x.shape[1]))
            x = mix_rate * x_local + (1 - mix_rate) * x

        # merge CNN's feature
        if hasattr(self, "cnn"):
            cnn_feat = self.cnn(input.transpose(1, 2).unsqueeze(1))
            _, cnn_channel, cnn_t, cnn_f = cnn_feat.shape
            assert cnn_channel == self.cnn_feat_dim
            assert cnn_f == 1
            cnn_feat = f.interpolate(cnn_feat.squeeze(-1), size=x.shape[1],
                                     mode=self.interpolate_module.mode).transpose(1, 2)  #[B, T, C]

            x = self.transformer_projector(x) +\
                self.merge_weight*self.cnn_projector(cnn_feat)
        else:
            x = self.transformer_projector(x)

        # decoder
        x = self.decoder_step(x, other_dict)

        #  Audio tagging branch
        if self.at_adpater:
            at_embedding = passt_out_dict["frame"].transpose(1, 2)[:, 2:, :]  #B,P,C
            other_dict = self.at_forward(at_embedding, other_dict)

        if self.mlm:
            x = self.mlm_mlp(x)
            return x, other_dict

        # localization
        x = self.classifier(x)

        # other_dict['fbank'] = fbank
        sed_out = torch.sigmoid(x / temp_w)
        if pad_mask is not None:
            sed_out[pad_mask] = 0

        # linear-softmax pool
        at_out = (sed_out * sed_out).sum(dim=1) / sed_out.sum(dim=1)
        at_out = torch.clamp(at_out, 1e-7, 1.)

        return sed_out.transpose(1, 2), at_out, other_dict

    def get_model_name(self):
        return "PaSST_CNN"
