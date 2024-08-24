import torch
import torch.nn as nn
import torch.nn.functional as f

from src.models.passt.passt_sed import PaSST_SED
from src.models.cnn import CNN
from src.models.transformer.mask import MlmModule


class PaSST_CNN(PaSST_SED):

    def __init__(self, passt_sed_param, cnn_param) -> None:
        super().__init__(**passt_sed_param)
        self.cnn = CNN(**cnn_param)
        self.cnn_feat_dim = cnn_param["nb_filters"][-1]
        self.cnn_projector = torch.nn.Linear(self.cnn_feat_dim, self.decoder_dim)
        self.transformer_projector = torch.nn.Linear(self.embed_dim, self.decoder_dim)
        self.merge_weight = torch.nn.Parameter(torch.Tensor([0.5]))

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
        cnn_feat = self.cnn(input.transpose(1, 2).unsqueeze(1))
        _, cnn_channel, cnn_t, cnn_f = cnn_feat.shape
        assert cnn_channel == self.cnn_feat_dim
        assert cnn_f == 1
        cnn_feat = f.interpolate(cnn_feat.squeeze(-1), scale_factor=4,
                                 mode=self.interpolate_module.mode).transpose(1, 2)  #[B, T, C]

        x = self.transformer_projector(x) +\
            self.merge_weight*self.cnn_projector(cnn_feat)

        if self.mlm:
            other_dict["frame_before_mask"] = x
            x, mask_id_seq = self.mlm_tool.setence_mask(x, self.mask_token)
            other_dict["mask_id_seq"] = mask_id_seq

        # decoder
        x = self.decoder_step(x, other_dict)

        #  Audio tagging branch
        if self.at_adpater:
            if self.passt_at_feature_layer == "frame_mean":
                at_embedding = passt_out_dict["frame"].transpose(1, 2)[:, 2:, :]  #B,P,C
            elif self.passt_at_feature_layer == 'token_mean':
                at_embedding = passt_out_dict["frame"].transpose(1, 2)[:, :2, :],
            else:
                at_embedding = passt_out_dict["layer{}_out".format(self.passt_at_feature_layer)].transpose(1, 2)[:,
                                                                                                                 2:, :]
            other_dict = self.at_forward(at_embedding, other_dict)

        if self.mlm:
            x = self.mlm_mlp(x)
            return x, other_dict
        embed_before_classifier = x

        other_dict["sed_embed"] = embed_before_classifier
        # localization
        x = self.classifier(x)
        other_dict["logit"] = x

        # other_dict['fbank'] = fbank
        sed_out = torch.sigmoid(x / temp_w)
        if pad_mask is not None:
            sed_out[pad_mask] = 0

        # linear-softmax pool
        out = (sed_out * sed_out).sum(dim=1) / sed_out.sum(dim=1)
        at_out = torch.clamp(out, 1e-7, 1.)
        other_dict['sed_logit'] = other_dict["logit"].transpose(1, 2)
        other_dict['at_logit'] = torch.logit(at_out)

        return sed_out.transpose(1, 2), at_out, other_dict

    def get_model_name(self):
        return "PaSST_CNN"
