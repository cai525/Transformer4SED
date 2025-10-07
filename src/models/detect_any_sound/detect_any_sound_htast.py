from typing import Union, List, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.detect_any_sound.detect_any_sound import DetectAnySoundModel
from src.models.htsat.htsat import CLAPAudioCfp, create_htsat_model
from src.models.passt.passt_sed import InterpolateModule


class DASM_HTSAT(DetectAnySoundModel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _init_joint_layers(self, transformer_embed_dim):
        self.interpolate_module = InterpolateModule(mode='linear')
        if hasattr(self, "cnn"):
            self.cnn_projector = torch.nn.Linear(self.cnn_feat_dim, self.decoder_dim)
            self.merge_weight = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=self.mlm_dict is not None)
        self.transformer_projector = torch.nn.Linear(transformer_embed_dim, self.decoder_dim)
        self.at_projector = torch.nn.Linear(transformer_embed_dim, self.decoder_dim)
        self.norm_before_pool = nn.LayerNorm(transformer_embed_dim)
        self.norm_after_merge = nn.LayerNorm(self.decoder_dim)

    def _init_transformer_backbone(self, backbone_param):
        self.backbone = create_htsat_model(CLAPAudioCfp)
        self.backbone.load_state_dict(torch.load(backbone_param["pretrain_model_path"]))

    def f_pool(self, passt_out_dict):
        raise NotImplementedError("Method 'f_pool' is not supported by Maskformer_HTAST")

    def forward(
        self,
        input,
        encoder_win=False,
        mix_rate=0.5,
        win_param=[512, 49],
        temp_w=0.1,
        pad_mask=None,
        query: Union[torch.Tensor, list] = None,
        query_type: Optional[Literal['text', 'audio']] = None,
        tgt_mask=None,
    ):
        other_dict = {}
        # shape of input (batch_size, 1, time_steps, mel_bins)
        # shape of htast_output is (Batch, Frames:32, Channel:768)
        htast_feat = self.backbone(input.clone())['fine_grained_embedding']
        # pooling
        x = self.interpolate_module(htast_feat, self.backbone_upsample_ratio)
        if encoder_win:
            from src.models.htsat.htast_win import HtsatSlideWindow
            slide_window_model = HtsatSlideWindow(net=self, win_param=win_param)
            x_local = slide_window_model(input.squeeze(1), emb_len=x.shape[1])
            x = mix_rate * x_local + (1 - mix_rate) * x

        # merge CNN's feature
        if hasattr(self, "cnn"):
            cnn_feat = self.cnn(input)
            _, cnn_channel, cnn_t, cnn_f = cnn_feat.shape
            assert cnn_channel == self.cnn_feat_dim
            assert cnn_f == 1
            cnn_feat = F.interpolate(cnn_feat.squeeze(-1), size=x.shape[1],
                                     mode=self.interpolate_module.mode).transpose(1, 2)  #[B, T, C]
            # TODO: add implementation for sliding-window strategy
            x = self.transformer_projector(x) +\
                self.merge_weight*self.cnn_projector(cnn_feat)
        else:
            x = self.transformer_projector(x)

        x = self.norm_after_merge(x)

        # AT decoder
        at_feat = self.at_projector(htast_feat)
        if isinstance(query, torch.Tensor) and query.ndim == 3:
            query = query[0, :, :]
        elif isinstance(query, list) and query[0].ndim == 3:
            for i in range(len(query)):
                query[i] = query[i][0, :, :]
            query = torch.nn.ParameterList(query)

        if tgt_mask is not None and tgt_mask.ndim == 3:
            tgt_mask = tgt_mask[0, :, :]
        other_dict["at_out"], mask_feat = self.at_branch(at_feat, query, query_type, tgt_mask)
        # mask language model
        if self.mlm_dict is not None:
            other_dict["frame_before_mask"] = x
            x, mask_id_seq = self.mlm_tool.setence_mask(x, self.mask_token)
            other_dict["mask_id_seq"] = mask_id_seq

        # sed decoder
        x = self.sed_branch(x)

        if self.mlm_dict is not None:
            x = self.mlm_mlp(x)
            return x, other_dict

        # localization
        x = self.sed_head(x)
        mask_embedding = self.mask_embedding_layer(mask_feat)
        x = torch.einsum("bqc, bct -> bqt", mask_embedding, x.transpose(1, 2)).transpose(1, 2)
        sed_out = torch.sigmoid(x / temp_w) * other_dict["at_out"].unsqueeze(1)  # with clip-level prior
        # sed_out = torch.sigmoid(x / temp_w) # without clip-level prior

        if pad_mask is not None:
            sed_out[pad_mask] = 0

        # linear-softmax pool
        sed_out = torch.clamp(sed_out, 1e-7, 1.)
        weak_out = (sed_out * sed_out).sum(dim=1) / sed_out.sum(dim=1)
        weak_out = torch.clamp(weak_out, 1e-7, 1.)
        return sed_out.transpose(1, 2), weak_out, other_dict

    def get_model_name(self):
        return 'Maskformer_HTSAT'

    def get_feature_extractor(self, mixup_lambda=None):
        return lambda wav: self.backbone.wav2mel(wav, mixup_lambda)
