import logging

import torch
import torch.nn as nn

from src.models.htsat.htsat_cnn import CLAPAudioCfp, create_htsat_model, MLP


class CLAP_SED(nn.Module):

    def __init__(self, embed_dim, pretrain_model_path, text_query_path) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # load audio branch
        self.backbone = create_htsat_model(CLAPAudioCfp)
        self.backbone.load_state_dict(torch.load(pretrain_model_path))
        # projector
        self.audio_projector = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        # load text query
        logging.info("<INFO> load query embeddings from path {}".format(text_query_path))
        text_query = torch.load(text_query_path)
        self.register_buffer("text_query", text_query)
        self.text_projector = MLP(text_query.shape[-1], self.embed_dim, self.embed_dim, 2)
        self.class_num = text_query.shape[0]

    def forward(self, input, query=None, temp_w=1, pad_mask=None, **kwargs):
        other_dict = {}
        htast_feat = self.backbone(input.clone())['fine_grained_embedding'].squeeze(1, 2)

        audio_embedding = self.audio_projector(htast_feat)
        text_embedding = self.text_projector(query if query is not None else self.text_query)\
            .expand(input.shape[0], -1, -1)
        logit = torch.einsum("bqc, bct -> bqt", text_embedding, audio_embedding.transpose(1, 2)).transpose(1, 2)
        other_dict['logit'] = ['logit']
        sed_out = nn.functional.sigmoid(logit / temp_w)
        if pad_mask is not None:
            sed_out[pad_mask] = 0
        weak_out = (sed_out * sed_out).sum(dim=1) / sed_out.sum(dim=1)
        weak_out = torch.clamp(weak_out, 1e-7, 1.)
        return sed_out.transpose(1, 2), weak_out, other_dict

    def get_backbone(self):
        return self.backbone

    def get_feature_extractor(self, mixup_lambda):
        return lambda wav: self.backbone.wav2mel(wav, mixup_lambda)

    def get_model_name(self):
        return 'CLAP_SED'
