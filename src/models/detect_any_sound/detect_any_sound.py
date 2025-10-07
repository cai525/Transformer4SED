import logging
from typing import Union, List, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pooling import AttentionPooling
from src.models.sed_model import SEDModel
from src.models.passt.passt_feature_extraction import PasstFeatureExtractor
from src.models.passt.passt_sed import InterpolateModule
from src.models.transformer_decoder import TransformerDecoder, TransformerXLDecoder, ConformerDecoder
from src.models.transformer.mask import MlmModule
from src.models.detect_any_sound.at_adapter import QueryBasedAudioTaggingDecoder
from src.models.cnn import CNN


class Maskformer(SEDModel):

    def __init__(
        self,
        cnn_param,
        backbone_param={
            "embed_dim": 768,
            "passt_feature_layer": 10,
            "pretrain_model_path": None,
            "lora_config": None,
        },
        at_param={
            "at_decoder_layer": 0,
            "query_projector": False,
            "query_dim": 768,
            "out_type": "logit",
            "query": None,
        },
        mlm_dict=None,
        backbone_upsample_ratio=10,
        decoder_dim=768,
        num_heads=12,
        decoder='gru',
        decoder_layer_num=2,
        decoder_pos_emd_len=1000,
        decoder_expand_rate=1,
        class_num=10,
    ) -> None:
        super().__init__()
        # initial cnn
        if cnn_param is not None:
            self._init_cnn(cnn_param)
        # initial transformer backbone
        self.backbone_param = backbone_param
        self._init_transformer_backbone(backbone_param)
        # init SED decoders
        self.decoder_name = decoder
        self.backbone_upsample_ratio = backbone_upsample_ratio
        self.class_num = class_num
        self.decoder_dim = decoder_dim
        self.num_heads = num_heads
        self.decoder_expand_rate = decoder_expand_rate
        self.mlm_dict = mlm_dict
        if mlm_dict is not None:
            self._init_mlm(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), mlm_dict=mlm_dict)
        self._init_SED_decoder(decoder_layer_num, decoder_pos_emd_len, at_param)
        # init AT decoders
        self._init_AT_head(at_param)
        # initial joint layers
        self._init_joint_layers(transformer_embed_dim=backbone_param["embed_dim"])

    def _init_joint_layers(self, transformer_embed_dim):
        self.interpolate_module = InterpolateModule(mode='linear')
        self.f_pool_module = AttentionPooling(embed_dim=transformer_embed_dim, num_head=6)
        if hasattr(self, "cnn"):
            self.cnn_projector = torch.nn.Linear(self.cnn_feat_dim, self.decoder_dim)
            self.merge_weight = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=self.mlm_dict is not None)
        self.transformer_projector = torch.nn.Linear(transformer_embed_dim, self.decoder_dim)
        self.at_projector = torch.nn.Linear(transformer_embed_dim, self.decoder_dim)
        self.norm_before_pool = nn.LayerNorm(transformer_embed_dim)
        self.norm_after_merge = nn.LayerNorm(self.decoder_dim)

    def _init_SED_decoder(self, decoder_layer_num, decoder_pos_emd_len, at_param):
        self.decoder_layer_num = decoder_layer_num
        if self.decoder_name == 'gru':
            self.sed_decoder = nn.GRU(self.decoder_dim,
                                      self.decoder_dim,
                                      bidirectional=True,
                                      batch_first=True,
                                      num_layers=decoder_layer_num,
                                      dropout=0)

        elif self.decoder_name == "transformer":
            self.sed_decoder = TransformerDecoder(
                input_dim=self.decoder_dim,
                decoder_layer_num=decoder_layer_num,
                pos_embed_strategy="sincos",
                seq_len=decoder_pos_emd_len,
                mlp_ratio=self.decoder_expand_rate,
                num_heads=self.num_heads,
            )
        elif self.decoder_name == "transformerXL":
            self.sed_decoder = TransformerXLDecoder(
                input_dim=self.decoder_dim,
                seq_len=decoder_pos_emd_len,
                decoder_layer_num=decoder_layer_num,
                mlp_ratio=self.decoder_expand_rate,
                num_heads=self.num_heads,
            )
        elif self.decoder_name == "conformer":
            self.sed_decoder = ConformerDecoder(
                input_dim=self.decoder_dim,
                seq_len=decoder_pos_emd_len,
                decoder_layer_num=decoder_layer_num,
                mlp_ratio=self.decoder_expand_rate,
                num_heads=self.num_heads,
            )
        elif self.decoder_name == 'no':
            self.sed_decoder = torch.nn.Identity()
        else:
            raise ValueError('invalid decoder block')

        self.mask_embedding_layer = MLP(self.decoder_dim, self.decoder_dim, self.decoder_dim, 3) if at_param["out_type"] \
                                            else nn.Identity()
        self.sed_head = nn.Linear(self.decoder_dim, self.decoder_dim)

    def _init_query(
        self,
        query_projector: bool = False,
        query: Union[str, List[str], torch.Tensor] = None,
        query_dim: Union[int, List[int]] = None,
    ):
        if not query_projector:
            # no query projector. Use learnable queries
            self.at_query = nn.Parameter(torch.zeros(self.class_num, self.decoder_dim))
            torch.nn.init.normal_(self.at_query, std=.02)
        else:
            # when the model has a query projector, it recieves external queries
            if isinstance(query_dim, int):
                self.query_projector = nn.Sequential(nn.Linear(query_dim, self.decoder_dim), nn.GELU())
            else:
                assert isinstance(query_dim, list)
                assert isinstance(query, list)
                self.query_projector = nn.ModuleList()
                for dim in query_dim:
                    self.query_projector.append(
                        nn.Sequential(
                            # MLP(dim, self.decoder_dim, self.decoder_dim, 2),
                            nn.Linear(dim, self.decoder_dim),
                            nn.GELU(),
                        ))
            if query is not None:
                if isinstance(query, list):
                    self.at_query = nn.ParameterList()
                    for q in query:
                        logging.info("<INFO> load query embeddings from path {}".format(q))
                        self.at_query.append(nn.Parameter(torch.load(q)))
                else:
                    if isinstance(query, str):
                        logging.info("<INFO> load query embeddings from path {}".format(query))
                        query = torch.load(query,
                                           map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                    # the argument "query" is not None, which means the external queries are prepared to be loaded
                    assert isinstance(query, torch.Tensor), \
                    "query are expected to be torch.Tensor, but not {0}".format(type(query))
                    # query's shape is (N_queries, Channels), for example (10, 768)
                    assert query.shape[0] == self.class_num
                    self.at_query = nn.Parameter(query)

    def _init_AT_head(self, at_param: dict):
        at_decoder_layer = at_param["at_decoder_layer"]
        self._init_query(at_param["query_projector"], at_param["query"], at_param["query_dim"])
        self.at_decoder = QueryBasedAudioTaggingDecoder(
            n_layers=at_decoder_layer,
            nhead=self.num_heads,
            d_model=self.decoder_dim,
            dim_ffn=self.decoder_dim * self.decoder_expand_rate,
        )
        if at_param["out_type"] == "logit":
            self.at_head = MLP(self.decoder_dim, self.decoder_dim, self.class_num + 1, 2)
        elif at_param["out_type"] == "sigmoid":
            self.at_head = MLP(self.decoder_dim, self.decoder_dim, 1, 2)
        elif at_param["out_type"] == None:
            self.at_head = None
        else:
            raise RuntimeError("Unknown output type for classification branch")

    def _init_mlm(self, device, mlm_dict=dict()):
        out_dim = mlm_dict["out_dim"]
        self.mlm_tool = MlmModule(device=device, **mlm_dict)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.mlm_mlp = nn.Sequential(torch.nn.Linear(self.decoder_dim, self.decoder_dim), torch.nn.GELU(),
                                     torch.nn.Linear(self.decoder_dim, out_dim))

    def _init_transformer_backbone(self, backbone_param):
        self.mel_trans = PasstFeatureExtractor(
            n_mels=128,
            sr=32000,
            win_length=800,
            hopsize=320,
            n_fft=1024,
            htk=False,
            fmin=0.0,
            fmax=None,
            wav_norm=True,
            fmin_aug_range=10,
            fmax_aug_range=2000,
        )
        passt_params_dict = {
            "img_size": (128, 998),
            "patch_size": 16,
            "stride": 10,
            "in_chans": 1,
            "embed_dim": backbone_param["embed_dim"],
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4,
            "qkv_bias": True,
            "distilled": True,
        }
        if backbone_param["lora_config"] is not None:
            passt_params_dict["lora_config"] = backbone_param["lora_config"]
            from src.models.passt.passt_lora import PaSST
        else:
            from src.models.passt.passt import PaSST

        if backbone_param["pretrain_model_path"] is not None:

            self.backbone = PaSST(**passt_params_dict)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            sd = torch.load(backbone_param['pretrain_model_path'], map_location=device)
            self.backbone.load_state_dict(sd, strict=False)

        else:
            self.backbone = PaSST(**passt_params_dict)

    def _init_cnn(self, cnn_param):
        self.cnn = CNN(**cnn_param)
        self.cnn_feat_dim = cnn_param["nb_filters"][-1] if "cnn_1d_dict" not in cnn_param else cnn_param["cnn_1d_dict"][
            "filters"][-1]

    def f_pool(self, passt_out_dict):
        # get feature frome pretrained backbone: shape of passt_feature is (N,C,P)
        encoder_feature = passt_out_dict["layer{}_out".format(self.backbone_param["passt_feature_layer"])][:, :, 2:]
        encoder_feature = encoder_feature.transpose(1, 2)  # N,C,P->N,P,C
        encoder_feature = self.norm_before_pool(encoder_feature)
        B_, P_, C_ = encoder_feature.shape
        encoder_feature = encoder_feature.reshape(B_, passt_out_dict['f_dim'], passt_out_dict['t_dim'], C_)
        encoder_feature = encoder_feature.transpose(1, 2).reshape(B_ * passt_out_dict['t_dim'], passt_out_dict['f_dim'],
                                                                  C_)
        frameout = self.f_pool_module(encoder_feature)
        frameout = frameout.reshape(B_, passt_out_dict['t_dim'], C_)

        return frameout

    def sed_branch(self, x):
        # time series model
        if self.decoder_name == 'gru':
            self.sed_decoder.flatten_parameters()
            x, _ = self.sed_decoder(x)
        else:
            x = self.sed_decoder(x)

        return x

    def at_branch(self, mask_feat, query=None, query_type=None, tgt_mask=None):
        query = self.at_query if query is None else query
        if not isinstance(query, nn.ParameterList):
            if isinstance(self.query_projector, nn.ModuleList):
                # choose the suitable query_projector
                if query_type == 'text':
                    query_projector = self.query_projector[0]
                elif query_type == 'audio':
                    query_projector = self.query_projector[1]
                else:
                    raise RuntimeError('You must assign a type to query. Supported types are \'text\' and \'audio\'.')
            else:
                query_projector = self.query_projector
            query = query_projector(query) if hasattr(self, "query_projector") else query

        else:
            # randomly select a query for each event from mulit-modal queries
            query_list = []
            for q, projector in zip(query, self.query_projector):
                query_list.append(projector(q))
            query = torch.stack(query_list, dim=1)  # shape=[n_queries, n_modals, channel]
            n_queries, n_modals, _ = query.shape
            indices = torch.arange(0, n_queries * n_modals, n_modals) + torch.randint(0, n_modals, (n_queries, ))
            query = query.view(n_queries * n_modals, -1)[indices, :]

        mask_feat = self.at_decoder(
            feat_encoder=mask_feat,
            queries=query.expand(mask_feat.shape[0], -1, -1),
            tgt_mask=tgt_mask,
        )
        if hasattr(self, 'at_head'):
            at_out = self.at_head(mask_feat)
            if at_out.shape[-1] == 1:
                at_out = F.sigmoid(at_out.squeeze(-1))
        else:
            at_out = None
        return at_out, mask_feat

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
        # input shape B,F,T i.e. 10x128x1000
        other_dict = {}
        x = input.clone().unsqueeze(1)

        #patch-wise context modeling
        passt_out_dict = self.backbone(x)
        # pooling
        x = self.f_pool(passt_out_dict)
        x = torch.cat((x, x[:, -1, :].unsqueeze(1)), dim=1)
        x = self.interpolate_module(x, self.backbone_upsample_ratio)
        if encoder_win:
            # siliding windows
            from src.models.passt.passt_win import PasstSlideWindow
            slide_window_model = PasstSlideWindow(net=self, win_param=win_param)
            x_local = slide_window_model(input, emb_len=x.shape[1])
            # merge the global feature map and the local feature map
            x = mix_rate * x_local + (1 - mix_rate) * x

        # merge CNN's feature
        if hasattr(self, "cnn"):
            cnn_feat = self.cnn(input.transpose(1, 2).unsqueeze(1))
            _, cnn_channel, cnn_t, cnn_f = cnn_feat.shape
            assert cnn_channel == self.cnn_feat_dim
            assert cnn_f == 1
            cnn_feat = F.interpolate(cnn_feat.squeeze(-1), size=x.shape[1],
                                     mode=self.interpolate_module.mode).transpose(1, 2)  #[B, T, C]
            x = self.transformer_projector(x) +\
                self.merge_weight*self.cnn_projector(cnn_feat)
        else:
            x = self.transformer_projector(x)
        x = self.norm_after_merge(x)

        # AT decoder
        # at_feat = x.clone()
        at_feat = passt_out_dict["frame"].transpose(1, 2)[:, 2:, :]
        at_feat = self.at_projector(at_feat)
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
        sed_out = torch.sigmoid(x / temp_w) * other_dict["at_out"].unsqueeze(1)
        # sed_out = torch.sigmoid(x / temp_w)

        if pad_mask is not None:
            sed_out[pad_mask] = 0

        # linear-softmax pool
        sed_out = torch.clamp(sed_out, 1e-7, 1.)
        weak_out = (sed_out * sed_out).sum(dim=1) / sed_out.sum(dim=1)
        weak_out = torch.clamp(weak_out, 1e-7, 1.)
        return sed_out.transpose(1, 2), weak_out, other_dict

    def get_model_name(self):
        return "Maskformer"

    def get_feature_extractor(self):
        return self.mel_trans

    def get_backbone(self):
        return self.backbone

    def get_backbone_upsample_ratio(self):
        return self.backbone_upsample_ratio


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
