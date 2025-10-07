import torch
import torch.nn as nn
import torch.nn.functional as f

from src.models.pooling import AttentionPooling
from src.models.sed_model import SEDModel
from src.models.passt.passt_feature_extraction import PasstFeatureExtractor
from src.models.passt.passt_sed import InterpolateModule
from src.models.transformer_decoder import TransformerDecoder, TransformerXLDecoder, ConformerDecoder
from src.models.transformer.mask import MlmModule
from src.models.detect_any_sound.at_adapter import QueryBasedAudioTaggingDecoder
from src.models.cnn import CNN


class TwoBranchBaseline(SEDModel):

    def __init__(
        self,
        cnn_param,
        backbone_param={
            "embed_dim": 768,
            "passt_feature_layer": 10,
            "load_pretrained_model": True,
            "lora_config": None,
        },
        mlm_dict=None,
        backbone_upsample_ratio=10,
        decoder_dim=768,
        num_heads=12,
        decoder='gru',
        decoder_layer_num=2,
        decoder_pos_emd_len=1000,
        decoder_expand_rate=1,
        at_decoder_layer=0,
        class_num=10,
    ) -> None:
        super().__init__()
        # initial cnn
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
        self._init_SED_decoder(decoder_layer_num, decoder_pos_emd_len)
        # init AT decoders
        if at_decoder_layer > 0:
            self._init_AT_head(at_decoder_layer)
        # initial joint layers
        self._init_joint_layers(transformer_embed_dim=backbone_param["embed_dim"])

    def _init_joint_layers(self, transformer_embed_dim):
        self.interpolate_module = InterpolateModule(mode='linear')
        self.f_pool_module = AttentionPooling(embed_dim=transformer_embed_dim, num_head=6)
        self.cnn_projector = torch.nn.Linear(self.cnn_feat_dim, self.decoder_dim)
        self.transformer_projector = torch.nn.Linear(transformer_embed_dim, self.decoder_dim)
        self.merge_weight = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=self.mlm_dict is not None)
        self.norm_before_pool = nn.LayerNorm(transformer_embed_dim)
        self.norm_after_merge = nn.LayerNorm(self.decoder_dim)

    def _init_SED_decoder(self, decoder_layer_num, decoder_pos_emd_len):
        self.decoder_layer_num = decoder_layer_num
        if self.decoder_name == 'gru':
            self.sed_decoder = nn.GRU(self.decoder_dim,
                                      self.decoder_dim,
                                      bidirectional=True,
                                      batch_first=True,
                                      num_layers=decoder_layer_num,
                                      dropout=0)
            self.sed_head = nn.Linear(2 * self.decoder_dim, self.class_num)

        elif self.decoder_name == "transformer":
            self.sed_decoder = TransformerDecoder(
                input_dim=self.decoder_dim,
                decoder_layer_num=decoder_layer_num,
                pos_embed_strategy="sincos",
                seq_len=decoder_pos_emd_len,
                mlp_ratio=self.decoder_expand_rate,
                num_heads=self.num_heads,
            )
            self.sed_head = nn.Linear(self.decoder_dim, self.class_num)
        elif self.decoder_name == "transformerXL":
            self.sed_decoder = TransformerXLDecoder(
                input_dim=self.decoder_dim,
                seq_len=decoder_pos_emd_len,
                decoder_layer_num=decoder_layer_num,
                mlp_ratio=self.decoder_expand_rate,
                num_heads=self.num_heads,
            )
            self.sed_head = nn.Linear(self.decoder_dim, self.class_num)
        elif self.decoder_name == "conformer":
            self.sed_decoder = ConformerDecoder(
                input_dim=self.decoder_dim,
                seq_len=decoder_pos_emd_len,
                decoder_layer_num=decoder_layer_num,
                mlp_ratio=self.decoder_expand_rate,
                num_heads=self.num_heads,
            )
            self.sed_head = nn.Linear(self.decoder_dim, self.class_num)
        elif self.decoder_name == 'no':
            self.sed_decoder = torch.nn.Identity()
            self.sed_head = nn.Linear(self.decoder_dim, self.class_num)
        else:
            raise ValueError('invalid decoder block')

    def _init_AT_head(self, at_decoder_layer):
        self.at_query = nn.Parameter(torch.zeros(self.class_num, self.decoder_dim))
        torch.nn.init.normal_(self.at_query, std=.02)
        self.at_decoder = QueryBasedAudioTaggingDecoder(
            n_layers=at_decoder_layer,
            nhead=self.num_heads,
            d_model=self.decoder_dim,
            dim_ffn=self.decoder_dim * self.decoder_expand_rate,
        )
        self.at_heads = nn.ModuleList([nn.Linear(self.decoder_dim, 1) for i in range(self.class_num)])

    def _init_mlm(self, device, mlm_dict=dict()):
        out_dim = mlm_dict["out_dim"]
        self.mlm_tool = MlmModule(device=device, **mlm_dict)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.mlm_mlp = nn.Sequential(torch.nn.Linear(self.decoder_dim, self.decoder_dim), torch.nn.GELU(),
                                     torch.nn.Linear(self.decoder_dim, out_dim))

    def _init_transformer_backbone(self, passt_param):
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
            "embed_dim": passt_param["embed_dim"],
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4,
            "qkv_bias": True,
            "distilled": True,
        }
        if passt_param["lora_config"] is not None:
            passt_params_dict["lora_config"] = passt_param["lora_config"]
            from src.models.passt.passt_lora import PaSST
        else:
            from src.models.passt.passt import PaSST

        if passt_param["load_pretrained_model"]:

            self.backbone = PaSST(**passt_params_dict)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            sd = torch.load('./pretrained_model/passt-s-f128-p16-s10-ap.476-swa.pt', map_location=device)
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

    def at_branch(self, at_feat):
        at_feat = self.at_decoder(
            feat_encoder=at_feat,
            queries=self.at_query.expand(at_feat.shape[0], -1, -1),
        )
        at_logit_list = []
        for i, at_head in enumerate(self.at_heads):
            at_logit_list.append(at_head(at_feat[:, i, :]))
        at_logit = torch.concat(at_logit_list, dim=-1)
        at_out = torch.sigmoid(at_logit)
        return at_out

    def forward(self, input, encoder_win=False, mix_rate=0.5, win_param=[512, 49], temp_w=1, pad_mask=None):
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
        cnn_feat = self.cnn(input.transpose(1, 2).unsqueeze(1))
        _, cnn_channel, cnn_t, cnn_f = cnn_feat.shape
        assert cnn_channel == self.cnn_feat_dim
        assert cnn_f == 1
        cnn_feat = f.interpolate(cnn_feat.squeeze(-1), size=x.shape[1],
                                 mode=self.interpolate_module.mode).transpose(1, 2)  #[B, T, C]

        x = self.transformer_projector(x) +\
            self.merge_weight*self.cnn_projector(cnn_feat)
        x = self.norm_after_merge(x)

        other_dict["at_out_specific"] = self.at_branch(x.clone())

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
        sed_out = torch.sigmoid(x / temp_w)
        if pad_mask is not None:
            sed_out[pad_mask] = 0

        # linear-softmax pool
        out = (sed_out * sed_out).sum(dim=1) / sed_out.sum(dim=1)
        at_out = torch.clamp(out, 1e-7, 1.)
        return sed_out.transpose(1, 2), at_out, other_dict

    def get_model_name(self):
        return "Maskformer"

    def get_feature_extractor(self):
        return self.mel_trans

    def get_backbone(self):
        return self.backbone

    def get_backbone_upsample_ratio(self):
        return self.backbone_upsample_ratio
