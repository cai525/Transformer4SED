import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.htsat.htsat import CLAPAudioCfp, create_htsat_model
from src.models.sed_model import SEDModel
from src.models.passt.passt_sed import InterpolateModule
from src.models.transformer_decoder import TransformerXLDecoder, ConformerDecoder
from src.models.transformer.mask import MlmModule
from src.models.cnn import CNN


class HTSAT_CNN(SEDModel):

    def __init__(
        self,
        cnn_param,
        backbone_param={
            "embed_dim": 768,
            "passt_feature_layer": 10,
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
        self._init_SED_decoder(decoder_layer_num, decoder_pos_emd_len)
        # initial joint layers
        self._init_joint_layers(transformer_embed_dim=backbone_param["embed_dim"])

    def _init_joint_layers(self, transformer_embed_dim):
        self.interpolate_module = InterpolateModule(mode='linear')
        if hasattr(self, "cnn"):
            self.cnn_projector = torch.nn.Linear(self.cnn_feat_dim, self.decoder_dim)
            self.merge_weight = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=self.mlm_dict is not None)
        self.transformer_projector = torch.nn.Linear(transformer_embed_dim, self.decoder_dim)
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

        self.sed_head = nn.Linear(self.decoder_dim, self.class_num)

    def _init_mlm(self, device, mlm_dict=dict()):
        out_dim = mlm_dict["out_dim"]
        self.mlm_tool = MlmModule(device=device, **mlm_dict)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.mlm_mlp = nn.Sequential(
            torch.nn.Linear(self.decoder_dim, self.decoder_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.decoder_dim, out_dim),
        )

    def _init_transformer_backbone(self, backbone_param):
        self.backbone = create_htsat_model(CLAPAudioCfp)
        self.backbone.load_state_dict(torch.load(backbone_param["pretrain_model_path"]))

    def _init_cnn(self, cnn_param):
        self.cnn = CNN(**cnn_param)
        self.cnn_feat_dim = cnn_param["nb_filters"][-1] if "cnn_1d_dict" not in cnn_param else \
            cnn_param["cnn_1d_dict"]["filters"][-1]

    def sed_branch(self, x):
        # time series model
        if self.decoder_name == 'gru':
            self.sed_decoder.flatten_parameters()
            x, _ = self.sed_decoder(x)
        else:
            x = self.sed_decoder(x)

        return x

    def forward(
        self,
        input,
        encoder_win=False,
        mix_rate=0.5,
        win_param=[512, 49],
        temp_w=0.1,
        pad_mask=None,
    ):
        other_dict = {}
        # shape of input (batch_size, 1, time_steps, mel_bins)
        # shape of htast_output is (Batch, Frames:32, Channel:768)
        htast_feat = self.backbone(input.clone())['fine_grained_embedding'].squeeze(1, 2)
        # pooling
        x = self.interpolate_module(htast_feat, self.backbone_upsample_ratio)
        # TODO: add implementation for sliding-window strategy
        # merge CNN's feature
        if hasattr(self, "cnn"):
            cnn_feat = self.cnn(input)
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
        other_dict['logit'] = x.transpose(1, 2)
        sed_out = torch.sigmoid(x / temp_w)

        if pad_mask is not None:
            sed_out[pad_mask] = 0

        # linear-softmax pool
        sed_out = torch.clamp(sed_out, 1e-7, 1.)
        weak_out = (sed_out * sed_out).sum(dim=1) / sed_out.sum(dim=1)
        weak_out = torch.clamp(weak_out, 1e-7, 1.)
        return sed_out.transpose(1, 2), weak_out, other_dict

    def get_model_name(self):
        return "HTSAT_CNN"

    def get_feature_extractor(self, mixup_lambda):
        return lambda wav: self.backbone.wav2mel(wav, mixup_lambda)

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
