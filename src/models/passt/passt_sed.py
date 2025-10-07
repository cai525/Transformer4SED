import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from src.models.sed_model import SEDModel
from src.models.pooling import FrequencyWiseTranformerPooling, AttentionPooling
from src.models.passt.passt_feature_extraction import PasstFeatureExtractor
from src.models.passt.tool_block import GAP
from src.models.transformer_decoder import TransformerDecoder, TransformerXLDecoder, ConformerDecoder
from src.models.transformer.mask import MlmModule


class InterpolateModule(nn.Module):
    """ Used for Interpolate
    The module is defined as a module becasue we need to register the hook function
    here to visualize the features.
    """

    def __init__(self, mode="linear") -> None:
        super().__init__()
        self.mode = mode

    def forward(self, seq: torch.Tensor, ratio):
        """
        Args:
            seq (B, T, C): sequence to be interpolated
            ratio: interpolate ratio
        """
        if ratio == 1:
            return seq
        seq = seq.transpose(1, 2)  # B,T,C->B,C,T
        seq = torch.nn.functional.interpolate(seq, scale_factor=ratio, mode=self.mode)
        seq = seq.transpose(1, 2)  # B,C,T->B,T,C
        return seq


class PaSST_SED(SEDModel):

    def __init__(self,
                 decode_ratio=10,
                 interpolate_mode='linear',
                 passt_feature_layer=10,
                 embed_dim=768,
                 decoder_dim=768,
                 f_pool='mean_pool',
                 s_patchout_f=0,
                 s_patchout_t=0,
                 decoder='gru',
                 decoder_layer_num=2,
                 decoder_pos_emd_len=1000,
                 load_pretrained_model=True,
                 class_num=10,
                 at_adapter=False,
                 decoder_win_len=None,
                 mlm=False,
                 mlm_dict=dict(),
                 lora_config=None):

        super().__init__()
        # block = getattr(BLOCK, block_name)
        #setting mel feature extrator
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

        #passt
        assert s_patchout_t == 0, "SED task do not support temporal patchout"

        passt_params_dict = {
            "u_patchout": 0,
            "s_patchout_t": s_patchout_t,
            "s_patchout_f": s_patchout_f,
            "img_size": (128, 998),
            "patch_size": 16,
            "stride": 10,
            "in_chans": 1,
            "num_classes": 527,
            "embed_dim": embed_dim,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4,
            "qkv_bias": True,
            "representation_size": None,
            "distilled": True,
            "drop_rate": 0,
            "attn_drop_rate": 0,
            "drop_path_rate": 0.,
            "norm_layer": None,
            "act_layer": None,
            "weight_init": '',
        }
        if lora_config != None:
            passt_params_dict["lora_config"] = lora_config
            from src.models.passt.passt_lora import PaSST
        else:
            from src.models.passt.passt import PaSST

        if load_pretrained_model:

            self.backbone = PaSST(**passt_params_dict)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            sd = torch.load('./pretrained_model/passt-s-f128-p16-s10-ap.476-swa.pt', map_location=device)
            self.backbone.load_state_dict(sd, strict=False)

        else:
            self.backbone = PaSST(**passt_params_dict)

        self.f_pool_name = f_pool
        self.passt_feature_layer = passt_feature_layer
        self.decoder_name = decoder
        self.decode_ratio = decode_ratio
        self.class_num = class_num
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim

        self.out_norm = nn.LayerNorm(embed_dim)

        self.init_f_pool(pool_name=f_pool, embed_dim=embed_dim)
        self.interpolate_module = InterpolateModule(mode=interpolate_mode)
        self.slide_window_layer = nn.Identity()  # Just for capture the output of the sliding windows outside the class
        self.mlm = mlm
        if mlm:
            self.init_mlm(device=device, mlm_dict=mlm_dict)

        self.init_decoder(decoder_win_len, decoder_layer_num, decoder_pos_emd_len)

        self.at_adpater = at_adapter
        if self.at_adpater:
            self.at_adpater = nn.Sequential(
                AttentionPooling(embed_dim=embed_dim, num_head=12),
                nn.Linear(embed_dim, class_num),
            )

    def init_f_pool(self, pool_name, embed_dim):
        if pool_name == 'mean_pool':
            pass
        elif pool_name == 'frequency_wise_tranformer_encoder':
            self.f_pool_module = FrequencyWiseTranformerPooling(embed_dim)
        elif pool_name == "attention":
            self.f_pool_module = AttentionPooling(embed_dim=embed_dim, num_head=6)
        else:
            raise NotImplementedError("pool method {0} hasn't been implemneted yet".format(pool_name))

    def init_decoder(self, win_len, decoder_layer_num, decoder_pos_emd_len):
        self.decoder_layer_num = decoder_layer_num
        if self.decoder_name == 'gru':
            self.decoder = nn.GRU(self.decoder_dim,
                                  self.decoder_dim,
                                  bidirectional=True,
                                  batch_first=True,
                                  num_layers=decoder_layer_num,
                                  dropout=0)
            self.classifier = nn.Linear(2 * self.decoder_dim, self.class_num)

        elif self.decoder_name == "transformer":
            self.decoder = TransformerDecoder(input_dim=self.decoder_dim,
                                              decoder_layer_num=decoder_layer_num,
                                              pos_embed_strategy="sincos",
                                              seq_len=decoder_pos_emd_len)
            self.classifier = nn.Linear(self.decoder_dim, self.class_num)
        elif self.decoder_name == "transformerXL":
            self.decoder = TransformerXLDecoder(input_dim=self.decoder_dim,
                                                seq_len=decoder_pos_emd_len,
                                                window_len=win_len,
                                                decoder_layer_num=decoder_layer_num)
            self.classifier = nn.Linear(self.decoder_dim, self.class_num)
        elif self.decoder_name == "conformer":
            self.decoder = ConformerDecoder(input_dim=self.decoder_dim,
                                            seq_len=decoder_pos_emd_len,
                                            window_len=win_len,
                                            decoder_layer_num=decoder_layer_num)
            self.classifier = nn.Linear(self.decoder_dim, self.class_num)
        elif self.decoder_name == 'no':
            self.decoder = torch.nn.Identity()
            self.classifier = nn.Linear(self.decoder_dim, self.class_num)
        else:
            raise ValueError('invalid decoder block')

    def init_mlm(self, device, mlm_dict=dict()):
        out_dim = mlm_dict["out_dim"]
        self.mlm_tool = MlmModule(device=device, **mlm_dict)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.mlm_mlp = nn.Sequential(torch.nn.Linear(self.decoder_dim, self.decoder_dim), torch.nn.GELU(),
                                     torch.nn.Linear(self.decoder_dim, out_dim))

    def f_pool(self, passt_out_dict):
        # get feature frome pretrained backbone: shape of passt_feature is (N,C,P)
        assert isinstance(self.passt_feature_layer, int)
        encoder_feature = passt_out_dict["layer{}_out".format(self.passt_feature_layer)][:, :, 2:]
        encoder_feature = encoder_feature.transpose(1, 2)  # N,C,P->N,P,C
        encoder_feature = self.out_norm(encoder_feature)
        B_, P_, C_ = encoder_feature.shape
        encoder_feature = encoder_feature.reshape(B_, passt_out_dict['f_dim'], passt_out_dict['t_dim'], C_)

        # pooling along the frequency dimension
        if self.f_pool_name == 'mean_pool':
            frameout = torch.mean(encoder_feature, dim=1)
        elif hasattr(self, "f_pool_module"):
            encoder_feature = encoder_feature.transpose(1, 2).reshape(B_ * passt_out_dict['t_dim'],
                                                                      passt_out_dict['f_dim'], C_)
            frameout = self.f_pool_module(encoder_feature)
            frameout = frameout.reshape(B_, passt_out_dict['t_dim'], C_)
        else:
            raise NotImplementedError("pool method {0} hasn't been implemneted yet".format(self.f_pool_name))
        return frameout

    def decoder_step(self, x, other_dict):
        # mask language model
        other_dict["frame_before_mask"] = x
        if self.mlm:
            x, mask_id_seq = self.mlm_tool.setence_mask(x, self.mask_token)
            other_dict["mask_id_seq"] = mask_id_seq

        # time series model
        if self.decoder_name == 'gru':
            self.decoder.flatten_parameters()
            x, _ = self.decoder(x)
        else:
            x = self.decoder(x)

        return x

    def at_forward(self, at_embedding, other_dict):
        at_adapter_logit = self.at_adpater(at_embedding)
        at_adapter_out = torch.sigmoid(at_adapter_logit)
        other_dict['at_out'] = at_adapter_out
        return other_dict

    def forward(self,
                input: torch.Tensor,
                encoder_win=False,
                mix_rate=0.5,
                win_param=[512, 49],
                temp_w=1,
                pad_mask=None):
        # input shape B,F,T i.e. 10x1000x128
        other_dict = {}
        org_input = input
        input = input.unsqueeze(1)

        #patch-wise context modeling
        passt_out_dict = self.backbone(input)
        # pooling
        x = self.f_pool(passt_out_dict)
        x = torch.cat((x, x[:, -1, :].unsqueeze(1)), dim=1)  # padding from 99 frames to 100 frames
        x = self.interpolate_module(x, self.decode_ratio)
        assert x.shape[1] == 1000
        # another way of padding
        # B, T, C = x.shape
        # x = x.reshape(B, self.decode_ratio, T//self.decode_ratio, C)
        # x = torch.cat((x, x[:,:, -1, :].unsqueeze(2)), dim=2).reshape(B, T+self.decode_ratio, C)

        if encoder_win:
            from src.models.passt.passt_win import PasstWithSlide
            # merge the global feature map and the local feature map
            slide_window_model = PasstWithSlide(net=self, win_param=win_param)
            x_local = self.slide_window_layer(slide_window_model(org_input, emb_len=x.shape[1]))
            x = mix_rate * x_local + (1 - mix_rate) * x
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
        out = (sed_out * sed_out).sum(dim=1) / sed_out.sum(dim=1)
        at_out = torch.clamp(out, 1e-7, 1.)

        return sed_out.transpose(1, 2), at_out, other_dict

    def get_feature_extractor(self):
        return self.mel_trans

    def get_model_name(self):
        return "PaSST_SED"

    def get_backbone_upsample_ratio(self):
        return self.decode_ratio

    def get_backbone(self):
        return self.backbone
