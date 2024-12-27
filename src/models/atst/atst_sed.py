import torch
import torch.nn as nn

from src.models.sed_model import SEDModel
from src.models.atst.atst_model import ATST
from src.models.atst.atst_feature_extraction import AtstFeatureExtractor
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
        seq = seq.transpose(1, 2)  # B,T,C->B,C,T
        seq = torch.nn.functional.interpolate(seq, scale_factor=ratio, mode=self.mode)
        seq = seq.transpose(1, 2)  # B,C,T->B,T,C
        return seq


class AtstSED(SEDModel):

    def __init__(self,
                 decode_interpolate_ratio=10,
                 interpolate_mode='linear',
                 feature_layer=10,
                 embed_dim=768,
                 atst_path=None,
                 atst_dropout=0,
                 f_pool='mean_pool',
                 decoder='gru',
                 decoder_layer_num=2,
                 class_num=10,
                 at_adapter=False,
                 at_feature_layer=12,
                 decoder_win_len=None,
                 mlm=False,
                 mlm_dict=dict()):

        super().__init__()
        #setting mel feature extrator and specaug
        self.mel_trans = AtstFeatureExtractor()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.f_pool_name = f_pool
        self.feature_layer = feature_layer
        self.decoder_name = decoder
        self.decode_interpolate_ratio = decode_interpolate_ratio
        self.class_num = class_num
        self.embed_dim = embed_dim

        self.encoder_out_norm = nn.LayerNorm(embed_dim)

        self.interpolate_module = InterpolateModule(mode=interpolate_mode)
        self.slide_window_layer = nn.Identity()  # Just for capture the output of the sliding windows outside the class
        self.mlm = mlm

        # Initial
        self.init_atst(atst_path, atst_dropout)
        if mlm:
            self.init_mlm(device=device, out_dim=self.embed_dim, mlm_dict=mlm_dict)

        self.init_decoder(embed_dim, class_num, decoder_win_len, decoder_layer_num)

        self.at_adpater = at_adapter
        if self.at_adpater:
            self.at_feature_layer = at_feature_layer
            self.at_adpater = nn.Sequential(nn.LayerNorm(768), GAP(), nn.Linear(768, class_num))

    def init_atst(self, path=None, atst_dropout=0):
        if path is None:
            self.atst_frame = ATST(None, atst_dropout=atst_dropout)
        else:
            self.atst_frame = ATST(path, atst_dropout=atst_dropout)

    def init_decoder(self, out_dim, class_num, win_len, decoder_layer_num):
        self.decoder_layer_num = decoder_layer_num
        if self.decoder_name == 'gru':
            self.GRU = nn.GRU(out_dim,
                              out_dim,
                              bidirectional=True,
                              batch_first=True,
                              num_layers=decoder_layer_num,
                              dropout=0)
            self.classifier = nn.Linear(2 * out_dim, class_num)

        elif self.decoder_name == "transformer":
            self.decoder = TransformerDecoder(input_dim=out_dim, decoder_layer_num=decoder_layer_num)
            self.classifier = nn.Linear(out_dim, class_num)
        elif self.decoder_name == "transformerXL":
            self.decoder = TransformerXLDecoder(input_dim=out_dim,
                                                seq_len=1000,
                                                window_len=win_len,
                                                decoder_layer_num=decoder_layer_num)
            self.classifier = nn.Linear(out_dim, class_num)
        elif self.decoder_name == "conformer":
            self.decoder = ConformerDecoder(input_dim=out_dim,
                                            seq_len=1000,
                                            window_len=win_len,
                                            decoder_layer_num=decoder_layer_num)
            self.classifier = nn.Linear(out_dim, class_num)
        elif self.decoder_name == 'no':
            self.classifier = nn.Linear(out_dim, class_num)
        else:
            raise ValueError('invalid decoder block')

    def init_mlm(self, device, out_dim, mlm_dict=dict()):
        self.mlm_tool = MlmModule(device=device, **mlm_dict)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, out_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.mlm_mlp = nn.Sequential(torch.nn.Linear(out_dim, out_dim), torch.nn.GELU(), torch.nn.Dropout(p=0.1),
                                     torch.nn.Linear(out_dim, out_dim))

    def encoder_step(self, input, encoder_win, mix_rate, win_param):
        org_input = input  # input shape: [B, F=64, 1001]
        atst_out_dict = self.atst_frame(input,
                                        n=self.get_backbone_encoder_depth() + 1 -
                                        min(self.feature_layer, self.at_feature_layer))
        x = atst_out_dict["feature_map_{layer}".format(layer=self.feature_layer)]
        # x.shape = [batch, time, channel]
        assert x.shape[-1] == self.embed_dim
        x = self.encoder_out_norm(x)
        # interpolate
        x = self.interpolate_module(x, self.decode_interpolate_ratio)

        if encoder_win:
            from src.models.atst.atst_win import AtstWithSlide
            # merge the global feature map and the local feature map
            slide_window_model = AtstWithSlide(net=self, win_param=win_param)
            x_local = self.slide_window_layer(slide_window_model(org_input, emb_len=x.shape[1]))
            x = mix_rate * x_local + (1 - mix_rate) * x
        return x, atst_out_dict

    def decoder_step(self, x, other_dict):
        # mask language model
        if self.mlm:
            other_dict["frame_before_mask"] = x
            x, mask_id_seq = self.mlm_tool.setence_mask(x, self.mask_token)
            other_dict["mask_id_seq"] = mask_id_seq

        # time series model
        if self.decoder_name == 'gru':
            self.GRU.flatten_parameters()
            x, _ = self.GRU(x)
        elif self.decoder_name.find("transformer") >= 0:
            x = self.decoder(x)
        elif self.decoder_name == 'no':
            x = x
        return x

    def forward(self, input: torch.Tensor, encoder_win=False, mix_rate=0.5, win_param=[512, 49], temp_w=1):
        other_dict = {}
        x, atst_out_dict = self.encoder_step(input, encoder_win, mix_rate, win_param)
        # decoder
        x = self.decoder_step(x, other_dict)

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

        # linear-softmax pool
        out = (sed_out * sed_out).sum(dim=1) / sed_out.sum(dim=1)
        at_out = torch.clamp(out, 1e-7, 1.)
        other_dict['sed_logit'] = other_dict["logit"].transpose(1, 2)
        other_dict['at_logit'] = torch.logit(at_out)

        #################################################################################
        #                            Audio tagging branch
        #################################################################################
        if self.at_adpater:
            at_embedding = atst_out_dict["feature_map_{layer}".format(layer=self.at_feature_layer)]
            at_adapter_logit = self.at_adpater(at_embedding)
            #print(at_adapter_embedding.shape)
            at_adapter_out = torch.sigmoid(at_adapter_logit)
            other_dict['at_out'] = at_adapter_out

        return sed_out.transpose(1, 2), at_out, other_dict

    def get_feature_extractor(self):
        return self.mel_trans

    def get_backbone_encoder(self):
        return self.atst_frame

    def get_model_name(self):
        return "ATST_SED"

    def get_encoder_depth(self):
        if not hasattr(self, "_encoder_depth"):
            self._encoder_depth = len(self.atst_frame.atst.blocks)
        return self._encoder_depth

    def get_backbone_upsample_ratio(self):
        return self.decode_interpolate_ratio
