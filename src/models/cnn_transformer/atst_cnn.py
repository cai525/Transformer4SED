import torch.nn as nn
import torch

from src.models.sed_model import SEDModel
from src.models.atst.atst_model import ATST
from src.models.atst.atst_feature_extraction import AtstFeatureExtractor, MelSpectrogram
from src.models.cnn.base import CNN
from src.models.encoder_slide_window import EncoderSlideWindow


class AtstWithSlide(EncoderSlideWindow):

    def __init__(self, net: SEDModel, win_param=[512, 29]):
        super().__init__(net, win_param)

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        atst_out_dict = self.net.get_backbone_encoder()(input)
        x = atst_out_dict["feature_map_{layer}".format(layer=12)]
        return x


class BidirectionalGRU(nn.Module):

    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        """
            Initialization of BidirectionalGRU instance
        Args:
            n_in: int, number of input
            n_hidden: int, number of hidden layers
            dropout: flat, dropout
            num_layers: int, number of layers
        """

        super(BidirectionalGRU, self).__init__()
        self.rnn = nn.GRU(
            n_in,
            n_hidden,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
            num_layers=num_layers,
        )

    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        return recurrent


class CRNN(SEDModel):

    def __init__(
        self,
        unfreeze_atst_layer=0,
        n_in_channel=1,
        nclass=10,
        activation="glu",
        dropout=0.5,
        rnn_type="BGRU",
        n_RNN_cell=128,
        n_layers_RNN=2,
        dropout_recurrent=0,
        embedding_size=768,
        atst_init=None,
        atst_dropout=0.0,
        **kwargs,
    ):
        super(CRNN, self).__init__()

        self.atst_mel_trans = AtstFeatureExtractor()
        self.cnn_mel_trans = MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            win_length=2048,
            hop_length=256,
            f_min=0,
            f_max=8000,
            n_mels=128,
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        )
        self.n_in_channel = n_in_channel
        self.atst_dropout = atst_dropout
        n_in_cnn = n_in_channel
        self.cnn = CNN(n_in_channel=n_in_cnn, activation=activation, conv_dropout=dropout, **kwargs)

        if rnn_type == "BGRU":
            nb_in = self.cnn.nb_filters[-1]
            nb_in = nb_in * n_in_channel
            self.rnn = BidirectionalGRU(
                n_in=nb_in,
                n_hidden=n_RNN_cell,
                dropout=dropout_recurrent,
                num_layers=n_layers_RNN,
            )
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell * 2, nclass)
        self.sigmoid = nn.Sigmoid()

        self.dense_softmax = nn.Linear(n_RNN_cell * 2, nclass)
        self.softmax = nn.Softmax(dim=-1)

        self.embedding_size = embedding_size
        self.cat_tf = torch.nn.Linear(nb_in + embedding_size, nb_in)

        self.init_atst(atst_init)

        self.unfreeze_atst_layer = unfreeze_atst_layer

    def init_atst(self, path=None):
        if path is None:
            self.atst_frame = ATST(None, atst_dropout=self.atst_dropout)
        else:
            self.atst_frame = ATST(path, atst_dropout=self.atst_dropout)

            print("Loading ATST from:", path)

    def init_model(self, path, mode=None):
        if path is None:
            pass
        else:
            if mode == "teacher":
                print("Loading teacher from:", path)
                state_dict = torch.load(path, map_location="cpu")["sed_teacher"]
            else:
                print("Loading student from:", path)
                state_dict = torch.load(path, map_location="cpu")["sed_student"]
            self.load_state_dict(state_dict, strict=False)
            print("Model loaded")

    def forward(self, x, pretrain_x, encoder_win, mix_rate, win_param, temp_w=1):
        x = x.transpose(1, 2).unsqueeze(1)
        # conv features
        x = self.cnn(x)
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)  # [bs, frames, chan]
        # rnn features
        atst_out_dict = self.atst_frame(pretrain_x)
        embeddings = atst_out_dict["feature_map_{layer}".format(layer=12)]
        if encoder_win:
            slide_window_model = AtstWithSlide(net=self, win_param=win_param)
            embeddings_local = slide_window_model(pretrain_x, emb_len=250)
            embeddings = mix_rate * embeddings_local + (1 - mix_rate) * embeddings

        embeddings = torch.nn.functional.adaptive_avg_pool1d(embeddings.transpose(1, 2), 156).transpose(1, 2)
        x = self.cat_tf(torch.cat((x, embeddings), -1))

        x = self.rnn(x)
        x = self.dropout(x)
        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong / temp_w)
        sof = self.dense_softmax(x)  # [bs, frames, nclass]
        sof = self.softmax(sof)
        sof = torch.clamp(sof, min=1e-7, max=1)
        weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
        return strong.transpose(1, 2), weak

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(CRNN, self).train(mode)

    def get_feature_extractor(self):
        return {"atst": self.atst_mel_trans, "cnn": self.cnn_mel_trans}

    def get_backbone_encoder(self):
        return self.atst_frame

    def get_model_name(self):
        return "ATST_CNN"

    def get_backbone_upsample_ratio(self):
        return 1
