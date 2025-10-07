import argparse
import logging
import os
import sys
import warnings

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

root = "/home/cpf/code/SSL_SED"
os.chdir(root)
sys.path.append(root)

from recipes.audioset_strong.setting import get_encoder
from src.codec.encoder import Encoder
from src.models.htsat.htsat import CLAPAudioCfp, create_htsat_model
from src.preprocess.dataset import StronglyLabeledDataset
from src.utils import load_yaml_with_relative_ref


class HTSAT_Wrapper(nn.Module):

    def __init__(self, model_weight_path, embed_dim) -> None:
        super().__init__()
        self.model = create_htsat_model(CLAPAudioCfp)
        self.model.load_state_dict(torch.load(model_weight_path))
        self.embed_dim = embed_dim

    def forward(self, wav):
        mel = self.model.wav2mel(wav, None)
        feat = self.model(mel)['fine_grained_embedding'].squeeze(1, 2)
        return feat


class EmbedDict():

    def __init__(self, event_num, embed_dim, device):
        self.event_num = event_num
        self.embed_dim = embed_dim
        self.embeddings = torch.zeros(event_num, embed_dim, device=device)  # sum of embeddings
        self.frame_num = [0 for _ in range(event_num)]  # num of frames per events
        self.exp_event_set = [301, 51, 314, 287, 192, 168, 78, 406, 85, 46, 208, 202]
        self.threshold = 0

    def update(self, embedding: torch.Tensor, label: torch.Tensor):
        """
        Update the embedding sum and frame count with new embeddings.
        
        Args:
            embedding (torch.Tensor): Tensor of shape (batch, frames, embed_dim)
            label (torch.Tensor): Tensor of shape (batch, frames, n_class), with one-hot encoding of the class labels
        """
        batch_size, num_frames, _ = embedding.shape
        embedding = embedding.reshape(batch_size * num_frames, self.embed_dim)
        label = label.reshape(batch_size * num_frames, self.event_num)

        for i in range(self.event_num):
            # Get the indices where the label is 1 (the current class)
            indices = label[:, i] == 1
            selected_embeddings = embedding[indices]
            if i not in self.exp_event_set or self.frame_num[i] <= self.threshold:
                # Update the sum of embeddings and the frame count
                self.embeddings[i, :] += selected_embeddings.sum(dim=0)
                self.frame_num[i] += selected_embeddings.shape[0]

    def values(self):
        """
        Compute the mean embedding for each class.
        
        Returns:
            torch.Tensor: Mean embeddings for each class of shape (event_num, embed_dim)
        """
        # Avoid division by zero by checking frame count
        means = torch.zeros_like(self.embeddings)
        for i in range(len(self.frame_num)):
            if self.frame_num[i] > 0:
                means[i] = self.embeddings[i] / self.frame_num[i]
        return means


class AudioEmbeddingExtractor():

    def __init__(self, net: HTSAT_Wrapper, dataloader, config, encoder: Encoder, device) -> None:
        self.net = net
        self.dataloader = dataloader
        self.config = config
        self.encoder = encoder
        self.device = device
        self.embed_dict = EmbedDict(
            event_num=len(encoder.labels),
            embed_dim=config['training']['embed_dim'],
            device=device,
        )

    def extract(self):
        tk0 = tqdm(self.dataloader, total=len(self.dataloader), leave=False, desc="Feature Extraction")
        self.net.eval()
        with torch.inference_mode():
            for wavs, label, _, _ in tk0:
                wavs = wavs.to(self.device)
                embedding = self.net(wavs)
                self.embed_dict.update(embedding, label.transpose(-1, -2))
        return self.embed_dict.values()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--config_path', type=str, help="path for configuration files")
    parser.add_argument('--save_path', type=str, help="path to save results")
    args = parser.parse_args()

    #set configurations
    configs = load_yaml_with_relative_ref(args.config_path)
    warnings.filterwarnings("ignore")
    device = torch.device("cuda:{}".format(args.gpu))  # note: don't support multi-gpu running.
    torch.cuda.set_device(device)

    #set encoder
    encoder = get_encoder(configs)
    #set network
    net = HTSAT_Wrapper(configs['training']['pretrain_model_path'], configs['training']['embed_dim']).to(device)

    #set Dataloaders
    data_df = pd.read_csv(configs["dataset"]["strong_tsv"], sep="\t")
    dataset = StronglyLabeledDataset(
        tsv_read=data_df,
        dataset_dir=configs["dataset"]["strong_folder"],
        return_name=False,
        encoder=encoder,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=configs["training"]["batch_size"],
        num_workers=configs["training"]["num_workers"],
    )
    extractor = AudioEmbeddingExtractor(
        net=net,
        dataloader=dataloader,
        config=configs,
        encoder=encoder,
        device=device,
    )

    feature_tensor = extractor.extract()
    torch.save(feature_tensor, args.save_path)
    print("=============== Ending  ================")
    logging.shutdown()
