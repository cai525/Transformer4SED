import argparse
import logging
import os
import pickle
import sys
import re
import warnings
from typing import Union, Optional

root = "ROOT-PATH"  # the root dir of the project
os.chdir(root)
sys.path.append(root)

import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from pycave.bayes.gmm import GaussianMixture
from pycave.clustering.kmeans import KMeans
from sklearn.decomposition import PCA

logging.getLogger('lightkit').setLevel(logging.CRITICAL)
logging.getLogger('pycave').setLevel(logging.CRITICAL)

from recipes.desed.pmam.setting import get_configs, get_encoder
from src.preprocess.dataset import UnlabeledDataset


class PseudoLabelGenerator():

    def __init__(self, net, dataloader, tokenizer: Union[GaussianMixture, KMeans], pca: Optional[PCA], device):
        self.net = net
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.pca = pca
        self.device = device

    def embedding2pseudo_label(self, feature_array: torch.Tensor):
        B, T, E, = feature_array.shape
        feature_array = feature_array.reshape(-1, E)
        if self.pca is not None:
            feature_array = pca.transformer(feature_array)
        if isinstance(self.tokenizer, GaussianMixture):
            prob = self.tokenizer.predict_proba(feature_array)
            C = prob.shape[-1]
            prob = prob.reshape((-1, T, C)).detach().cpu().numpy()
        elif isinstance(self.tokenizer, KMeans):
            prob = self.tokenizer.predict(feature_array).detach().cpu().numpy()
            assert prob.ndim <= 1
            C = self.tokenizer.model_.centroids.shape[0]
            prob = label_to_one_hot(prob, C)
            prob = prob.reshape((-1, T, C))
        else:
            raise RuntimeError("The tokenizer is expected to be GMMMixture or KMeans, but not {}".\
                format(type(self.tokenizer)))
        return prob

    def extract(self, feature_layer: str):
        """ Extract 1-d feature sequence from wavforme.
            Args:
                feature_layer: the layer name where the function extracts features from.
                downsample rate: the  rate to downsample the extracted features.
            Return:
                feature array with shape (L, C), where L is the length of array and C is the dimension of features.
        """
        pseudo_label_list = []
        filename_list = []

        def get_hook(feature_layer: str):
            if 'transformer' in feature_layer:

                def hook_fn(module, fea_in, fea_out):
                    feature = fea_out.detach().transpose(0, 1)
                    pseudo_label = self.embedding2pseudo_label(feature)
                    pseudo_label_list.append(pseudo_label)

                layer_id = int(re.search(r"transformer_(\d+)", feature_layer).group(1))
                hook = self.net.decoder.encoder_blocks[layer_id].register_forward_hook(hook_fn)

            elif feature_layer == "after_interpolate":

                def hook_fn(module, fea_in, fea_out):
                    feature = fea_out.detach()
                    pseudo_label = self.embedding2pseudo_label(feature)
                    pseudo_label_list.append(pseudo_label)

                hook = self.net.interpolate_module.register_forward_hook(hook_fn)
            else:
                raise RuntimeError("Unknown layer name {}".format(feature_layer))
            return hook

        hook = get_hook(feature_layer)

        tk0 = tqdm(self.dataloader, total=len(self.dataloader), leave=False, desc="Feature Extraction")
        with torch.inference_mode():
            self.net.eval()
            for wavs, label, pad_mask, idx, filenames, path in tk0:
                wavs = wavs.to(self.device)
                # Data preprocessing
                feature_extractor = self.net.get_feature_extractor()
                mel = feature_extractor(wavs)
                mel = feature_extractor.normalize(mel)
                # features are extracted by the hook
                self.net(mel, encoder_win=False)
                filename_list.extend(list(filenames))

        if hook is not None:
            hook.remove()

        pseudo_label_array = np.concatenate(pseudo_label_list, axis=0)
        assert len(filename_list) == pseudo_label_array.shape[0], "{} file names but {} pseudo labels".\
            format(len(filename_list), pseudo_label_array.shape[0])

        return pseudo_label_array, filename_list


def load_dataset(configs, encoder, dataset_type, batch_size):
    dataset = UnlabeledDataset(
        configs["dataset"]["{}_folder".format(dataset_type)],
        True,
        encoder,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=configs["generals"]["num_workers"],
    )
    return dataloader


def save_prob(file_name, prob: np.ndarray, sr):
    class_num = prob.shape[-1]
    interval = 1 / sr
    time_tensor = np.zeros((prob.shape[0], 2))
    time_tensor[:, 0] = np.arange(0, interval * prob.shape[0], interval).T
    time_tensor[:, 1] = time_tensor[:, 0] + interval
    df_data = np.concatenate((time_tensor, prob), axis=1)
    df_data = np.round(df_data, decimals=4)
    df = pd.DataFrame(df_data, columns=["onset", "offset", *range(class_num)])
    df.to_csv(file_name, sep='\t', index=False)


def label_to_one_hot(array, C):
    N = array.size
    one_hot = np.zeros((N, C))
    one_hot[np.arange(N), array] = 1
    return one_hot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--config_dir', type=str, help="directory for configuration files")
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--pca_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--feature_layer', type=str, help="name of layers where features are extracted from")
    parser.add_argument('--algorithm', type=str, default="GMM")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    # load model
    with open(args.tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
        assert isinstance(tokenizer, GaussianMixture) or isinstance(tokenizer, KMeans)
        tokenizer.trainer_params['enable_progress_bar'] = False

    if args.pca_path is not None:
        with open(args.pca_path, "rb") as f:
            pca = pickle.load(f)
            assert isinstance(pca, PCA)
    else:
        pca = None

    #set configurations
    configs = get_configs(config_path=args.config_dir)
    encoder = get_encoder({0: 0}, configs)  # note: label dict is useless here

    #set network
    if "PaSST_SED" in configs.keys():
        from src.models.passt.passt_sed import PaSST_SED
        net = PaSST_SED(**configs["PaSST_SED"])
    elif "PaSST_CNN" in configs.keys():
        from src.models.cnn_transformer.passt_cnn import PaSST_CNN
        net = PaSST_CNN(**configs["PaSST_CNN"])
    else:
        raise RuntimeError("Unknown model structure.")

    device = torch.device("cuda:0")
    net = net.to(device=device)
    # tokenizer = tokenizer.to(device=device)
    torch.cuda.set_device(device)

    # load existed model
    if "pretrain_model_path" in configs["generals"]:
        pretrain_model_path = configs["generals"]["pretrain_model_path"]
        logging.info("loading pretrained model from {path}".format(path=pretrain_model_path))
        params_dict = torch.load(pretrain_model_path)
        net.load_state_dict(params_dict, strict=False)

    for data_type in ['strong', 'weak', 'unlabeled', 'val']:
        dataloader = load_dataset(configs, encoder, data_type, batch_size=36)
        pseudo_label_generator = PseudoLabelGenerator(
            net=net,
            dataloader=dataloader,
            tokenizer=tokenizer,
            pca=pca,
            device=device,
        )
        prob, filename_list = pseudo_label_generator.extract(feature_layer=args.feature_layer)
        os.mkdir(os.path.join(args.save_dir, data_type))
        for i, filename in enumerate(filename_list):
            prob_i = prob[i]
            tsv_name = os.path.join(args.save_dir, data_type, "{0}.tsv".format(filename.replace(".wav", "")))
            save_prob(tsv_name, prob_i, sr=100)
