import argparse
import logging
import os
import pickle
import re
import random
import sys
import warnings
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

root = "ROOT-PATH"  # the root dir of the project
os.chdir(root)
sys.path.append(root)

from recipes.desed.pmam.setting import *
from src.utils.statistics.model_statistic import count_parameters
from src.preprocess.dataset import UnlabeledDataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_dataset(configs, encoder):
    dataset_type_tuple = ("strong", "weak", "unlabeled")  # all the real data in the train dataset
    dataset_list = []
    for dataset_type in dataset_type_tuple:
        dataset = UnlabeledDataset(
            configs["dataset"]["{}_folder".format(dataset_type)],
            True,
            encoder,
        )

        dataset_list.append(dataset)

    dataset = torch.utils.data.ConcatDataset(dataset_list)
    samplers = [torch.utils.data.RandomSampler(x) for x in dataset_list]
    batch_sampler = ConcatDatasetBatchSampler(samplers, configs["generals"]["batch_size"])
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=configs["generals"]["num_workers"],
    )
    return dataloader


class FeatureExtractor():

    def __init__(self, net, dataloader, config, encoder, device):
        self.net = net
        self.dataloader = dataloader
        self.config = config
        self.encoder = encoder
        self.device = device

    def sample_feature(self, feature_array, downsample_rate):
        intervals = torch.arange(0, len(feature_array), downsample_rate)
        random_offsets = torch.randint(0, downsample_rate, size=(intervals.shape[0], ))
        index = intervals + random_offsets
        ret = feature_array[index]
        return ret

    def extract(self, feature_layer: str, downsample_rate: int):
        """ Extract 1-d feature sequence from wavforme.
            Args:
                feature_layer: the layer name where the function extracts features from.
                downsample rate: the  rate to downsample the extracted features.
            Return:
                feature array with shape (L, C), where L is the length of array and C is the dimension of features.
        """
        feature_list = []

        def get_hook(feature_layer: str):
            if 'transformer' in feature_layer:

                def hook_fn(module, fea_in, fea_out):
                    feature = fea_out.detach().transpose(0, 1).reshape(-1, fea_out.shape[-1]).cpu()
                    feature = self.sample_feature(feature, downsample_rate)
                    feature_list.append(feature)

                layer_id = int(re.search(r"transformer_(\d+)", feature_layer).group(1))
                hook = self.net.module.decoder.encoder_blocks[layer_id].register_forward_hook(hook_fn)

            elif feature_layer == "after_interpolate":

                def hook_fn(module, fea_in, fea_out):
                    feature = fea_out.detach().reshape(-1, fea_out.shape[-1]).cpu()
                    feature = self.sample_feature(feature, downsample_rate)
                    feature_list.append(feature)

                hook = self.net.module.interpolate_module.register_forward_hook(hook_fn)
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
                feature_extractor = self.net.module.get_feature_extractor()
                mel = feature_extractor(wavs)
                mel = feature_extractor.normalize(mel)
                # features are extracted by the hook
                self.net(mel, encoder_win=False)

        if hook is not None:
            hook.remove()

        feature_array = torch.cat(feature_list, axis=0)
        logging.info("Feature's shape is {}".format(feature_array.shape))
        logging.info("Feature's size is {} MB".format(sys.getsizeof(feature_array) * 1e-6))

        return feature_array


if __name__ == "__main__":
    print("=" * 50 + "start!!!!" + "=" * 50)
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--gpu', default=0, type=int,\
        help='selection of gpu when you run separate trainings on single server',
    )
    parser.add_argument('--config_dir', type=str, help="directory for configuration files")
    parser.add_argument('--save_folder', type=str, help="directory to save results")
    parser.add_argument('--feature_layer', type=str, help="name of layers where features are extracted from")
    parser.add_argument('--downsample_rate', type=int, help="rate to downsample features")
    args = parser.parse_args()

    #set configurations
    configs = get_configs(config_path=args.config_dir)

    #set save directories
    configs = get_save_directories(configs, args.save_folder)

    warnings.filterwarnings("ignore")
    #torch information
    logging.info("date & time of start is : " + str(datetime.now()).split('.')[0])
    logging.info("torch version is: " + str(torch.__version__))
    assert torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu))  # note: don't support multi-gpu running.
    print(device)
    torch.cuda.set_device(device)
    logging.info("device: " + str(device))

    #class label dictionary
    LabelDict = get_labeldict()

    #set encoder
    encoder = get_encoder(LabelDict, configs)

    #set network
    if "PaSST_SED" in configs.keys():
        from src.models.passt.passt_sed import PaSST_SED
        net = PaSST_SED(**configs["PaSST_SED"])
    elif "PaSST_CNN" in configs.keys():
        from src.models.cnn_transformer.passt_cnn import PaSST_CNN
        net = PaSST_CNN(**configs["PaSST_CNN"])
    else:
        raise RuntimeError("Unknown model structure.")

    logging.info("Total Trainable Params: %.3f M" %
                 (count_parameters(net) * 1e-6))  #print number of learnable parameters in the model

    # move to gpus
    net = nn.DataParallel(net)
    net = net.to(device)

    # load existed model
    if "pretrain_model_path" in configs["generals"]:
        pretrain_model_path = configs["generals"]["pretrain_model_path"]
        print("loading pretrained model from {path}".format(path=pretrain_model_path))
        params_dict = torch.load(pretrain_model_path)
        net.load_state_dict(params_dict, strict=False)

    #set Dataloaders
    dataloader = load_dataset(configs, encoder)
    extractor = FeatureExtractor(
        net=net,
        dataloader=dataloader,
        config=configs,
        encoder=encoder,
        device=device,
    )

    feature_array = extractor.extract(feature_layer=args.feature_layer, downsample_rate=args.downsample_rate)
    # save array
    with open(os.path.join(configs["generals"]["save_folder"], "feature.pkl"), 'wb') as f:
        pickle.dump(feature_array, f)
    print("=============== Ending  ================")
    logging.shutdown()
