import json
import logging
import os
import yaml
from collections import OrderedDict
from src.models.passt.passt_sed import PaSST_SED

import torch
from torch.utils.data import DataLoader


from src.codec.encoder import Encoder
from src.dataset import UnlabeledDataset, ConcatDatasetBatchSampler


def get_encoder_passt(LabelDict, config):
    return Encoder(list(LabelDict.keys()),
                   audio_len=config['feature']['audio_max_len'],
                   frame_len=config['feature']["win_length"],
                   frame_hop=config['feature']["hopsize"],
                   net_pooling=config['feature']["net_subsample"],
                   sr=config['feature']["sr"])


def get_models_mlm(configs):
    net = PaSST_SED(**configs["PaSST_SED"])
    for k, p in net.patch_transformer.named_parameters():
        p.requires_grad = False
    return net

def get_datasets_mlm(configs, train_cfg):
    encoder = train_cfg["encoder"]
    dataset_cfg = configs["dataset"]
    batch_size_val = configs["generals"]["batch_size_val"]
    num_workers = configs["generals"]["num_workers"]
    batch_sizes = configs["generals"]["batch_size"]

    strong_train_dataset =  UnlabeledDataset(dataset_cfg["strong_folder"], False, encoder)
    weak_train_dataset = UnlabeledDataset(dataset_cfg["weak_folder"], False, encoder)
    unlabeled_dataset = UnlabeledDataset(dataset_cfg["unlabeled_folder"], False, encoder)
    test_dataset = UnlabeledDataset(dataset_cfg["test_folder"], False, encoder)
    valid_dataset = UnlabeledDataset(dataset_cfg["test_folder"], False, encoder)
    use_external_dataset = False
    if len(batch_sizes) == 4:
        audioset_balance_dataset = UnlabeledDataset(dataset_cfg["audioset_balance"], False, encoder)
        use_external_dataset = True
    # build dataloaders
    # get train dataset
    if not use_external_dataset:
        train_data = [strong_train_dataset, weak_train_dataset, unlabeled_dataset]
    else:
        train_data = [strong_train_dataset, weak_train_dataset, unlabeled_dataset, audioset_balance_dataset]
        
    train_dataset = torch.utils.data.ConcatDataset(train_data)
    train_samplers = [torch.utils.data.RandomSampler(x) for x in train_data]
    
    print("batch_sizes = " , batch_sizes)
    train_batch_sampler = ConcatDatasetBatchSampler(train_samplers, batch_sizes)
    
    train_cfg["trainloader"] = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=num_workers)
    train_cfg["validloader"] = DataLoader(valid_dataset, batch_size=batch_size_val, num_workers=num_workers)
    train_cfg["testloader"] = DataLoader(test_dataset, batch_size=batch_size_val, num_workers=num_workers)
    
    return train_cfg




def get_configs(config_dir):
    #get hyperparameters from yaml
    with open(config_dir, "r") as f:
        configs = yaml.safe_load(f)

    train_cfg = configs["training"]
    return configs, train_cfg


def get_save_directories(configs, train_cfg, save_folder):
    general_cfg = configs["generals"]
    savepsds = general_cfg["savepsds"]

    # set save folder
    configs["generals"]["save_folder"] = save_folder 
    
    if os.path.isdir(save_folder):
        print("Saveing path has already existed")
    else:
        os.makedirs(save_folder, exist_ok=True)  # saving folder
    with open(os.path.join(save_folder, 'config.yaml'), 'w') as f:
        yaml.dump(configs, f)  # save yaml in the saving folder
            
    print("save directory : " + save_folder)
    #set best paths
    stud_best_path = os.path.join(save_folder, "best_student.pt")
    tch_best_path = os.path.join(save_folder, "best_teacher.pt")
    train_cfg["best_paths"] = [stud_best_path, tch_best_path]

    # psds folder
    if savepsds:
        stud_psds_folder = os.path.join(save_folder, "psds_student")
        tch_psds_folder = os.path.join(save_folder, "psds_teacher")
        psds_folders = [stud_psds_folder, tch_psds_folder]
    else:
        psds_folders = [None, None]
        
    train_cfg["psds_folders"] = psds_folders
    train_cfg["save_folder"] = save_folder 

    return configs, train_cfg


def get_logger(save_folder):
    logger = logging.getLogger()
    while len(logger.handlers) > 0:  # check if handler already exists
        logger.removeHandler(logger.handlers[0])  # Logger already exists
    formatter = logging.Formatter('%(message)s')
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(save_folder, "log.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_labeldict():
    with open("./meta/desed/labeldict_DESED.json") as f:
        ret = json.load(f, object_pairs_hook=OrderedDict)
    return ret






