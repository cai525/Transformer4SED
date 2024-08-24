import json
import logging
import os
import yaml
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader


from src.codec.encoder import Encoder
from src.preprocess.dataset import UnlabeledDataset, ConcatDatasetBatchSampler


def get_encoder(LabelDict, config):
    return Encoder(list(LabelDict.keys()),
                   audio_len=config['feature']['audio_max_len'],
                   frame_len=config['feature']["win_length"],
                   frame_hop=config['feature']["hopsize"],
                   net_pooling=config['feature']["net_subsample"],
                   sr=config['feature']["sr"])


def get_datasets_mlm(configs, encoder, return_name=False):
    dataset_cfg = configs["dataset"]
    batch_size_val = configs["generals"]["batch_size_val"]
    num_workers = configs["generals"]["num_workers"]
    batch_sizes = configs["generals"]["batch_size"]

    strong_train_dataset =  UnlabeledDataset(dataset_cfg["strong_folder"], return_name, encoder) if batch_sizes[0] > 0 else None
    weak_train_dataset = UnlabeledDataset(dataset_cfg["weak_folder"], return_name, encoder) if batch_sizes[1] > 0 else None
    unlabeled_dataset = UnlabeledDataset(dataset_cfg["unlabeled_folder"], return_name, encoder) if batch_sizes[2] > 0 else None
    valid_dataset = UnlabeledDataset(dataset_cfg["test_folder"], False, encoder)
    use_external_dataset = False
    if len(batch_sizes) == 4:
        audioset_balance_dataset = UnlabeledDataset(dataset_cfg["audioset_balance"], False, encoder)
        use_external_dataset = True
    # build dataloaders
    # get train dataset
    if not use_external_dataset:
        train_data = [data for data in (strong_train_dataset, weak_train_dataset, unlabeled_dataset) if data]
        
    else:
        train_data = [strong_train_dataset, weak_train_dataset, unlabeled_dataset, audioset_balance_dataset]
        
    train_dataset = torch.utils.data.ConcatDataset(train_data)
    train_samplers = [torch.utils.data.RandomSampler(x) for x in train_data]
    
    print("batch_sizes = " , batch_sizes)
    batch_sizes = [b for b in batch_sizes if b > 0]
    train_batch_sampler = ConcatDatasetBatchSampler(train_samplers, batch_sizes)
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size_val, num_workers=num_workers)
    
    return train_loader, valid_loader




def get_configs(config_dir):
    #get hyperparameters from yaml
    with open(config_dir, "r") as f:
        configs = yaml.safe_load(f)
    return configs


def get_save_directories(configs, save_folder):
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
    configs["generals"]["best_paths"] = os.path.join(save_folder, "best_student.pt")
    return configs


def get_logger(save_folder, log_level=logging.INFO):
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
    logger.setLevel(log_level)
    return logger


def get_labeldict():
    with open("./archive/sound_event_class/labeldict_DESED.json") as f:
        ret = json.load(f, object_pairs_hook=OrderedDict)
    return ret