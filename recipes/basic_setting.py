import json
import logging
import os
import shutil
import yaml
from collections import OrderedDict

import pandas as pd
import torch

from src.codec.encoder import Encoder
from src.preprocess.dataset import (StronglyLabeledDataset, WeaklyLabeledDataset, UnlabeledDataset,
                                    ConcatDatasetBatchSampler)
from src.utils.log import Logger


def get_configs(config_dir):
    with open(config_dir, "r") as f:
        configs = yaml.safe_load(f)
    return configs


def get_save_directories(configs, save_folder):
    general_cfg = configs["generals"]
    train_cfg = configs["training"]
    savepsds = general_cfg["savepsds"]

    # set save folder
    configs["generals"]["save_folder"] = save_folder
    if os.path.isdir(save_folder):
        print("folder {0} has already existed".format(save_folder))
    else:
        os.makedirs(save_folder, exist_ok=True)  # saving folder

    with open(os.path.join(save_folder, 'config.yaml'), 'w') as f:
        yaml.dump(configs, f)  # save yaml in the saving folder

    print("save directory : " + save_folder)

    # set paths to save best model
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

    return configs


def get_encoder(LabelDict, config):
    return Encoder(list(LabelDict.keys()),
                   audio_len=config["feature"]["audio_max_len"],
                   frame_len=config["feature"]["win_length"],
                   frame_hop=config["feature"]["hopsize"],
                   net_pooling=config["feature"]["net_subsample"],
                   sr=config["feature"]["sr"])


def get_logger(save_folder, need_tensorboard=True, record_carbon=False, log_level=logging.INFO) -> Logger:
    logger = Logger()

    formatter = logging.Formatter('%(message)s')
    # standard io handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    # file handler
    file_handler = logging.FileHandler(os.path.join(save_folder, "log.txt"))
    file_handler.setFormatter(formatter)
    logger.init_logger([stream_handler, file_handler], level=log_level)
    if need_tensorboard:
        logger.init_tensorboard_writer(save_folder)
    if record_carbon:
        logger.init_carbon_tracker(save_folder)

    return logger


def get_labeldict():
    with open("./archive/sound_event_class/labeldict_DESED.json") as f:
        ret = json.load(f, object_pairs_hook=OrderedDict)
    return ret


def get_labeldict_audioset_strongOOD() -> OrderedDict:
    with open("./archive/sound_event_class/labeldict_audioset_strongOOD.json") as f:
        ret = json.load(f, object_pairs_hook=OrderedDict)
    return ret


def get_datasets(config, encoder, evaluation, test_only, logger):
    assert config["training"]["batch_size_val"] % int(torch.cuda.device_count()) == 0, \
        "The validation batch size(={0}) must be set to an integer multiple of the number of GPUs(={1})"\
            .format(config["training"]["batch_size_val"], int(torch.cuda.device_count()))
    if config["generals"].get("predict", False):
        devtest_dataset = UnlabeledDataset(dataset_dir=config["dataset"]["pubeval_folder"],
                                           return_name=True,
                                           encoder=encoder)
        logger.info(">>>> Detecte the sound events in the  wavs of the evaluation dataset! >>>> ")
        logger.info("Obj datasets contains {0} wavs".format(len(devtest_dataset)))
    elif not evaluation:
        devtest_df = pd.read_csv(config["dataset"]["test_tsv"], sep="\t")
        devtest_dataset = StronglyLabeledDataset(tsv_read=devtest_df,
                                                 dataset_dir=config["dataset"]["test_folder"],
                                                 return_name=True,
                                                 encoder=encoder)
    else:
        devtest_df = pd.read_csv(config["dataset"]["pubeval_tsv"], sep="\t")
        devtest_dataset = StronglyLabeledDataset(tsv_read=devtest_df,
                                                 dataset_dir=config["dataset"]["pubeval_folder"],
                                                 return_name=True,
                                                 encoder=encoder)

    test_dataset = devtest_dataset
    if not test_only:
        assert sum(config["training"]["batch_size"]) % int(torch.cuda.device_count()) == 0, \
        "The training batch size(={0}) must be set to an integer multiple of the number of GPUs(={1})"\
            .format(sum(config["training"]["batch_size"]), int(torch.cuda.device_count()))
            
        ##### data prep train & valid ##########
        synth_df = pd.read_csv(config["synth_dataset"]["synth_train_tsv"], sep="\t")
        synth_set = StronglyLabeledDataset(tsv_read=synth_df,
                                           dataset_dir=config["synth_dataset"]["synth_train_folder"],
                                           return_name=False,
                                           encoder=encoder)

        train_strong_df = pd.read_csv(config["dataset"]["strong_tsv"], sep="\t")
        strong_set = StronglyLabeledDataset(tsv_read=train_strong_df,
                                            dataset_dir=config["dataset"]["strong_folder"],
                                            return_name=False,
                                            encoder=encoder)

        weak_df = pd.read_csv(config["dataset"]["weak_tsv"], sep="\t")
        train_weak_df = weak_df.sample(
            frac=1.0,
            random_state=config["training"]["seed"],
        )
        valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
        train_weak_df = train_weak_df.reset_index(drop=True)
        weak_set = WeaklyLabeledDataset(tsv_read=train_weak_df,
                                        dataset_dir=config["dataset"]["weak_folder"],
                                        return_name=False,
                                        encoder=encoder)

        unlabeled_set = UnlabeledDataset(dataset_dir=config["dataset"]["unlabeled_folder"],
                                         return_name=False,
                                         encoder=encoder)

        valid_strong_df = pd.read_csv(config["dataset"]["val_tsv"], sep="\t")
        strong_val = StronglyLabeledDataset(tsv_read=valid_strong_df,
                                            dataset_dir=config["dataset"]["val_folder"],
                                            return_name=True,
                                            encoder=encoder)

        weak_val = WeaklyLabeledDataset(tsv_read=valid_weak_df,
                                        dataset_dir=config["dataset"]["weak_folder"],
                                        return_name=True,
                                        encoder=encoder)
        tot_train_data = [strong_set, synth_set, weak_set, unlabeled_set]
        train_dataset = torch.utils.data.ConcatDataset(tot_train_data)

        batch_sizes = config["training"]["batch_size"]
        samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
        batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)
        valid_dataset = torch.utils.data.ConcatDataset([strong_val, weak_val])

        # logging
        logger.info("train: strong-{strong}, syn-{syn} ,weak-{weak}, unlabeled-{unlabeled}".format(
            strong=len(strong_set), syn=len(synth_set), weak=len(weak_set), unlabeled=len(unlabeled_set)))
        logger.info("val: strong-{strong} ,weak-{weak}".format(strong=len(strong_val), weak=len(weak_val)))
    else:
        train_dataset = None
        valid_dataset = None
        batch_sampler = None
        tot_train_data = None

    logger.info("test: {0}".format(len(test_dataset)))
    return train_dataset, valid_dataset, test_dataset, batch_sampler, tot_train_data
