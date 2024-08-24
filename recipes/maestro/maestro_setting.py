import os
import yaml

import pandas as pd
import torch

from src.preprocess.dataset import StronglyLabeledDataset
from recipes.basic_setting import get_configs, get_save_directories, get_encoder, get_logger
from recipes.finetune.cnn_trans.setting import get_models_passt_cnn, get_params


def get_labeldict_maestro():
    from archive.sound_event_class.classes_dict_mastro import classes_labels_maestro_real
    return classes_labels_maestro_real


def get_save_directories(configs, save_folder):
    general_cfg = configs["generals"]
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
    general_cfg["best_paths"] = os.path.join(save_folder, "best_model.pt")

    # psds folder
    if savepsds:
        psds_folders = os.path.join(save_folder, "psds")
    else:
        psds_folders = None

    general_cfg["psds_folders"] = psds_folders
    general_cfg["save_folder"] = save_folder

    return configs


def get_datasets(config, encoder, test_only, logger):
    devtest_df = pd.read_csv(config["dataset"]["test_tsv"], sep="\t")
    devtest_dataset = StronglyLabeledDataset(tsv_read=devtest_df,
                                             dataset_dir=config["dataset"]["test_folder"],
                                             return_name=True,
                                             encoder=encoder)
    test_dataset = devtest_dataset
    if not test_only:
        ##### data prep train & valid ##########
        train_strong_df = pd.read_csv(config["dataset"]["strong_tsv"], sep="\t")
        train_dataset = StronglyLabeledDataset(tsv_read=train_strong_df,
                                               dataset_dir=config["dataset"]["strong_folder"],
                                               return_name=False,
                                               encoder=encoder)

        valid_strong_df = pd.read_csv(config["dataset"]["val_tsv"], sep="\t")
        valid_dataset = StronglyLabeledDataset(tsv_read=valid_strong_df,
                                               dataset_dir=config["dataset"]["val_folder"],
                                               return_name=True,
                                               encoder=encoder)

        batch_sizes = config["training"]["batch_size"]
        sampler = torch.utils.data.RandomSampler(train_dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_sizes, drop_last=False)

        # logging
        logger.info("train: strong-{strong}".format(strong=len(train_dataset)))
        logger.info("val: strong-{strong}".format(strong=len(valid_dataset)))
    else:
        train_dataset = None
        valid_dataset = None
        batch_sampler = None

    logger.info("test: {0}".format(len(test_dataset)))
    return train_dataset, valid_dataset, test_dataset, batch_sampler
