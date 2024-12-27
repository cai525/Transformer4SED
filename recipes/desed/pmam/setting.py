import os
import yaml
from collections import OrderedDict

import torch

from src.preprocess.dataset import FrameWiseLabeledDataset, ConcatDatasetBatchSampler


def get_save_directories(configs, save_folder):
    general_cfg = configs["generals"]

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

    general_cfg["save_folder"] = save_folder

    return configs


def get_pseudo_label_dict(class_num):
    return OrderedDict([(i, i) for i in range(class_num)])


def get_pseudo_dataset(config, encoder, logger):
    assert config["training"]["batch_size_val"] % int(torch.cuda.device_count()) == 0, \
    "The validation batch size(={0}) must be set to an integer multiple of the number of GPUs(={1})"\
        .format(config["training"]["batch_size_val"], int(torch.cuda.device_count()))

    assert sum(config["training"]["batch_size"]) % int(torch.cuda.device_count()) == 0, \
    "The training batch size(={0}) must be set to an integer multiple of the number of GPUs(={1})"\
        .format(sum(config["training"]["batch_size"]), int(torch.cuda.device_count()))

    strong_set = FrameWiseLabeledDataset(
        tsv_dir=config["dataset"]["strong_tsv_folder"],
        dataset_dir=config["dataset"]["strong_audio_folder"],
        return_name=False,
        encoder=encoder,
    )
    weak_set = FrameWiseLabeledDataset(
        tsv_dir=config["dataset"]["weak_tsv_folder"],
        dataset_dir=config["dataset"]["weak_audio_folder"],
        return_name=False,
        encoder=encoder,
    )
    unlabeled_set = FrameWiseLabeledDataset(
        tsv_dir=config["dataset"]["unlabeled_tsv_folder"],
        dataset_dir=config["dataset"]["unlabeled_audio_folder"],
        return_name=False,
        encoder=encoder,
    )

    valid_dataset = FrameWiseLabeledDataset(
        tsv_dir=config["dataset"]["val_tsv_folder"],
        dataset_dir=config["dataset"]["val_audio_folder"],
        return_name=False,
        encoder=encoder,
    )

    tot_train_data = [strong_set, weak_set, unlabeled_set]
    train_dataset = torch.utils.data.ConcatDataset(tot_train_data)
    batch_sizes = config["training"]["batch_size"]
    samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
    batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)

    # logging
    logger.info("train: strong-{strong} ,weak-{weak}, unlabeled-{unlabeled}, val-{val}".format(
        strong=len(strong_set), weak=len(weak_set), unlabeled=len(unlabeled_set), val=len(valid_dataset)))

    return train_dataset, valid_dataset, batch_sampler
