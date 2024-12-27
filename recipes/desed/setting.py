import argparse
from datetime import datetime
import json
import logging
import os
import random
import warnings
import yaml
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.codec.encoder import Encoder
from src.preprocess.dataset import (StronglyLabeledDataset, WeaklyLabeledDataset, UnlabeledDataset,
                                    ConcatDatasetBatchSampler)
from src.utils import load_yaml_with_relative_ref, Logger, ExponentialDown


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


def get_encoder(config):
    with open("./meta/desed/labeldict_DESED.json") as f:
        label_dict = json.load(f, object_pairs_hook=OrderedDict)
    return Encoder(sorted(list(label_dict.keys())),
                   audio_len=config["feature"]["audio_max_len"],
                   frame_len=config["feature"]["win_length"],
                   frame_hop=config["feature"]["hopsize"],
                   net_pooling=config["feature"]["net_subsample"],
                   sr=config["feature"]["sr"])


def get_logger(save_folder, need_tensorboard=False, record_carbon=False, log_level=logging.INFO) -> Logger:
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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def prepare_run():

    # parse the argument
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--gpu',
                        default=0,
                        type=int,
                        help='selection of gpu when you run separate trainings on single server')
    parser.add_argument('--multigpu', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--random_seed', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--config_dir', type=str)
    parser.add_argument('--save_folder', type=str)
    args = parser.parse_args()

    #set configurations
    configs = load_yaml_with_relative_ref(args.config_dir)

    print("=" * 50 + "start!!!!" + "=" * 50)
    if configs["generals"]["test_only"]:
        print(" " * 40 + "<" * 10 + "test only" + ">" * 10)

    configs = get_save_directories(configs, args.save_folder)
    # set logger
    my_logger = get_logger(configs["generals"]["save_folder"], (not configs["generals"]["test_only"]),
                           log_level=eval("logging." + configs["generals"]["log_level"].upper()))

    my_logger.logger.info("date & time of start is : " + str(datetime.now()).split('.')[0])
    my_logger.logger.info("torch version is: " + str(torch.__version__))

    # set device
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    my_logger.logger.info("number of GPUs: " + str(torch.cuda.device_count()))
    configs["training"]["device"] = device
    my_logger.logger.info("device: " + str(device))

    # set seed
    if args.random_seed:
        seed = random.randint(0, 10000)
        setup_seed(seed)
        my_logger.logger.info("use random seed {}".format(seed))
        configs["training"]["seed"] = seed
    else:
        seed = configs["training"]["seed"]
        setup_seed(seed)
        my_logger.logger.info("use fix seed {}".format(seed))

    # do not show warning
    if not configs["generals"]["warn"]:
        warnings.filterwarnings("ignore")

    return configs, my_logger, args


def dataset_setting(encoder, configs, logger):
    # get dataset
    train_dataset, valid_dataset, test_dataset, batch_sampler, tot_train_data = get_datasets(
        configs,
        encoder,
        evaluation=configs["generals"]["test_on_public_eval"],
        test_only=configs["generals"]["test_only"],
        logger=logger)

    # set dataloader
    test_loader = DataLoader(test_dataset,
                             batch_size=configs["training"]["batch_size_val"],
                             num_workers=configs["training"]["num_workers"])
    if not configs["generals"]["test_only"]:
        train_loader = DataLoader(train_dataset,
                                  batch_sampler=batch_sampler,
                                  num_workers=configs["training"]["num_workers"])
        val_loader = DataLoader(valid_dataset,
                                batch_size=configs["training"]["batch_size_val"],
                                num_workers=configs["training"]["num_workers"])
    else:
        train_loader, val_loader = None, None
    return train_loader, val_loader, test_loader


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
        weak_set = WeaklyLabeledDataset(tsv_read=weak_df,
                                        dataset_dir=config["dataset"]["weak_folder"],
                                        return_name=False,
                                        encoder=encoder)

        unlabeled_set = UnlabeledDataset(dataset_dir=config["dataset"]["unlabeled_folder"],
                                         return_name=False,
                                         encoder=encoder)

        valid_strong_df = pd.read_csv(config["dataset"]["val_tsv"], sep="\t")
        valid_dataset = StronglyLabeledDataset(tsv_read=valid_strong_df,
                                               dataset_dir=config["dataset"]["val_folder"],
                                               return_name=True,
                                               encoder=encoder)

        tot_train_data = [strong_set, synth_set, weak_set, unlabeled_set]
        train_dataset = torch.utils.data.ConcatDataset(tot_train_data)

        batch_sizes = config["training"]["batch_size"]
        samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
        batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)

        # logging
        logger.info("train: strong-{strong}, syn-{syn} ,weak-{weak}, unlabeled-{unlabeled}".format(
            strong=len(strong_set), syn=len(synth_set), weak=len(weak_set), unlabeled=len(unlabeled_set)))
        logger.info("val: {}".format(len(valid_dataset)))
    else:
        train_dataset = None
        valid_dataset = None
        batch_sampler = None
        tot_train_data = None

    logger.info("test: {0}".format(len(test_dataset)))
    return train_dataset, valid_dataset, test_dataset, batch_sampler, tot_train_data


def optimizer_and_scheduler_setting(lr, train_loader, configs, logger):
    # set learning rate
    optim_kwargs = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-8}

    optimizer = optim.AdamW(lr, **optim_kwargs)

    if not configs["generals"]["test_only"]:
        total_iter = configs["training"]["scheduler"]["n_epochs"] * len(train_loader)
        start_iter = configs["training"]["scheduler"]["n_epochs_cut"] * len(train_loader)

        logger.info("learning rate keep no change until iter{}, then expdown, total iter:{}".format(
            start_iter, total_iter))

        scheduler = ExponentialDown(optimizer=optimizer,
                                    start_iter=start_iter,
                                    total_iter=total_iter,
                                    exponent=configs["training"]["scheduler"]['exponent'],
                                    warmup_iter=configs["training"]["scheduler"]["lr_warmup_epochs"] *
                                    len(train_loader),
                                    warmup_rate=configs["training"]["scheduler"]["lr_warmup_rate"])

        logger.info("learning rate warmup until iter{}, then keep, total iter:{}".format(start_iter, total_iter))
    else:
        scheduler = None
    return optimizer, scheduler
