import argparse
import copy
import logging
import os.path
import random
import sys
from datetime import datetime
from time import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

root = "ROOT-PATH"
os.chdir(root)
sys.path.append(root)

from recipes.desed.setting import get_encoder, get_logger
from recipes.desed.pmam.setting import get_pseudo_dataset, get_save_directories
from recipes.desed.pmam.train import Trainer
from src.models.cnn_transformer.passt_cnn import PaSST_CNN
from src.models.passt.passt_sed import PaSST_SED
from src.models.lora import mark_only_lora_as_trainable
from src.utils.statistics.model_statistic import count_parameters
from src.utils import load_yaml_with_relative_ref, ExponentialDown


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
    parser.add_argument('--gmm_means_path', type=str)
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

    return configs, my_logger, args


if __name__ == "__main__":
    configs, my_logger, args = prepare_run()

    if "PaSST_SED" in configs.keys():
        from src.models.passt.passt_sed import PaSST_SED
        from recipes.desed.finetune.passt.setting import get_params

        net = PaSST_SED(**configs["PaSST_SED"]["init_kwargs"])
        class_num = configs["PaSST_SED"]["init_kwargs"]["class_num"]

    elif "PaSST_CNN" in configs.keys():
        from src.models.cnn_transformer.passt_cnn import PaSST_CNN
        from recipes.desed.finetune.cnn_trans.setting import get_param_lr

        net = PaSST_CNN(**configs["PaSST_CNN"]["init_kwargs"])
        class_num = configs["PaSST_CNN"]["init_kwargs"]["passt_sed_param"]["class_num"]
    else:
        raise RuntimeError("Unknown model structure.")

    mark_only_lora_as_trainable(net.patch_transformer)

    # set encoder
    encoder = get_encoder(configs)

    # get gmm means
    gmm_means = torch.load(args.gmm_means_path)
    # if gmm_means.ndim == 2:
    #     gmm_means = gmm_means.unsqueeze(0)
    my_logger.logger.info("gmm's shape: {}".format(gmm_means.shape))

    # get dataset
    train_dataset, valid_dataset, batch_sampler = get_pseudo_dataset(
        configs,
        encoder,
        logger=my_logger.logger,
    )

    # logger.info("---------------model structure---------------")
    # logger.info(train_cfg['net'])

    # set dataloader
    train_loader = DataLoader(train_dataset,
                              batch_sampler=batch_sampler,
                              num_workers=configs["training"]["num_workers"])
    val_loader = DataLoader(valid_dataset,
                            batch_size=configs["training"]["batch_size_val"],
                            num_workers=configs["training"]["num_workers"])

    # set learning rate
    optim_kwargs = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-8}
    total_lr = get_param_lr(net, configs, my_logger.logger)

    optimizer = optim.AdamW(total_lr, **optim_kwargs)

    my_logger.logger.info("Total  Params: %.3f M" % (count_parameters(net, trainable_only=False) * 1e-6))

    my_logger.logger.info("Total Trainable Params: %.3f M" % (count_parameters(net) * 1e-6))

    total_iter = configs["training"]["scheduler"]["n_epochs"] * len(train_loader)
    start_iter = configs["training"]["scheduler"]["n_epochs_cut"] * len(train_loader)

    my_logger.logger.info("learning rate keep no change until iter{}, then expdown, total iter:{}".format(
        start_iter, total_iter))

    scheduler = ExponentialDown(
        optimizer=optimizer,
        start_iter=start_iter,
        total_iter=total_iter,
        exponent=configs["training"]["scheduler"]['exponent'],
        warmup_iter=configs["training"]["scheduler"]["lr_warmup_epochs"] * len(train_loader),
        warmup_rate=configs["training"]["scheduler"]["lr_warmup_rate"],
    )

    my_logger.logger.info("learning rate warmup until iter{}, then keep, total iter:{}".format(start_iter, total_iter))

    #### move to gpus ########
    if args.multigpu:
        net = nn.DataParallel(net)

    net = net.to(configs["training"]["device"])

    ##############################                TRAIN/VALIDATION                ##############################
    trainer = Trainer(optimizer=optimizer,
                      my_logger=my_logger,
                      net=net,
                      config=configs,
                      encoder=encoder,
                      scheduler=scheduler,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      test_loader=None,
                      gmm_means=gmm_means,
                      device=configs["training"]["device"])

    my_logger.logger.info('   training starts!')
    start_time = time()
    best_val_metrics = float('inf')
    best_model_state_dict = None

    # load existed model
    if configs['generals']["load_from_existed_path"]:
        params_dict = torch.load(configs['generals']["best_paths"])
        params_dict = {k: v for k, v in params_dict.items() if "mlm_mlp." not in k}
        trainer.net.load_state_dict(params_dict, strict=False)
        val_metrics = trainer.validation(epoch=-1)
        best_val_metrics = val_metrics

    for epoch in range(configs['training']["scheduler"]["n_epochs"]):
        epoch_time = time()
        #training
        trainer.train(epoch)
        # validation
        val_metrics = trainer.validation(epoch)
        if val_metrics < best_val_metrics:
            trainer.net.eval()  # to merge lora weight to the pretrain weight
            best_val_metrics = val_metrics
            best_model_state_dict = copy.deepcopy(trainer.net.state_dict())
            torch.save(best_model_state_dict, configs["generals"]["best_paths"])
            my_logger.logger.info("New best model at epoch {0}!".format(epoch + 1))
        my_logger.logger.info("\n\n")
    #save model parameters & history dictionary
    my_logger.logger.info("Best validation metrics is {0}".format(best_val_metrics))
    my_logger.logger.info("   training took %.2f mins" % ((time() - start_time) / 60))

    my_logger.logger.info("date & time of end is : " + str(datetime.now()).split('.')[0])

    print("<" * 30 + "DONE!" + ">" * 30)
    my_logger.close()
