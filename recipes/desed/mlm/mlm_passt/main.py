import argparse
import copy
import logging
import os
import os.path
import random
import warnings
import sys
from datetime import datetime
from time import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

root = "ROOT-PATH"
os.chdir(root)
sys.path.append(root)

from recipes.desed.setting import get_logger, get_encoder
from recipes.desed.mlm.mlm_passt.passt_mlm_setting import get_datasets_mlm, get_save_directories, get_model_passt
from recipes.desed.mlm.mlm_passt.train import MLMTrainer
from recipes.desed.finetune.passt.setting import get_params
from src.utils import ExponentialDown, count_parameters, load_yaml_with_relative_ref


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def str2bool(str):
    return True if str.lower == 'true' else False


def main(iteration=None):
    print("=" * 50 + "start!!!!" + "=" * 50)
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--gpu',
                        default=0,
                        type=int,
                        help='selection of gpu when you run separate trainings on single server')
    parser.add_argument('--multigpu', default=False, type=bool)
    parser.add_argument('--random_seed', default=False, type=bool)
    parser.add_argument('--config_dir', type=str)
    parser.add_argument('--save_folder', type=str)
    args = parser.parse_args()

    #set configurations
    configs = load_yaml_with_relative_ref(args.config_dir)

    #set save directories
    configs = get_save_directories(configs, args.save_folder)

    #set logger
    logger = get_logger(configs["generals"]["save_folder"], log_level=configs["generals"]["log_level"]).logger
    #torch information
    logger.info("date & time of start is : " + str(datetime.now()).split('.')[0])
    logger.info("torch version is: " + str(torch.__version__))
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    logger.info("device: " + str(device))

    #seed
    if args.random_seed:
        seed = random.randint(0, 10000)
        setup_seed(seed)
        logger.info("use random seed {}".format(seed))
        configs["training"]["seed"] = seed  #add in 2022 12.8, weak split will use the seed
    else:
        seed = configs["training"]["seed"]
        setup_seed(seed)
        logger.info("use fix seed{}".format(seed))

    #do not show warning
    if not configs["generals"]["warn"]:
        warnings.filterwarnings("ignore")

    #set encoder
    encoder = get_encoder(configs)

    #set Dataloaders
    train_loader, valid_loader = get_datasets_mlm(configs, encoder)

    #set network
    net = get_model_passt(configs)

    logger.info("Total Trainable Params: %.3f M" %
                (count_parameters(net) * 1e-6))  #print number of learnable parameters in the model

    # optimizer
    optim_kwargs = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-4}

    total_lr = get_params(net, configs, logger)
    optimizer = optim.AdamW(total_lr, **optim_kwargs)

    total_iter = configs["training"]["n_epochs"] * len(train_loader)
    start_iter = configs["training"]["n_epochs_cut"] * len(train_loader)

    if configs["training"]["scheduler_name"] == 'ExponentialDown':
        logger.info("learning rate keep no change until iter{}, then expdown, total iter:{}".format(
            start_iter, total_iter))

        scheduler = ExponentialDown(optimizer,
                                    start_iter=start_iter,
                                    total_iter=total_iter,
                                    exponent=configs['opt']['exponent'],
                                    warmup_iter=configs["training"]["lr_warmup_epochs"] * len(train_loader),
                                    warmup_rate=configs["training"]["lr_warmup_rate"])

        logger.info("learning rate warmup until iter{}, then keep, total iter:{}".format(start_iter, total_iter))
    else:
        raise ValueError('invalid scheduler_name:{name}'.format(name=configs["training"]["scheduler_name"]))

    ####move to gpus########
    if args.multigpu and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net = net.to(device)

    ##############################                TRAIN/VALIDATION                ##############################
    best_val_metrics = float('inf')
    stud_best_state_dict = None

    # load existed model
    if configs["generals"]["load_from_existed_path"]:
        pretrain_model_path = configs["generals"].get("pretrain_model_path", configs["generals"]["best_paths"])
        logger.info("loading pretrained model from {path}".format(path=pretrain_model_path))
        net.load_state_dict(torch.load(pretrain_model_path), strict=False)

    trainer = MLMTrainer(net=net,
                         train_loader=train_loader,
                         val_loader=valid_loader,
                         config=configs,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         encoder=encoder,
                         logger=logger,
                         device=device)

    logger.info('   training starts!')
    start_time = time()
    for epoch in range(configs["training"]["n_epochs"]):
        logger.info("[epoch {0}]".format(epoch))
        trainer.train(epoch)
        val_metrics = trainer.validation(epoch)
        if val_metrics < best_val_metrics:
            # Remember not to save the reference!
            stud_best_state_dict = copy.deepcopy(net.state_dict())
            torch.save(stud_best_state_dict, configs["generals"]["best_paths"])
            best_val_metrics = val_metrics
            logger.info("Get best val metric at epoch {epoch}".format(epoch=epoch))

    logger.info("   training took %.2f mins" % ((time() - start_time) / 60))
    logger.info("Best validation metrics is {0}".format(best_val_metrics))
    logger.info("date & time of end is : " + str(datetime.now()).split('.')[0])
    logging.shutdown()
    print("<" * 30 + "DONE!" + ">" * 30)


if __name__ == "__main__":
    n_repeat = 1
    for iter in range(n_repeat):
        main(iter)
