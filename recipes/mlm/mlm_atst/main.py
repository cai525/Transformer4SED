import argparse
import copy
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

root = "/home/cpf/code/open/Transformer4SED"
os.chdir(root)
sys.path.append(root)

from recipes.mlm.mlm_atst.atst_mlm_setting import *
from recipes.mlm.mlm_atst.train import train
from recipes.mlm.mlm_atst.validation import validation

from src.utils.statistics.model_statistic import count_parameters
from src.utils.scheduler import ExponentialDown


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
    parser.add_argument('--config_server', default="./config/config_server.yaml", type=str)
    parser.add_argument('--save_folder', type=str)
    args = parser.parse_args()

    #set configurations
    configs, train_cfg = get_configs(config_dir=args.config_dir)

    #declare test_only/debugging mode
    if train_cfg["test_only"]:
        print(" " * 40 + "<" * 10 + "test only" + ">" * 10)
    if train_cfg["debug"]:
        train_cfg["div_dataset"] = True
        train_cfg["n_epochs"] = 1
        print("!" * 10 + "   DEBUGGING MODE   " + "!" * 10)

    #set save directories
    configs, train_cfg = get_save_directories(configs, train_cfg, args.save_folder)

    #set logger
    logger = get_logger(configs["generals"]["save_folder"])
    train_cfg['logger'] = logger
    #torch information
    logger.info("date & time of start is : " + str(datetime.now()).split('.')[0])
    logger.info("torch version is: " + str(torch.__version__))
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    train_cfg["n_gpu"] = torch.cuda.device_count()
    logger.info("number of GPUs: " + str(train_cfg["n_gpu"]))
    train_cfg["device"] = device
    logger.info("device: " + str(device))

    #seed
    if args.random_seed:
        seed = random.randint(0, 10000)
        setup_seed(seed)
        logger.info("use random seed {}".format(seed))
        train_cfg["seed"] = seed  #add in 2022 12.8, weak split will use the seed

    else:
        seed = train_cfg["seed"]
        setup_seed(seed)
        logger.info("use fix seed{}".format(seed))

    #do not show warning
    if not configs["generals"]["warn"]:
        warnings.filterwarnings("ignore")

    #class label dictionary
    LabelDict = get_labeldict()

    #set encoder
    train_cfg["encoder"] = get_encoder(LabelDict, configs)

    #set Dataloaders
    train_cfg = get_datasets_mlm(configs, train_cfg)

    #set network
    train_cfg["net"] = get_model_atst(configs)

    logger.info("Total Trainable Params: %.3f M" %
                (count_parameters(train_cfg["net"]) * 1e-6))  #print number of learnable parameters in the model

    train_cfg["criterion_cons"] = nn.MSELoss().cuda()

    # optimizer
    optim_kwargs = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-4}

    patch_transformer_params = list(map(id, train_cfg["net"].get_encoder().parameters()))
    base_params = filter(lambda p: id(p) not in patch_transformer_params, train_cfg['net'].parameters())

    train_cfg["optimizer"] = optim.AdamW([{"params": base_params, 'lr': configs['opt']['lr_big']}], **optim_kwargs)

    total_iter = train_cfg["n_epochs"] * len(train_cfg['trainloader'])
    start_iter = train_cfg["n_epochs_cut"] * len(train_cfg['trainloader'])
    train_cfg['total_iter'] = total_iter
    train_cfg["dataset_len"] = len(train_cfg['trainloader'])

    if train_cfg['scheduler_name'] == 'ExponentialDown':
        logger.info("learning rate keep no change until iter{}, then expdown, total iter:{}".format(
            start_iter, total_iter))

        train_cfg["scheduler"] = ExponentialDown(train_cfg["optimizer"],
                                                 start_iter=start_iter,
                                                 total_iter=total_iter,
                                                 exponent=configs['opt']['exponent'],
                                                 warmup_iter=train_cfg["lr_warmup_epochs"] *
                                                 len(train_cfg['trainloader']),
                                                 warmup_rate=train_cfg["lr_warmup_rate"])

        logger.info("learning rate warmup until iter{}, then keep, total iter:{}".format(start_iter, total_iter))
    else:
        raise ValueError('invalid scheduler_name')

    ####move to gpus########
    if args.multigpu and train_cfg['n_gpu'] > 1:
        train_cfg['net'] = nn.DataParallel(train_cfg['net'])
    train_cfg['net'] = train_cfg['net'].to(train_cfg['device'])

    ##############################                TRAIN/VALIDATION                ##############################
    best_val_metrics = float('inf')
    stud_best_state_dict = None
    if not train_cfg["test_only"]:
        # load existed model
        if train_cfg["load_from_existed_path"]:
            train_cfg["net"].load_state_dict(torch.load(train_cfg["best_paths"]))

        logger.info('   training starts!')
        start_time = time()
        for train_cfg["epoch"] in range(train_cfg["n_epochs"]):
            print("[epoch {0}]".format(train_cfg["epoch"]))
            train(train_cfg=train_cfg)
            val_metrics = validation(train_cfg)
            if val_metrics < best_val_metrics:
                # Remember not to save the reference!
                stud_best_state_dict = copy.deepcopy(train_cfg["net"].state_dict())
                torch.save(stud_best_state_dict, train_cfg["best_paths"])
                best_val_metrics = val_metrics

        logger.info("   training took %.2f mins" % ((time() - start_time) / 60))

    ##############################                        TEST                        ##############################
    logger.info("Best validation metrics is {0}".format(best_val_metrics))
    logger.info("date & time of end is : " + str(datetime.now()).split('.')[0])
    logging.shutdown()
    print("<" * 30 + "DONE!" + ">" * 30)


if __name__ == "__main__":
    n_repeat = 1
    for iter in range(n_repeat):
        main(iter)
