import argparse
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
from torch.utils.data import DataLoader

root = "/home/cpf/code/open/Transformer4SED"
os.chdir(root)
sys.path.append(root)

from recipes.finetune.atst.atst_setting import *
from recipes.finetune.atst.atst_train import AtstTrainer
from src.utils.statistics.model_statistic import count_parameters
from src.utils.log import BestModels
from src.utils.scheduler import ExponentialDown


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
    configs = get_configs(config_dir=args.config_dir)

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


if __name__ == "__main__":
    configs, my_logger, args = prepare_run()

    # set network
    net, ema_net = get_models_atst(configs)

    # class label dictionary
    LabelDict = get_labeldict()

    # set encoder
    encoder = get_encoder(LabelDict, configs)

    # get dataset
    train_dataset, valid_dataset, test_dataset, batch_sampler, tot_train_data = get_datasets(
        configs,
        encoder,
        evaluation=configs["generals"]["test_on_public_eval"],
        test_only=configs["generals"]["test_only"],
        logger=my_logger.logger)

    # logger.info("---------------model structure---------------")
    # my_logger.logger.info(net)
    # for k, p in net.named_parameters():
    #     my_logger.logger.info(k)

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

    # set learning rate
    optim_kwargs = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-8}
    total_lr = get_params(net, configs, my_logger.logger)

    optimizer = optim.AdamW(total_lr, **optim_kwargs)

    my_logger.logger.info("Total  Params: %.3f M" % (count_parameters(net, trainable_only=False) * 1e-6))

    my_logger.logger.info("Total Trainable Params: %.3f M" % (count_parameters(net) * 1e-6))

    if not configs["generals"]["test_only"]:
        total_iter = configs["training"]["n_epochs"] * len(train_loader)
        start_iter = configs["training"]["n_epochs_cut"] * len(train_loader)

        my_logger.logger.info("learning rate keep no change until iter{}, then expdown, total iter:{}".format(
            start_iter, total_iter))

        scheduler = ExponentialDown(optimizer=optimizer,
                                    start_iter=start_iter,
                                    total_iter=total_iter,
                                    exponent=configs['opt']['exponent'],
                                    warmup_iter=configs["training"]["lr_warmup_epochs"] * len(train_loader),
                                    warmup_rate=configs["training"]["lr_warmup_rate"])

        my_logger.logger.info("learning rate warmup until iter{}, then keep, total iter:{}".format(
            start_iter, total_iter))
    else:
        scheduler = None

    #### move to gpus ########
    if args.multigpu:
        net = nn.DataParallel(net)
        ema_net = nn.DataParallel(ema_net)
    else:
        logging.warning("Run with only single GPU!")
    net = net.to(configs["training"]["device"])
    ema_net = ema_net.to(configs["training"]["device"])

    ##############################                TRAIN/VALIDATION                ##############################
    trainer = AtstTrainer(optimizer=optimizer,
                          my_logger=my_logger,
                          net=net,
                          ema_net=ema_net,
                          config=configs,
                          encoder=encoder,
                          scheduler=scheduler,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          test_loader=test_loader,
                          device=configs["training"]["device"])

    if not configs['generals']["test_only"]:
        my_logger.logger.info('   training starts!')
        start_time = time()
        bestmodels = BestModels()

        # load existed model
        if configs['generals'].get("finetune_mlm", None):
            params = torch.load(configs['training']["best_paths"][0])
            # params = {k: v
            #           for k, v in params_dict.items() if ".decoder" in k}  # only load decoder part from pretrain model
            trainer.ema_net.load_state_dict(params, strict=False)
            trainer.net.load_state_dict(params, strict=False)
            my_logger.logger.info("<INFO> load pretrained masked language model")

        # load existed model
        elif configs['generals'].get("load_from_existed_path", None):
            trainer.ema_net.load_state_dict(torch.load(configs['training']["best_paths"][1]))
            trainer.net.load_state_dict(torch.load(configs['training']["best_paths"][0]))
            my_logger.logger.info("<INFO> load existed model")

            val_metrics = trainer.validation(epoch=-1)
            bestmodels.update(net, ema_net, 0, my_logger.logger, val_metrics)

        for epoch in range(configs['training']["n_epochs"]):
            epoch_time = time()
            #training
            trainer.train(epoch)
            # validation
            if (not epoch % configs["generals"]["validation_interval"]) or (epoch
                                                                            == configs['training']["n_epochs"] - 1):
                val_metrics = trainer.validation(epoch)
                bestmodels.update(net, ema_net, epoch + 1, my_logger.logger, val_metrics)
            if not epoch % 2:
                bestmodels.save_bests(configs['training']["best_paths"])

        #save model parameters & history dictionary
        my_logger.logger.info("        best student/teacher val_metrics: %.3f / %.3f" %
                              bestmodels.save_bests(configs['training']["best_paths"]))
        my_logger.logger.info("   training took %.2f mins" % ((time() - start_time) / 60))

    ##############################                        TEST                        ##############################
    my_logger.logger.info("   test starts!")
    # test on best model
    trainer.net.load_state_dict(torch.load(configs['training']["best_paths"][0]))
    trainer.ema_net.load_state_dict(torch.load(configs['training']["best_paths"][1]))

    trainer.test()

    my_logger.logger.info("date & time of end is : " + str(datetime.now()).split('.')[0])

    print("<" * 30 + "DONE!" + ">" * 30)
    my_logger.close()
