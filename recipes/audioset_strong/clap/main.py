import logging
import os.path
import sys
from datetime import datetime
from time import time

import torch
import torch.nn as nn

root = "/home/cpf/code/SSL_SED"
os.chdir(root)
sys.path.append(root)

from recipes.audioset_strong.setting import prepare_run, dataset_setting, get_encoder, optimizer_and_scheduler_setting, BestModels
from recipes.audioset_strong.base.htsat_cnn.train import HTSAT_CNN_Trainer
from recipes.audioset_strong.clap.train import CommonOnlyClapTrainer
from recipes.audioset_strong.base.htsat_cnn.set_lr import get_param_lr
from src.models.htsat.clap_sed import CLAP_SED

if __name__ == "__main__":
    configs, my_logger, args = prepare_run()

    # set network
    net = CLAP_SED(**configs["CLAP_SED"]["init_kwargs"])
    total_lr = get_param_lr(net, configs, my_logger.logger)

    # set encoder
    encoder = get_encoder(configs)

    train_loader, val_loader, test_loader = dataset_setting(encoder, configs, my_logger.logger)

    optimizer, scheduler = optimizer_and_scheduler_setting(total_lr, train_loader, configs, my_logger.logger)

    # data parallel
    if args.multigpu:
        net = nn.DataParallel(net)
    net = net.to(configs["training"]["device"])

    ##############################                TRAIN/VALIDATION                ##############################
    Trainer_type = CommonOnlyClapTrainer if args.open_vocabulary else HTSAT_CNN_Trainer
    trainer = Trainer_type(
        optimizer=optimizer,
        my_logger=my_logger,
        net=net,
        config=configs,
        encoder=encoder,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=configs["training"]["device"],
    )

    if not configs['generals']["test_only"]:
        my_logger.logger.info('   training starts!')
        start_time = time()
        bestmodels = BestModels()

        # load existed model
        if configs['generals']["load_from_existed_path"]:
            trainer.net.load_state_dict(torch.load(configs['training']["best_paths"]))

        # training / validation
        for epoch in range(configs['training']["scheduler"]["n_epochs"]):
            epoch_time = time()
            #training
            trainer.train(epoch)
            # validation
            val_metrics = trainer.validation(epoch)
            bestmodels.update(trainer.net, epoch + 1, my_logger.logger, val_metrics)
            bestmodels.save_bests(configs['training']["best_paths"])

        #save model parameters & history dictionary
        my_logger.logger.info("        best student/teacher val_metrics: %.3f" %
                              bestmodels.save_bests(configs['training']["best_paths"]))
        my_logger.logger.info("   training took %.2f mins" % ((time() - start_time) / 60))

    ##############################                        TEST                        ##############################
    my_logger.logger.info("   test starts!")
    # test on best model
    trainer.net.load_state_dict(torch.load(configs['training']["best_paths"]))

    trainer.test()

    my_logger.logger.info("date & time of end is : " + str(datetime.now()).split('.')[0])

    print("<" * 30 + "DONE!" + ">" * 30)
    my_logger.close()
