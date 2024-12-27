import logging
import os.path
import sys
from datetime import datetime
from time import time

import torch
import torch.nn as nn

root = "ROOT-PATH"
os.chdir(root)
sys.path.append(root)

from recipes.desed.setting import prepare_run, get_encoder, dataset_setting, optimizer_and_scheduler_setting
from recipes.desed.finetune.cnn_trans.setting import get_models_passt_cnn, get_param_lr
from recipes.desed.finetune.cnn_trans.train import PaSST_CNN_Trainer
from src.utils import BestModels

if __name__ == "__main__":
    configs, my_logger, args = prepare_run()

    # set network
    net, ema_net = get_models_passt_cnn(configs)
    total_lr = get_param_lr(net, configs, my_logger.logger)

    encoder = get_encoder(configs)
    train_loader, val_loader, test_loader = dataset_setting(encoder, configs, my_logger.logger)
    optimizer, scheduler = optimizer_and_scheduler_setting(total_lr, train_loader, configs, my_logger.logger)

    #### move to gpus ########
    if args.multigpu:
        net = nn.DataParallel(net)
        ema_net = nn.DataParallel(ema_net)
    else:
        logging.warning("Run with only single GPU!")
    net = net.to(configs["training"]["device"])
    ema_net = ema_net.to(configs["training"]["device"])

    ##############################                TRAIN/VALIDATION                ##############################
    trainer = PaSST_CNN_Trainer(optimizer=optimizer,
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
        if ("finetune_mlm" in configs['generals'].keys()) and configs['generals']["finetune_mlm"]:
            params_dict = torch.load(configs['training']["best_paths"][0])
            params = {k: v for k, v in params_dict.items() if "classifier." not in k and "at_adpater.1" not in k}
            trainer.ema_net.load_state_dict(params, strict=False)
            trainer.net.load_state_dict(params, strict=False)

        # load existed model
        elif configs['generals']["load_from_existed_path"]:
            trainer.ema_net.load_state_dict(torch.load(configs['training']["best_paths"][1]))
            trainer.net.load_state_dict(torch.load(configs['training']["best_paths"][0]))
            val_metrics = trainer.validation(epoch=-1)
            bestmodels.update(net, ema_net, 0, my_logger.logger, val_metrics)

        for epoch in range(configs['training']["scheduler"]["n_epochs"]):
            epoch_time = time()
            #training
            trainer.train(epoch)
            # validation
            if (not epoch % configs["generals"]["validation_interval"]) or (
                    epoch == configs['training']["scheduler"]["n_epochs"] - 1):
                val_metrics = trainer.validation(epoch)
                bestmodels.update(trainer.net, trainer.ema_net, epoch + 1, my_logger.logger, val_metrics)
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
