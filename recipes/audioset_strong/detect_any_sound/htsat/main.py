import os.path
import sys
from datetime import datetime
from time import time

import torch
import torch.nn as nn

root = "/home/cpf/code/SSL_SED"
os.chdir(root)
sys.path.append(root)

from recipes.audioset_strong.setting import prepare_run, get_encoder, dataset_setting, optimizer_and_scheduler_setting, BestModels
from recipes.audioset_strong.detect_any_sound.htsat.train import DASM_HTSAT_Trainer
from recipes.audioset_strong.detect_any_sound.htsat.open_vocabulary import OV_DASM_HTSAT_Trainer
from recipes.audioset_strong.detect_any_sound.htsat.lr_set import get_lr_htsat
from src.models.detect_any_sound.detect_any_sound_htast import DASM_HTSAT

if __name__ == "__main__":
    configs, my_logger, args = prepare_run()

    # set network
    net = DASM_HTSAT(**configs["DASM_HTSAT"]["init_kwargs"])
    total_lr = get_lr_htsat(net, configs, my_logger.logger)

    encoder = get_encoder(configs)

    train_loader, val_loader, test_loader = dataset_setting(encoder, configs, my_logger.logger)
    optimizer, scheduler = optimizer_and_scheduler_setting(total_lr, train_loader, configs, my_logger.logger)

    #### move to gpus ########
    if args.multigpu:
        net = nn.DataParallel(net)
    net = net.to(configs["training"]["device"])

    ##############################                TRAIN/VALIDATION                ##############################
    Trainer_class = DASM_HTSAT_Trainer if not args.open_vocabulary else OV_DASM_HTSAT_Trainer
    trainer = Trainer_class(optimizer=optimizer,
                            my_logger=my_logger,
                            net=net,
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
            params_dict = torch.load(configs['training']["best_paths"])
            params = {
                k: v
                for k, v in params_dict.items()
                if all(keyword not in k for keyword in ["at_query", "query_projector", "at_head.", "sed_head."])
            }
            trainer.net.load_state_dict(params, strict=False)

        # load existed model
        elif configs['generals']["load_from_existed_path"]:
            trainer.net.load_state_dict(torch.load(configs['training']["best_paths"]))
            val_metrics = trainer.validation(epoch=-1)
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
    params_dict = torch.load(configs['training']["best_paths"])
    params = {k: v for k, v in params_dict.items() if 'at_query' not in k}
    trainer.net.load_state_dict(params, strict=False)

    trainer.test()

    my_logger.logger.info("date & time of end is : " + str(datetime.now()).split('.')[0])

    print("<" * 30 + "DONE!" + ">" * 30)
    my_logger.close()
