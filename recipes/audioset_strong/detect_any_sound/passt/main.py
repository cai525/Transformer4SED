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

from recipes.audioset_strong.setting import prepare_run, get_encoder, dataset_setting, optimizer_and_scheduler_setting, BestModels
from recipes.desed.maskformer.maskformer.finetune.setting import get_param_lr
from recipes.audioset_strong.maskformer.passt.train import MaskformerTrainer
from recipes.audioset_strong.maskformer.passt.open_vocabulary import OV_Maskformer_Trainer
from src.models.maskformer.maskformer import Maskformer
from src.models.lora.utils import mark_only_lora_as_trainable

if __name__ == "__main__":
    configs, my_logger, args = prepare_run()

    # set network
    net = Maskformer(**configs["Maskformer"]["init_kwargs"])
    if configs["Maskformer"]["init_kwargs"]["backbone_param"]["lora_config"] is not None:
        mark_only_lora_as_trainable(net.backbone)
    total_lr = get_param_lr(net, configs, my_logger.logger)

    encoder = get_encoder(configs)

    train_loader, val_loader, test_loader = dataset_setting(encoder, configs, my_logger.logger)

    optimizer, scheduler = optimizer_and_scheduler_setting(total_lr, train_loader, configs, my_logger.logger)

    # Data Parallel
    if args.multigpu:
        net = nn.DataParallel(net)
    net = net.to(configs["training"]["device"])

    ##############################                TRAIN/VALIDATION                ##############################
    Trainer_class = MaskformerTrainer if not args.open_vocabulary else OV_Maskformer_Trainer
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
            trainer.net.load_state_dict(torch.load(configs['training']["best_paths"]), strict=False)
            val_metrics = trainer.validation(epoch=-1)
            bestmodels.update(trainer.net, 0, my_logger.logger, val_metrics)
            bestmodels.save_bests(configs['training']["best_paths"])

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
