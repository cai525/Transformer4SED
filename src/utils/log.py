import copy
import logging
import os

import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self) -> None:
        self.logger = None
        self.tensorboard_writer = None
    
    def init_logger(self, handler_list, level=logging.INFO):
        self.logger = logging.getLogger()
        while len(self.logger.handlers) > 0:  # check if handler already exists
            self.logger.removeHandler(self.logger.handlers[0])  
        self.logger.setLevel(level)
        for handler in handler_list:
            self.logger.addHandler(handler)
        return
    
    def init_tensorboard_writer(self, save_folder):
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder, exist_ok=True)  # saving folder
        self.tensorboard_writer = SummaryWriter(save_folder)
        return
    
    def __del__(self):
        """ destructor """
        if self.logger:
            logging.shutdown()
        if self.tensorboard_writer:
            self.tensorboard_writer.close()

class BestModels:
    # Class to keep track of the best student/teacher models and save them after training
    def __init__(self):
        #self.stud_best_val_metric = 0.0
        #self.tch_best_val_metric = 0.0
        self.stud_best_val_metric = -0.00001
        self.tch_best_val_metric = -0.00001
        self.stud_best_state_dict = None
        self.tch_best_state_dict = None

    def update(self, net, ema_net, epoch, logger, val_metrics):
        stud_update = False
        tch_update = False
        if val_metrics[0] > self.stud_best_val_metric:
            self.stud_best_val_metric = val_metrics[0]
            #self.stud_best_state_dict = train_cfg["net"].state_dict()
            self.stud_best_state_dict = copy.deepcopy(net.state_dict())

            stud_update = True
            # lr_reduc = 0
        if val_metrics[1] > self.tch_best_val_metric:
            self.tch_best_val_metric = val_metrics[1]
            #self.tch_best_state_dict = train_cfg["ema_net"].state_dict()
            self.tch_best_state_dict = copy.deepcopy(ema_net.state_dict())

            tch_update = True
            # lr_reduc = 0

        #if train_cfg["epoch"] > int(train_cfg["n_epochs"] * 0.5):
        if epoch >= 0:
            if stud_update:
                if tch_update:
                    logger.info("     best student & teacher model updated at epoch %d!" % epoch)
                else:
                    logger.info("     best student model updated at epoch %d!" % epoch)
            elif tch_update:
                logger.info("     best teacher model updated at epoch %d!" % epoch)
        return 

    def save_bests(self, best_paths):
        torch.save(self.stud_best_state_dict, best_paths[0])
        torch.save(self.tch_best_state_dict, best_paths[1])
        return self.stud_best_val_metric, self.tch_best_val_metric