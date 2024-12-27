import random
from collections import OrderedDict, namedtuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.codec.encoder import Encoder
from src.preprocess.data_aug import mixup, frame_shift, feature_transformation
from src.utils import Logger, DataParallelWrapper


class Trainer:

    def __init__(self, optimizer, my_logger: Logger, net, scheduler, encoder: Encoder, train_loader, val_loader,
                 test_loader, gmm_means: torch.Tensor, config, device):
        self.optimizer = optimizer
        self.my_logger = my_logger
        self.net = net
        self.net = DataParallelWrapper(net)
        self.config = config
        self.scheduler = scheduler
        self.encoder = encoder
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # gmm parameter
        self.gmm_means = F.normalize(gmm_means, dim=-1).to(device)  # shape: (N, C), for example (10, 768)
        # loss function
        self.bce_loss = torch.nn.BCELoss().to(device)

    ################ tool functions for training process ###################
    def get_mask(self, wavs, strong_num, weak_num, unlabel_num):
        batch_num = wavs.size(0)
        assert batch_num == strong_num + weak_num + unlabel_num
        # mask strong
        mask_strong = torch.zeros(batch_num).to(wavs).bool()
        mask_strong[:strong_num] = 1
        # mask weak
        mask_weak = torch.zeros(batch_num).to(wavs).bool()
        mask_weak[strong_num:(strong_num + weak_num)] = 1  # mask_weak size = [bs]
        # mask unlabel
        mask_unlabel = torch.zeros(batch_num).to(wavs).bool()
        mask_unlabel[(strong_num + weak_num):] = 1  # mask_weak size = [bs]
        return mask_strong, mask_weak, mask_unlabel

    def preprocess(self, wav, label, strong_mask, weak_mask, unlabel_mask):
        # Transform to mel respectively
        extractor = self.net.get_feature_extractor()
        mel = extractor(wav)
        # normalization
        mel = extractor.normalize(mel)
        # data augumentation
        # time shift
        mel, label = frame_shift(mel, label, net_pooling=self.encoder.net_pooling)
        # mixup (frequence)
        if random.random() < 0.5:
            mel, label = mixup(mel, label)
        # Do label-independent augmentation
        mel = feature_transformation(mel, log=True, norm_std=5.0, **self.config["training"]["transform"])
        return mel, label

    @property
    def train_epoch_len(self):
        if not hasattr(self, "_train_epoch_len"):
            _train_epoch_len = len(self.train_loader)
        return _train_epoch_len

    def train_log(self, log_dict: OrderedDict, epoch):
        for key, value in log_dict.items():
            self.my_logger.tensorboard_writer.add_scalar("Train/{key}".format(key=key), value, global_step=epoch)
        message = ["=============== train  epoch{0} =================".format(epoch)]
        for i, k in enumerate(log_dict):
            if not i % 3:
                message.append("\n")
            message.append("   {k} = {v:.5f};".format(k=k, v=log_dict[k]))

        message.append("\n")
        self.my_logger.logger.info("".join(message))

    def get_predict_from_logit(self, logit, temperature=0.1):
        # get consine similarity
        logit = F.normalize(logit, dim=-1) @ self.gmm_means.T
        logit = F.leaky_relu(logit, negative_slope=0.2) * 2 - 1
        pred = torch.sigmoid(logit / temperature)
        return pred

    def train(self, epoch):
        # set  train mode
        self.net.train()
        n_train = len(self.train_loader)
        # set batch size
        strong_num, weak_num, unlabel_num = self.config["training"]["batch_size"]
        log_dict = OrderedDict([("loss_total", 0), ("loss_strong", 0), ("loss_weak", 0)])

        tk = tqdm(self.train_loader, total=n_train, leave=False, desc="training processing")

        for batch in tk:
            wav, labels, pad_mask, _ = batch
            wav, labels = wav.to(self.device), labels.to(self.device)
            mask_strong, mask_weak, mask_unlabel = self.get_mask(wav, strong_num, weak_num, unlabel_num)
            mel, labels = self.preprocess(wav, labels, mask_strong, mask_weak, mask_unlabel)
            label_weak = (torch.sum(labels, -1) >= 1).float()
            # ==================== forward process ==================
            logit, other_dict = self.net(mel, **self.config[self.net.get_model_name()]["train_kwargs"])
            mask_id_seq = other_dict["mask_id_seq"]
            strong = self.get_predict_from_logit(logit)

            # (B, C, T) -> (B, T, C), to match the shape of mask_id_seq(B, T)
            labels = labels.transpose(1, 2)
            loss_strong = self.bce_loss(strong[mask_id_seq], labels[mask_id_seq])

            if self.config["training"]["w_AT"] > 0:
                weak = other_dict['at_out']
                loss_weak = self.bce_loss(weak, label_weak)
            else:
                loss_weak = 0

            loss_total = loss_strong + self.config["training"]["w_AT"] * loss_weak

            if torch.isnan(loss_total).any():
                raise Exception("Get loss value: None")

            # optimizer step
            if self.config["training"]["clip_grad"]:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=20, norm_type=2)
            loss_total.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # self.optimizer.zero_grad()
            self.scheduler.step()
            # logging
            for k in log_dict.keys():
                v = eval(k)
                if isinstance(v, torch.Tensor):
                    log_dict[k] += v.item() / n_train
                else:
                    log_dict[k] += v / n_train

        log_dict["lr_scaler"] = self.scheduler._get_scale()
        self.train_log(log_dict, epoch + 1)
        return

    ############### tool function for validation process ##################
    def preprocess_eval(self, wavs):
        extractor = self.net.get_feature_extractor()
        mel = extractor(wavs)
        feat = extractor.normalize(mel)
        return feat

    def validation(self, epoch):
        self.net.eval()
        n_valid = len(self.val_loader)
        mean_loss = 0

        with torch.inference_mode():
            tk1 = tqdm(self.val_loader, total=n_valid, leave=False, desc="validation processing")
            for _, (wavs, labels, pad_mask, _) in enumerate(tk1, 0):
                wavs, labels, pad_mask = wavs.to(self.device), labels.to(self.device), pad_mask.to(self.device)
                mel = self.preprocess_eval(wavs)
                logit, other_dict = self.net(mel, pad_mask=pad_mask, **self.config["PaSST_CNN"]["val_kwargs"])
                mask_id_seq = other_dict["mask_id_seq"]
                strong = self.get_predict_from_logit(logit)
                mask = torch.logical_and(torch.logical_not(pad_mask), mask_id_seq)
                loss = self.bce_loss(strong[mask], labels.transpose(1, 2)[mask])
                mean_loss += loss.item() / n_valid

        self.my_logger.logger.info("[Validation]".format(epoch + 1))
        self.my_logger.logger.info("Epoch {0}: Validation loss is {1:.4f}".format(epoch + 1, mean_loss))
        self.my_logger.logger.info("Epoch {0}: cnn_weight is {1}".format(epoch + 1, self.net.merge_weight))
        return mean_loss
