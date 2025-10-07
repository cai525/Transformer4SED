from collections import OrderedDict

from tqdm import tqdm
import torch

from recipes.audioset_strong.base.htsat_cnn.train import HTSAT_CNN_Trainer


class CommonOnlyClapTrainer(HTSAT_CNN_Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, epoch):
        # set  train mode
        self.net.train()
        n_train = len(self.train_loader)
        log_dict = OrderedDict([
            ("loss_class_strong", 0),
        ])

        tk = tqdm(self.train_loader, total=n_train, leave=False, desc="training processing")
        query = self.net.text_query[self.common_type_mask, :]
        for i, batch in enumerate(tk):
            wav, labels, _, _ = batch
            wav, labels = wav.to(self.device), labels.to(self.device)
            labels = labels[:, self.common_type_mask, :]
            self.optimizer.zero_grad()
            feat, labels, labels_weak = self.preprocess(wav, labels)
            # ==================== forward process ==================
            pred = self.net(feat, query=query, **self.config[self.net.get_model_name()]["train_kwargs"])
            pred = self.Pred(*pred)
            if torch.isnan(pred.strong).any():
                raise Exception("Error: Out of range")
            # ==================== calculate loss ====================
            # supervised_loss for SED branch
            loss_class_strong = self.supervised_loss(pred.strong, labels)
            # total loss
            loss_total = loss_class_strong

            # optimizer step
            if self.config["training"]["clip_grad"]:
                torch.nn.utils.clip_grad_norm(self.net.parameters(), max_norm=20, norm_type=2)
            loss_total.backward()
            self.optimizer.step()
            self.scheduler.step()

            # logging
            tk.set_postfix(loss=float(loss_total), rel_lr=self.scheduler._get_scale())
            for k in log_dict.keys():
                v = eval(k)
                if isinstance(v, torch.Tensor):
                    log_dict[k] += v.item() / n_train
                else:
                    log_dict[k] += v / n_train

        log_dict["lr_scaler"] = self.scheduler._get_scale()
        self.train_log(log_dict, epoch + 1)
        return
