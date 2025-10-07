import json
import logging
import random
import os
from collections import OrderedDict
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from tqdm import tqdm

from src.codec.encoder import Encoder
from src.codec.decoder import batched_decode_preds
from src.functional.loss import loss_function_factory
from src.models.sed_model import SEDModel
from src.preprocess.data_aug import mixup, frame_shift, feature_transformation
from src.evaluation_measures import compute_psds_from_scores
from src.utils import DataParallelWrapper
from src.utils.log import Logger

logging.getLogger('matplotlib.font_manager').disabled = True


def pool_strong_labels(x: torch.Tensor):
    x = torch.clamp(x, 1e-5, 1.)
    x = torch.clamp((x * x).sum(dim=-1) / x.sum(dim=-1), 1e-7, 1.)
    return x


class Trainer():

    def __init__(self, optimizer, my_logger: Logger, net: Union[SEDModel, nn.DataParallel], scheduler, encoder: Encoder,
                 train_loader, val_loader, test_loader, config, device):
        self.optimizer = optimizer
        self.my_logger = my_logger
        self.net = DataParallelWrapper(net) if isinstance(net, nn.DataParallel) else net
        self.config = config
        self.scheduler = scheduler
        self.encoder = encoder
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # loss function
        self.supervised_loss = loss_function_factory(
            name=config['class_loss']['loss_name'],
            kwargs=config['class_loss']['kwargs'],
        ).to(device)

    # inner class
    class Pred:

        def __init__(self, strong, weak, other_dict) -> None:
            self.strong = strong
            self.weak = weak
            self.other_dict = other_dict

    ################ tool functions for training process ###################
    def preprocess(self, wav, label):
        extractor = self.net.get_feature_extractor()
        mel = extractor(wav)
        # normalization
        mel = extractor.normalize(mel)

        # # time shift
        mel, label = frame_shift(
            mel,
            label,
            net_pooling=mel.shape[-1] / label.shape[-1],
            max_shift_frame=2 * self.encoder.sr,
        )
        # mixup (frequence)
        if random.random() < 0.5:
            mel, label = mixup(mel, label, c=np.random.beta(10, 0.5))
        # Do label-independent augmentation
        mel = feature_transformation(mel, log=True, norm_std=5.0, **self.config["training"]["transform"])

        # weak labels
        label_weak = pool_strong_labels(label)
        return mel, label, label_weak

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

    @property
    def train_epoch_len(self):
        if not hasattr(self, "_train_epoch_len"):
            _train_epoch_len = len(self.train_loader)
        return _train_epoch_len

    def train(self, epoch):
        # set  train mode
        self.net.train()
        n_train = len(self.train_loader)
        log_dict = OrderedDict([
            ("loss_class_strong", 0),
        ])

        tk = tqdm(self.train_loader, total=n_train, leave=False, desc="training processing")

        for i, batch in enumerate(tk):
            wav, labels, _, _ = batch
            wav, labels = wav.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            feat, labels, labels_weak = self.preprocess(wav, labels)
            # ==================== forward process ==================
            pred = self.net(feat, **self.config[self.net.get_model_name()]["train_kwargs"])
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

    ############### tool function for validation process ##################
    def preprocess_eval(self, wav):
        extractor = self.net.get_feature_extractor()
        mel = extractor(wav)
        return extractor.normalize(mel)

    @property
    def median_fiter(self) -> int:
        if not hasattr(self, "_median_filter"):
            pred_len = self.config["feature"]["pred_len"]
            self._median_filter = int(self.config["training"]["median_window"] / 156 * pred_len)
        return self._median_filter

    @property
    def val_events_set(self):
        if not hasattr(self, "_val_events"):
            val_df = pd.read_csv(self.config["dataset"]["val_tsv"], sep='\t')
            self._val_events_set = set(val_df["event_label"])
        return self._val_events_set

    def _remove_extra_events(self, scores_buffer: dict, events):
        for audio_id, df in scores_buffer.items():
            scores_buffer[audio_id] = df.drop(events, axis=1)
        return scores_buffer

    def psds(self, buffer, ground_truth, audio_durations, save_dir, events_set: set):
        buffer = self._remove_extra_events(buffer, set(self.encoder.labels) - events_set)
        return compute_psds_from_scores(
            buffer,
            ground_truth,
            audio_durations,
            save_dir=save_dir,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=0,
        )

    def dict_log(self, d: dict, head=None):
        self.my_logger.logger.info("-" * 30)
        self.my_logger.logger.info(head)
        self.my_logger.logger.info("-" * 30)
        for key, value in d.items():
            self.my_logger.logger.info("{key}: {value:.5f}".format(key=key, value=value))
        self.my_logger.logger.info("-" * 30)

    def val_log(self, log_dict: OrderedDict, epoch):
        for key, value in log_dict.items():
            self.my_logger.tensorboard_writer.add_scalar("validation/{key}".format(key=key), value, global_step=epoch)
        message = ["=============== validation  epoch{0} =================".format(epoch)]
        for i, k in enumerate(log_dict):
            if not i % 3:
                message.append("\n")
            message.append("   {k} = {v:.5f};".format(k=k, v=log_dict[k]))
        message.append("\n")
        self.my_logger.logger.critical("".join(message))

    @property
    def common_type_mask(self) -> torch.Tensor:
        if not hasattr(self, '_common_type_mask'):
            self._common_type_mask = torch.zeros(len(self.encoder.labels), dtype=torch.bool).to(self.device)
            for i, label in enumerate(self.encoder.labels):
                if self.type_dict[label] == 'common':
                    self._common_type_mask[i] = True
        return self._common_type_mask

    @property
    def type_dict(self):
        if not hasattr(self, '_type_dict'):
            with open(self.config['dataset']['event_state']) as f:
                self._type_dict = json.load(f)
        return self._type_dict

    def mean_psds_per_type(self, single_psds_dict: Dict[str, float]):
        """
        Calculate the mean psds according to single_psds_dict and type_dict.
        Exampe:
            single_psds_dict = {a1: psds_a1, a2: psds_a2, b1: psds_b1}
            type_dict = {a1: a, a2: a, b1: b}
            return: {a: (psds_a1 + psds_a2)/2, b: psds_b1}
        """
        type_set = set(self.type_dict.values())
        ret_dict = {category: [] for category in type_set}
        for event, psds in single_psds_dict.items():
            ret_dict[self.type_dict[event]].append(psds)
        for category, psds_list in ret_dict.items():
            ret_dict[category] = sum(psds_list) / len(psds_list)
        return ret_dict

    def validation(self, epoch):
        self.net.eval()
        get_mAP = torchmetrics.classification.MultilabelAveragePrecision(
            len(self.encoder.labels),
            average="macro",
            compute_on_step=False,
        ).to(self.device)

        # buffer for event based scores which we compute using sed-eval
        val_scores_postprocessed_buffer = {}
        n_valid = len(self.val_loader)

        with torch.no_grad():
            tk = tqdm(self.val_loader, total=n_valid, leave=False, desc="validation processing")
            for batch in tk:
                wav, labels, pad_mask, idx, filename, file_path = batch
                wav, labels = wav.to(self.device), labels.to(self.device)
                feat = self.preprocess_eval(wav)
                # prediction for student
                pred = self.net(
                    input=feat,
                    pad_mask=pad_mask,
                    **self.config[self.net.get_model_name()]["val_kwargs"],
                )
                pred = self.Pred(*pred)

                # accumulate f1 score for weak labels
                labels_weak = (torch.sum(labels, -1) >= 1).float()
                get_mAP(pred.weak, labels_weak.to(self.device))

                # psds

                (
                    scores_raw_strong,
                    scores_postprocessed_strong,
                ) = batched_decode_preds(
                    strong_preds=pred.strong,
                    filenames=file_path,
                    encoder=self.encoder,
                    weak_preds=None,
                    need_weak_mask=False,
                    filter=self.median_fiter,
                )

                val_scores_postprocessed_buffer.update(scores_postprocessed_strong)

        # calculate PSDS
        val_tsv = self.config["dataset"]["val_tsv"]
        val_dur = self.config["dataset"]["val_dur"]
        # psds_folders = self.config["training"]["psds_folders"]

        psds_sed_scores, psds_sed_scores_single = self.psds(
            val_scores_postprocessed_buffer,
            val_tsv,
            val_dur,
            save_dir=None,
            events_set=self.val_events_set,
        )

        # calculate F1 score for weak label
        mAP = get_mAP.compute()
        obj_metric = psds_sed_scores

        # logging
        partial_psds = self.mean_psds_per_type(psds_sed_scores_single)
        log_dict = OrderedDict([
            ("psds", psds_sed_scores),
            ("mAP", mAP),
            ("psds (common)", partial_psds['common']),
            ("psds (rare)", partial_psds['rare']),
        ])

        with open(os.path.join(self.config["generals"]["save_folder"], 'single_psds.json'), 'w') as f:
            json.dump(OrderedDict(sorted(psds_sed_scores_single.items(), key=lambda x: x[1])), f, indent=4)
        self.val_log(log_dict, epoch + 1)
        return obj_metric

    def test_log(self, log_dict: OrderedDict):
        message = ["================== test ===================="]
        for i, k in enumerate(log_dict):
            if not i % 3:
                message.append("\n")
            message.append("   {k} = {v:.5f};".format(k=k, v=log_dict[k]))
        message.append("\n")
        self.my_logger.logger.critical("".join(message))

    def test(self):
        self.net.eval()
        n_test = len(self.test_loader)

        # score buffer
        score_buffer = dict()
        # filter type for post-processing
        filter_type = self.config["training"]["filter_type"] if "filter_type" in self.config["training"].keys(
        ) else "median"

        with torch.no_grad():
            tk = tqdm(self.test_loader, total=n_test, leave=False, desc="test processing")
            for batch in tk:
                wav, labels, pad_mask, idx, filename, path = batch
                wav, labels = wav.to(self.device), labels.to(self.device)
                labels_weak = (torch.sum(labels, -1) >= 1).float()
                feat = self.preprocess_eval(wav)
                # prediction for student
                pred = self.net(input=feat, pad_mask=pad_mask, **self.config[self.net.get_model_name()]["test_kwargs"])
                pred = self.Pred(*pred)

                # =========== update psds score buffer ===========
                raw_scores, scores = batched_decode_preds(
                    strong_preds=pred.strong,
                    filenames=path,
                    encoder=self.encoder,
                    filter=self.median_fiter,
                    weak_preds=None,
                    need_weak_mask=False,
                    filter_type=filter_type,
                )

                score_buffer.update(scores)

        # calculate psds
        # =============== calculate psds =============================
        if self.config["generals"]["test_on_public_eval"]:
            test_tsv = self.config["dataset"]["pubeval_tsv"]
            test_dur = self.config["dataset"]["pubeval_dur"]
        else:
            test_tsv = self.config["dataset"]["test_tsv"]
            test_dur = self.config["dataset"]["test_dur"]
        psds_folders = self.config["training"]["psds_folder"]
        test_df = pd.read_csv(self.config["dataset"]["test_tsv"], sep='\t')
        test_events_set = set(test_df["event_label"])
        psds, psds_single = self.psds(
            score_buffer,
            test_tsv,
            test_dur,
            save_dir=psds_folders,
            events_set=test_events_set,
        )
        partial_psds = self.mean_psds_per_type(psds_single)
        # logging
        log_dict = OrderedDict([
            ("psds", psds),
            ("psds (common)", partial_psds['common']),
            ("psds (rare)", partial_psds['rare']),
        ])
        with open(os.path.join(self.config["generals"]["save_folder"], 'single_psds.json'), 'w') as f:
            json.dump(OrderedDict(sorted(psds_single.items(), key=lambda x: x[1])), f, indent=4)
        self.test_log(log_dict)
        return
