import os
import json
import logging
import random
from collections import OrderedDict, namedtuple

import numpy as np
import pandas as pd
import torch
import torchmetrics
import torch.nn.functional as F
from tqdm import tqdm

from recipes.audioset_strong.base.passt_cnn.train import Trainer, pool_strong_labels
from src.codec.decoder import batched_decode_preds
from src.preprocess.data_aug import mixup, frame_shift, feature_transformation

logging.getLogger('matplotlib.font_manager').disabled = True


def multi_label_to_multi_class(multi_label_pred: torch.Tensor):
    """ Convert multi-label prediction to multi-class prediction """
    B, C = multi_label_pred.shape
    multi_class_pred = torch.zeros(B, C, C + 1, device=multi_label_pred.device)
    multi_class_pred[:, :, :-1] = torch.eye(C, device=multi_label_pred.device) * multi_label_pred.unsqueeze(-1)
    multi_class_pred[:, :, -1] = 1 - multi_label_pred
    return multi_class_pred


def multi_class_to_multi_label(multi_class_pred: torch.Tensor):
    """ Convert multi-class prediction back to multi-label prediction by extracting diagonal values """
    multi_label_pred = multi_class_pred[:, :, :-1].diagonal(dim1=1, dim2=2)
    return multi_label_pred


class MaskformerTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ce_loss = torch.nn.CrossEntropyLoss()

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

    def train(self, epoch):
        # set  train mode
        self.net.train()
        n_train = len(self.train_loader)
        log_dict = OrderedDict([
            ("loss_total", 0),
            ("loss_class_strong", 0),
            ("loss_class_at_specific", 0),
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
            # clip-level prediction from audio tagging branch
            # classifier loss for audio tagging branch
            if self.config[self.net.get_model_name()]["init_kwargs"]["at_param"]["out_type"] == "logit":
                at_out_logit = pred.other_dict['at_out']
                loss_class_at_specific = self.ce_loss(
                    input=at_out_logit.transpose(1, 2),
                    target=multi_label_to_multi_class(labels_weak).transpose(1, 2),
                )
            elif self.config[self.net.get_model_name()]["init_kwargs"]["at_param"]["out_type"] == "sigmoid":
                loss_class_at_specific = self.supervised_loss(
                    input=pred.other_dict['at_out'],
                    target=labels_weak,
                )
            else:
                raise RuntimeError("Unknown output type for classification branch")

            # supervised_loss for SED branch
            loss_class_strong = self.supervised_loss(pred.strong, labels)
            # total loss
            loss_total = loss_class_strong + loss_class_at_specific * self.config["training"]["w_AT"]

            # optimizer step
            if self.config["training"]["clip_grad"]:
                torch.nn.utils.clip_grad_norm(self.net.parameters(), max_norm=20, norm_type=2)
            loss_total.backward()
            self.optimizer.step()
            self.scheduler.step()

            # logging
            tk.set_postfix(loss=float(loss_total),
                           strong=float(loss_class_strong),
                           at=float(loss_class_at_specific),
                           rel_lr=self.scheduler._get_scale())
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
    def validation(self, epoch):
        self.net.eval()
        get_at_map = torchmetrics.classification.MultilabelAveragePrecision(
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
                if self.config[self.net.get_model_name()]["init_kwargs"]["at_param"]["out_type"] == "logit":
                    weak_preds = multi_class_to_multi_label(F.softmax(pred.other_dict['at_out'], dim=-1))
                else:
                    weak_preds = pred.other_dict['at_out']

                # accumulate f1 score for weak labels
                labels_weak = (torch.sum(labels, -1) >= 1).float()
                get_at_map(weak_preds, labels_weak.long().to(self.device))

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
        at_mAP = get_at_map.compute()
        obj_metric = psds_sed_scores
        partial_psds = self.mean_psds_per_type(psds_sed_scores_single)
        # logging
        log_dict = OrderedDict([
            ("psds", psds_sed_scores),
            ("psds (common)", partial_psds['common']),
            ("psds (rare)", partial_psds['rare']),
            ("at mAP", at_mAP),
        ])
        psds_sed_scores_single = {k: round(v, 4) for k, v in psds_sed_scores_single.items()}
        with open(os.path.join(self.config["generals"]["save_folder"], 'single_psds.json'), 'w') as f:
            json.dump(OrderedDict(sorted(psds_sed_scores_single.items(), key=lambda x: x[1])), f, indent=4)
        self.val_log(log_dict, epoch + 1)
        return obj_metric

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
                pred = self.net(
                    input=feat,
                    pad_mask=pad_mask,
                    **self.config[self.net.get_model_name()]["test_kwargs"],
                )
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
            ("psds/s", psds),
            ("psds (common)", partial_psds['common']),
            ("psds (rare)", partial_psds['rare']),
        ])
        # save single psds scores
        psds_single = {k: round(v, 4) for k, v in psds_single.items()}
        with open(os.path.join(self.config["generals"]["save_folder"], 'single_psds.json'), 'w') as f:
            json.dump(OrderedDict(sorted(psds_single.items(), key=lambda x: x[1])), f, indent=4)

        self.test_log(log_dict)
        return
