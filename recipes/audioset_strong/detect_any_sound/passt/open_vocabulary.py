import os, json, logging
from collections import OrderedDict

import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from tqdm import tqdm

from recipes.audioset_strong.detect_any_sound.passt.train import DASMTrainer
from src.codec.decoder import batched_decode_preds

logging.getLogger('numba').setLevel(logging.WARNING)


class OV_DASM_Trainer(DASMTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_common_query(self):
        num_gpu = torch.cuda.device_count()
        query_list = [self.net.at_query] if isinstance(self.net.at_query, torch.Tensor) else self.net.at_query
        common_query = list()
        for q in query_list:
            q = q[self.common_type_mask, :]
            if num_gpu > 1:
                q = q.unsqueeze(0).expand(num_gpu, -1, -1)
            common_query.append(q.to(self.device))
        if len(common_query) == 1:
            common_query = common_query[0]
        return common_query

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
        common_query = self.get_common_query()
        for i, batch in enumerate(tk):
            wav, labels, _, _ = batch
            wav, labels = wav.to(self.device), labels.to(self.device)
            labels = labels[:, self.common_type_mask, :]
            self.optimizer.zero_grad()
            feat, labels, labels_weak = self.preprocess(wav, labels)
            # ==================== forward process ==================
            pred = self.net(
                feat,
                query=common_query,
                **self.config[self.net.get_model_name()]["train_kwargs"],
            )
            pred = self.Pred(*pred)
            if torch.isnan(pred.strong).any():
                raise Exception("Error: Out of range")
            # ==================== calculate loss ====================
            # clip-level prediction from audio tagging branch
            # classifier loss for audio tagging branch
            loss_class_at_specific = self.supervised_loss(
                input=pred.other_dict['at_out'],
                target=labels_weak,
            )

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
    def get_att_mask(self):
        """
        Generate attention mask for self-attention modules in the inference stage.
        """
        common_size = torch.count_nonzero(self.common_type_mask)
        rare_size = len(self.common_type_mask) - common_size
        att_mask = torch.ones(common_size + rare_size, common_size + rare_size, dtype=torch.bool)
        att_mask[:, :common_size] = False
        att_mask.fill_diagonal_(False)  # fill diagonal in place
        num_gpu = torch.cuda.device_count()
        if num_gpu > 1:
            # support DP mode for multiple GPU running
            att_mask = att_mask.unsqueeze(0).expand(num_gpu, -1, -1)
        return att_mask.to(self.device)

    def get_common_first_query(self):
        """
        Get queries for validation and test. The common classes are sorted before rare classes.
        """
        query_list = self.net.at_query
        num_gpu = torch.cuda.device_count()
        if isinstance(query_list, torch.Tensor):
            query_list = [query_list]
        common_first_query = nn.ParameterList().to(self.device)
        for q in query_list:
            q = torch.cat([
                q[self.common_type_mask, :],
                q[torch.logical_not(self.common_type_mask), :],
            ])
            if num_gpu > 1:
                q = q.unsqueeze(0).expand(num_gpu, -1, -1)
            common_first_query.append(q)
        if len(common_first_query) == 1:
            common_first_query = common_first_query[0]
        return common_first_query

    def reorder_pred(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Recover the order of classes in the prediction to the origin order.
        pred: torch Tensor with shape [batch, n_class, ...]
        """
        common_size = torch.count_nonzero(self.common_type_mask)
        ret = torch.zeros_like(pred)
        ret[:, self.common_type_mask, ...] = pred[:, :common_size, ...]  # recover common classes
        ret[:, torch.logical_not(self.common_type_mask), ...] = pred[:, common_size:, ...]  # recover rare classes
        return ret

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
        att_mask = self.get_att_mask()
        common_first_query = self.get_common_first_query()

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
                    query=common_first_query,
                    tgt_mask=att_mask,
                    **self.config[self.net.get_model_name()]["val_kwargs"],
                )
                pred = self.Pred(*pred)
                # reorder classes
                pred.strong = self.reorder_pred(pred.strong)
                pred.weak = self.reorder_pred(pred.weak)
                pred.other_dict['at_out'] = self.reorder_pred(pred.other_dict['at_out'])

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
        filter_type = self.config["training"].get("filter_type", "median")
        att_mask = self.get_att_mask()
        common_first_query = self.get_common_first_query()

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
                    query=common_first_query,
                    tgt_mask=att_mask,
                    **self.config[self.net.get_model_name()]["test_kwargs"],
                )
                pred = self.Pred(*pred)
                # reorder classes
                pred.strong = self.reorder_pred(pred.strong)
                pred.weak = self.reorder_pred(pred.weak)
                pred.other_dict['at_out'] = self.reorder_pred(pred.other_dict['at_out'])

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
