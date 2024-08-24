import logging
import os
import random
import time
from collections import OrderedDict, namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import sed_scores_eval
import numpy as np
import pandas as pd
import torch
import torchmetrics
from tqdm import tqdm

from src.codec.encoder import Encoder
from src.codec.decoder import batched_decode_preds, decode_pred_batch_fast
from src.models.sed_model import SEDModel
from src.evaluation_measures import (compute_psds_from_scores, log_sedeval_metrics)
from src.preprocess.data_aug import mixup, frame_shift, feature_transformation
from src.utils.scheduler import update_ema
from src.utils.log import Logger
from src.utils.statistics.mean_statistic import ProbMeanValue

logging.getLogger('matplotlib.font_manager').disabled = True

ScoreBufferTuple = namedtuple("ScoreBufferTuple", ["raw_student", "raw_teacher", "post_student", "post_teacher"])


class Trainer:

    def __init__(self, optimizer, my_logger: Logger, net, ema_net, scheduler, encoder: Encoder, train_loader,
                 val_loader, test_loader, config, device):
        self.optimizer = optimizer
        self.my_logger = my_logger
        self.net = net
        self.ema_net = ema_net
        if isinstance(net, torch.nn.parallel.DataParallel):
            assert isinstance(net.module, SEDModel)
        if isinstance(ema_net, torch.nn.parallel.DataParallel):
            assert isinstance(ema_net.module, SEDModel)
        self.config = config
        self.scheduler = scheduler
        self.encoder = encoder
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # loss function
        self.supervised_loss = torch.nn.BCELoss().to(device)
        self.selfsup_loss = torch.nn.MSELoss().to(device)

    # inner class
    Pred = namedtuple("Pred", ["strong", "weak", "other_dict"])

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
        mel = self.net.module.get_feature_extractor()(wav)

        # time shift
        mel, label = frame_shift(mel, label, net_pooling=self.encoder.net_pooling)
        label_weak = (torch.sum(label, -1) >= 1).float()
        # mixup (frequence)
        if random.random() < 0.5:
            # 对无标签样本的mixup破坏了原始的分布，和mean-teacher的核心思想不一致
            # 所以不对无标签样本加mixup
            for m in [strong_mask, weak_mask]:
                mel[m], label[m] = mixup(mel[m], label[m], c=np.random.beta(10, 0.5))
        # Do label-independent augmentation
        stu_mel, tch_mel = feature_transformation(mel, **self.config["training"]["transform"])
        label_weak[weak_mask] = torch.sum(label[weak_mask], -1)
        return tch_mel, stu_mel, label, label_weak

    @property
    def train_epoch_len(self):
        if not hasattr(self, "_train_epoch_len"):
            _train_epoch_len = len(self.train_loader)
        return _train_epoch_len

    def get_self_weight(self):

        def sigmoid(x, k):
            return 1 / (1 + np.exp(-k * x))

        epoch_len = self.train_epoch_len
        # phase1 : weight of teacher becomes bigger
        if self.scheduler.step_num < self.config["training"]['self_loss_warmup'] * epoch_len:
            warmup_value = (self.scheduler.step_num) / (self.config["training"]['self_loss_warmup'] * epoch_len)
            if self.config["training"]["cons_scheduler_name"] == "Sigmoid":
                f = lambda x: sigmoid(x - 0.5, 10)
                warmup_value = f(warmup_value)
            elif self.config["training"]["cons_scheduler_name"] == "Linear":
                pass
            else:
                raise RuntimeError("Unknown cons_scheduler_name")
        # phase2 : weight of teacher becomes biggest
        else:
            warmup_value = 1
        return warmup_value

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

    def train(self, epoch):
        # set  train mode
        self.net.train()
        self.ema_net.train()
        n_train = len(self.train_loader)
        # set batch size
        strong_num, syn_num, weak_num, unlabel_num = self.config["training"]["batch_size"]
        strong_num += syn_num
        log_dict = OrderedDict([("loss_total", 0), ("loss_class_strong", 0), ("loss_class_weak", 0),
                                ("loss_class_at_specific", 0), ("loss_cons_strong", 0), ("loss_cons_weak", 0),
                                ("loss_cons_at_specific", 0), ("w_cons", 0)])

        tk = tqdm(self.train_loader, total=n_train, leave=False, desc="training processing")

        for i, batch in enumerate(tk):
            wav, labels, _, _ = batch
            wav, labels = wav.to(self.device), labels.to(self.device)
            mask_strong, mask_weak, mask_unlabel = self.get_mask(wav, strong_num, weak_num, unlabel_num)
            self.optimizer.zero_grad()
            tch_mel, stu_mel, labels, labels_weak = self.preprocess(wav, labels, mask_strong, mask_weak, mask_unlabel)
            # ==================== forward process ==================
            stu_pred = self.net(stu_mel, **self.config[self.net.module.get_model_name()]["train_stu_kwargs"])
            stu_pred = self.Pred(*stu_pred)
            if torch.isnan(stu_pred.strong).any():
                raise Exception("Error: Out of range")
            with torch.no_grad():
                tch_pred = self.ema_net(tch_mel, **self.config[self.net.module.get_model_name()]["train_tch_kwargs"])
                tch_pred = self.Pred(*tch_pred)
                if torch.isnan(tch_pred.strong).any():
                    raise Exception("Error: Out of range")

            # ==================== calculate loss ====================
            # clip-level prediction from audio tagging branch
            at_out_specific_stud = stu_pred.other_dict['at_out_specific']
            at_out_specific_teacher = tch_pred.other_dict['at_out_specific']

            # classifier loss for audio tagging branch
            loss_class_at_specific = self.supervised_loss(at_out_specific_stud[mask_weak],
                                                          labels_weak[mask_weak])

            loss_cons_at_specific = self.selfsup_loss(at_out_specific_stud, at_out_specific_teacher.detach())

            # supervised_loss for SED branch
            loss_class_strong = self.supervised_loss(stu_pred.strong[mask_strong], labels[mask_strong])

            loss_class_weak = self.supervised_loss(stu_pred.weak[mask_weak],
                                                   labels_weak[mask_weak])

            # consistent loss for SED branch
            loss_cons_strong = self.selfsup_loss(stu_pred.strong, tch_pred.strong.detach())

            loss_cons_weak = self.selfsup_loss(stu_pred.weak, at_out_specific_teacher.detach())

            warmup_value = self.get_self_weight()
            w_cons = max(self.config["training"]["w_cons_max"] * warmup_value, self.config["training"]["w_cons_min"])

            self_loss = (loss_cons_strong + self.config["training"]["w_weak_cons"]*loss_cons_weak\
                + self.config["training"]["w_AT"]*loss_cons_at_specific) * w_cons
            at_branch_loss = loss_class_at_specific * self.config["training"]["w_AT"]
            # total loss
            loss_total = loss_class_strong + self.config["training"]["w_weak"] * loss_class_weak\
                +  self_loss +  at_branch_loss

            if torch.isnan(loss_total).any():
                raise Exception("Get loss value: None")

            # optimizer step
            if self.config["training"]["clip_grad"]:
                torch.nn.utils.clip_grad_norm(self.net.parameters(), max_norm=20, norm_type=2)
            loss_total.backward()
            # if not i % self.config["training"].get("grad_accum", 1):
            self.optimizer.step()
            self.scheduler.step()
            self.ema_net = update_ema(self.net, self.ema_net, self.scheduler.step_num,
                                        self.config["training"]["ema_factor"])
            # logging
            for k in log_dict.keys():
                v = eval(k)
                if isinstance(v, torch.Tensor):
                    log_dict[k] += v.item() / n_train
                else:
                    log_dict[k] += v / n_train

        log_dict["lr_scaler"] = self.scheduler._get_scale()
        log_dict["w_cons"] = w_cons
        self.train_log(log_dict, epoch + 1)
        return

    ############### tool function for validation process ##################
    def preprocess_eval(self, wav):
        mel = self.net.module.get_feature_extractor()(wav)
        return mel

    @property
    def median_fiter(self):
        if not hasattr(self, "_median_filter"):
            pred_len = self.config["feature"]["pred_len"]
            self._median_filter = [int(i / 156 * pred_len) for i in self.config["training"]["median_window"]]
            self.my_logger.logger.info("median filter:{0}".format(self._median_filter))
        return self._median_filter

    def psds1(self, input, ground_truth, audio_durations, save_dir):
        return compute_psds_from_scores(
            input,
            ground_truth,
            audio_durations,
            save_dir=save_dir,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
        )

    def psds2(self, input, ground_truth, audio_durations, save_dir):
        return compute_psds_from_scores(
            input,
            ground_truth,
            audio_durations,
            save_dir=save_dir,
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1,
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

    def validation(self, epoch):
        self.net.eval()
        self.ema_net.eval()
        get_weak_student_f1_seg_macro = torchmetrics.classification.f_beta.MultilabelF1Score(
            len(self.encoder.labels),
            average="macro",
            compute_on_step=False,
        ).to(self.device)

        get_weak_teacher_f1_seg_macro = torchmetrics.classification.f_beta.MultilabelF1Score(
            len(self.encoder.labels),
            average="macro",
            compute_on_step=False,
        ).to(self.device)

        val_thds = self.config["training"]["val_thresholds"]
        val_stud_buffer = pd.DataFrame()
        val_tch_buffer = pd.DataFrame()

        # buffer for event based scores which we compute using sed-eval
        val_scores_postprocessed_buffer_student_real = {}
        val_scores_postprocessed_buffer_teacher_real = {}
        n_valid = len(self.val_loader)

        with torch.no_grad():
            tk = tqdm(self.val_loader, total=n_valid, leave=False, desc="validation processing")
            for batch in tk:
                wav, labels, pad_mask, idx, filename, path = batch
                wav, labels = wav.to(self.device), labels.to(self.device)
                mel = self.preprocess_eval(wav)
                # prediction for student
                stu_pred = self.net(mel, pad_mask=pad_mask, **self.config[self.net.module.get_model_name()]["val_kwargs"])
                stu_pred = self.Pred(*stu_pred)
                strong_preds_student = stu_pred.strong
                weak_preds_student = stu_pred.other_dict['at_out_specific']

                # prediction for teacher
                tch_pred = self.ema_net(mel, pad_mask=pad_mask, **self.config[self.net.module.get_model_name()]["val_kwargs"])
                tch_pred = self.Pred(*tch_pred)
                strong_preds_teacher = tch_pred.strong
                weak_preds_teacher = tch_pred.other_dict['at_out_specific']

                mask_weak = (torch.tensor([
                    str(Path(x).parent) == str(Path(self.config["dataset"]["weak_folder"])) for x in path
                ]).to(self.device).bool())

                mask_real = (torch.tensor([
                    str(Path(x).parent) == str(Path(self.config["dataset"]["val_folder"])) for x in path
                ]).to(self.device).bool())

                if torch.any(mask_weak):
                    labels_weak = (torch.sum(labels[mask_weak], -1) >= 1).float()

                    # accumulate f1 score for weak labels
                    get_weak_student_f1_seg_macro(weak_preds_student[mask_weak], labels_weak.long().to(self.device))
                    get_weak_teacher_f1_seg_macro(weak_preds_teacher[mask_weak], labels_weak.long().to(self.device))

                if torch.any(mask_real):
                    # accumulate f1 score for weak labels
                    labels_weak = (torch.sum(labels[mask_real], -1) >= 1).float()
                    get_weak_student_f1_seg_macro(weak_preds_student[mask_real], labels_weak.long().to(self.device))
                    get_weak_teacher_f1_seg_macro(weak_preds_teacher[mask_real], labels_weak.long().to(self.device))

                    # psds
                    filenames_real = [x for x in path if Path(x).parent == Path(self.config["dataset"]["val_folder"])]

                    (
                        scores_raw_student_strong,
                        scores_postprocessed_student_strong,
                    ) = batched_decode_preds(
                        strong_preds=strong_preds_student[mask_real],
                        filenames=filenames_real,
                        encoder=self.encoder,
                        weak_preds=stu_pred.weak[mask_real],
                        need_weak_mask=True,
                        filter=self.median_fiter,
                    )

                    val_scores_postprocessed_buffer_student_real.update(scores_postprocessed_student_strong)

                    (scores_raw_teacher_strong, scores_postprocessed_teacher_strong) = batched_decode_preds(
                        strong_preds=strong_preds_teacher[mask_real],
                        filenames=filenames_real,
                        encoder=self.encoder,
                        weak_preds=tch_pred.weak[mask_real],
                        need_weak_mask=True,
                        filter=self.median_fiter,
                    )

                    val_scores_postprocessed_buffer_teacher_real.update(scores_postprocessed_teacher_strong)

                    # calculate event-based F1 score
                    stud_pred_dfs = decode_pred_batch_fast(strong_preds_student[mask_real], stu_pred.weak[mask_real],
                                                           filenames_real, self.encoder, [0.5], self.median_fiter,
                                                           self.config["training"]["decode_weak_valid"])

                    tch_pred_dfs = decode_pred_batch_fast(strong_preds_teacher[mask_real], tch_pred.weak[mask_real],
                                                          filenames_real, self.encoder, [0.5], self.median_fiter,
                                                          self.config["training"]["decode_weak_valid"])
                    val_stud_buffer = val_stud_buffer.append(stud_pred_dfs[0.5], ignore_index=True)
                    val_tch_buffer = val_tch_buffer.append(tch_pred_dfs[0.5], ignore_index=True)

        # calculate PSDS
        val_tsv = self.config["dataset"]["val_tsv"]
        val_dur = self.config["dataset"]["val_dur"]
        # psds_folders = self.config["training"]["psds_folders"]

        psds1_student_sed_scores_real, psds1_student_sed_scores_single = self.psds1(
            val_scores_postprocessed_buffer_student_real, val_tsv, val_dur, save_dir=None)
        psds2_student_sed_scores_real, psds2_student_sed_scores_single = self.psds2(
            val_scores_postprocessed_buffer_student_real, val_tsv, val_dur, save_dir=None)
        psds1_teacher_sed_scores_real, psds1_teacher_sed_scores_single = self.psds1(
            val_scores_postprocessed_buffer_teacher_real, val_tsv, val_dur, save_dir=None)
        psds2_teacher_sed_scores_real, psds2_teacher_sed_scores_single = self.psds2(
            val_scores_postprocessed_buffer_teacher_real, val_tsv, val_dur, save_dir=None)

        # # calculate event-based F1 score
        # stud_intersection_f1 = compute_per_intersection_macro_f1(val_stud_buffer,
        #                                                          self.config["dataset"]["val_tsv"],
        #                                                          self.config["dataset"]["val_dur"])
        # tch_intersection_f1 = compute_per_intersection_macro_f1(val_tch_buffer,
        #                                                         self.config["dataset"]["val_tsv"],
        #                                                         self.config["dataset"]["val_dur"])
        stud_event_macro_F1, stud_event_micro_F1, stud_seg_macro_F1, stud_seg_micro_F1 = log_sedeval_metrics(
            val_stud_buffer, val_tsv, None)

        tch_event_macro_F1, tch_event_micro_F1, tch_seg_macro_F1, tch_seg_micro_F1 = log_sedeval_metrics(
            val_tch_buffer, val_tsv, None)
        # calculate F1 score for weak label
        stu_weak_f1 = get_weak_student_f1_seg_macro.compute()
        tch_weak_f1 = get_weak_teacher_f1_seg_macro.compute()

        tch_obj_metric = psds1_teacher_sed_scores_real
        stu_obj_metric = psds1_student_sed_scores_real

        # logging
        log_dict = OrderedDict([("psds1/t", psds1_teacher_sed_scores_real), ("psds2/t", psds2_teacher_sed_scores_real),
                                ("psds1/s", psds1_student_sed_scores_real), ("psds2/s", psds2_student_sed_scores_real),
                                ("event-based F1/t", tch_event_macro_F1), ("event-based F1/s", stud_event_macro_F1),
                                ("weak f1/t", tch_weak_f1), ("weak f1/s", stu_weak_f1)])

        self.val_log(log_dict, epoch + 1)
        # self.dict_log(psds1_student_sed_scores_single, head="psds1 single class(student):")
        # self.dict_log(psds1_teacher_sed_scores_single, head="psds1 single class(teacher):")
        return stu_obj_metric, tch_obj_metric

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
        self.ema_net.eval()
        n_test = len(self.test_loader)
        # buffer to store psds score in each batch
        score_buffer = ScoreBufferTuple(dict(), dict(), dict(), dict())

        # For compute F1 score
        stud_test_f1_buffer = pd.DataFrame()
        tch_test_f1_buffer = pd.DataFrame()

        # For compute mean value
        mean_computer = ProbMeanValue(n_bins=1000)

        # hook setting
        hook_state = None

        # filter type for post-processing
        filter_type = self.config["training"]["filter_type"] if "filter_type" in self.config["training"].keys(
        ) else "median"


        with torch.no_grad():
            tk = tqdm(self.test_loader, total=n_test, leave=False, desc="test processing")
            for batch in tk:
                wav, labels, pad_mask, idx, filename, path = batch
                wav, labels = wav.to(self.device), labels.to(self.device)
                mel = self.preprocess_eval(wav)
                # prediction for student
                stu_pred = self.net(mel, pad_mask=pad_mask, **self.config[self.net.module.get_model_name()]["val_kwargs"])
                stu_pred = self.Pred(*stu_pred)
                weak_preds_student = stu_pred.other_dict['at_out_specific']
                # prediction for teacher
                tch_pred = self.ema_net(mel, pad_mask=pad_mask, **self.config[self.net.module.get_model_name()]["val_kwargs"])
                tch_pred = self.Pred(*tch_pred)
                weak_preds_teacher = tch_pred.other_dict['at_out_specific']

                # =========== update mean buffer =================
                labels_weak = (torch.sum(labels, -1) > 0).int()
                mean_computer.update_buffer(preds=tch_pred.strong.detach().cpu().numpy(),
                                            labels=labels.detach().cpu().numpy(),
                                            weak_preds=weak_preds_teacher.detach().cpu().numpy(),
                                            labels_weak=labels_weak.detach().cpu().numpy())
                # =========== update psds score buffer ===========
                stud_raw_scores, stud_scores = batched_decode_preds(strong_preds=stu_pred.strong,
                                                                    filenames=path,
                                                                    encoder=self.encoder,
                                                                    filter=self.median_fiter,
                                                                    weak_preds=stu_pred.weak,
                                                                    need_weak_mask=True,
                                                                    filter_type=filter_type)
                tch_raw_scores, tch_scores = batched_decode_preds(strong_preds=tch_pred.strong,
                                                                  filenames=path,
                                                                  encoder=self.encoder,
                                                                  filter=self.median_fiter,
                                                                  weak_preds=tch_pred.weak,
                                                                  need_weak_mask=True,
                                                                  filter_type=filter_type)

                score_buffer.raw_student.update(stud_raw_scores)
                score_buffer.raw_teacher.update(tch_raw_scores)
                score_buffer.post_student.update(stud_scores)
                score_buffer.post_teacher.update(tch_scores)

                # =========== calculate F1 score buffer ===========
                stud_pred_df_halfpoint = decode_pred_batch_fast(stu_pred.strong, stu_pred.weak, path, self.encoder,
                                                                [0.5], self.median_fiter,
                                                                self.config["training"]["decode_weak_test"])
                tch_pred_df_halfpoint = decode_pred_batch_fast(tch_pred.strong, tch_pred.weak, path, self.encoder,
                                                               [0.5], self.median_fiter,
                                                               self.config["training"]["decode_weak_test"])
                stud_test_f1_buffer = stud_test_f1_buffer.append(stud_pred_df_halfpoint[0.5], ignore_index=True)
                tch_test_f1_buffer = tch_test_f1_buffer.append(tch_pred_df_halfpoint[0.5], ignore_index=True)
        
        if self.config["generals"].get("predict", False):
            sed_scores_eval.io.write_sed_scores(score_buffer.raw_teacher, os.path.join(self.config["generals"]["save_folder"], "raw_teacher"))
            sed_scores_eval.io.write_sed_scores(score_buffer.raw_student, os.path.join(self.config["generals"]["save_folder"], "raw_student"))
            sed_scores_eval.io.write_sed_scores(score_buffer.post_teacher, os.path.join(self.config["generals"]["save_folder"], "post_teacher"))
            sed_scores_eval.io.write_sed_scores(score_buffer.post_student, os.path.join(self.config["generals"]["save_folder"], "post_student"))
            return 
        

        # calculate mean statistics
        mean_container = mean_computer.compute_mean_prob()
        self.my_logger.logger.info("weak_pos_mean : {0:0.3f}".format(np.mean(mean_container.weak_pos_mean)))
        self.my_logger.logger.info("weak_neg_mean : {0:0.3f}".format(np.mean(mean_container.weak_neg_mean)))
        self.my_logger.logger.info("strong_pos_mean : {0:0.3f}".format(np.mean(mean_container.strong_pos_mean)))
        self.my_logger.logger.info("strong_neg_mean : {0:0.3f}".format(np.mean(mean_container.strong_neg_mean)))
        # # save hist
        # mean_computer.save_hist(path=os.path.join(self.config["training"]["save_folder"], "hist"))
        # # save buffer
        # with open(os.path.join(self.config["training"]["save_folder"], "score_buffer"), "wb") as f:
        #     pickle.dump(score_buffer,f)

        # calculate psds
        # =============== calculate psds =============================
        if self.config["generals"]["test_on_public_eval"]:
            test_tsv = self.config["dataset"]["pubeval_tsv"]
            test_dur = self.config["dataset"]["pubeval_dur"]
        else:
            test_tsv = self.config["dataset"]["test_tsv"]
            test_dur = self.config["dataset"]["test_dur"]
        psds_folders = self.config["training"]["psds_folders"]
        # def psds1(self, input, ground_truth, audio_durations):
        stud_psds1, stud_psds1_single = self.psds1(score_buffer.post_student,
                                                   test_tsv,
                                                   test_dur,
                                                   save_dir=psds_folders[0])
        stud_psds2, stud_psds2_single = self.psds2(score_buffer.post_student,
                                                   test_tsv,
                                                   test_dur,
                                                   save_dir=psds_folders[0])
        tch_psds1, tch_psds1_single = self.psds1(score_buffer.post_teacher,
                                                 test_tsv,
                                                 test_dur,
                                                 save_dir=psds_folders[1])
        tch_psds2, tch_psds2_single = self.psds2(score_buffer.post_teacher,
                                                 test_tsv,
                                                 test_dur,
                                                 save_dir=psds_folders[1])

        stud_event_macro_F1, stud_event_micro_F1, stud_seg_macro_F1, stud_seg_micro_F1 = log_sedeval_metrics(
            stud_test_f1_buffer, test_tsv, psds_folders[0])

        tch_event_macro_F1, tch_event_micro_F1, tch_seg_macro_F1, tch_seg_micro_F1 = log_sedeval_metrics(
            tch_test_f1_buffer, test_tsv, psds_folders[1])

        # logging
        log_dict = OrderedDict([("psds1/t", tch_psds1), ("psds2/t", tch_psds2), ("psds1/s", stud_psds1),
                                ("psds2/s", stud_psds2), ("event-based f1/s", stud_event_macro_F1),
                                ("event_based f1/t", tch_event_macro_F1)])

        self.dict_log(stud_psds1_single, head="psds1 single class(student):")
        self.dict_log(tch_psds1_single, head="psds1 single class(teacher):")
        self.test_log(log_dict)
        return
