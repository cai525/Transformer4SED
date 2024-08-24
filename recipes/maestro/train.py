import logging
import random
from collections import OrderedDict, namedtuple

import numpy as np
import pandas as pd
import torch
import torchmetrics
import sed_scores_eval
from tqdm import tqdm

from archive.sound_event_class.classes_dict_mastro import classes_labels_maestro_real_eval, classes_labels_maestro_real
from src.codec.encoder import Encoder
from src.codec.decoder import batched_decode_preds, merge_maestro_ground_truth, get_segment_scores_and_overlap_add
from src.models.sed_model import SEDModel
from src.preprocess.data_aug import mixup, frame_shift, feature_transformation
from src.utils.log import Logger
from src.utils.statistics.mean_statistic import ProbMeanValue

logging.getLogger('matplotlib.font_manager').disabled = True

ScoreBufferTuple = namedtuple("ScoreBufferTuple", ["raw_student", "raw_teacher", "post_student", "post_teacher"])


class Trainer:

    def __init__(self, optimizer, my_logger: Logger, net, scheduler, encoder: Encoder, train_loader, val_loader,
                 test_loader, config, device):
        self.optimizer = optimizer
        self.my_logger = my_logger
        self.net = net
        if isinstance(net, torch.nn.parallel.DataParallel):
            assert isinstance(net.module, SEDModel)
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
    def preprocess(self, wav, label):
        # Transform to mel respectively
        mel = self.net.module.get_feature_extractor()(wav)

        # # time shift
        # mel, label = frame_shift(mel, label, net_pooling=mel.shape[-1]/label.shape[-1])
        # mixup (frequence)
        if random.random() < 0.5:
            mel, label = mixup(mel, label, c=max(np.random.beta(10, 0.5), 0.9))
        # Do label-independent augmentation
        mel = feature_transformation(mel, n_transform=1, **self.config["training"]["transform"])
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

    def frame_pred_pooling(self, strong, pooling_size=10):
        pool_f = lambda x: torch.nn.functional.avg_pool1d(x, kernel_size=pooling_size, stride=pooling_size)
        strong = pool_f(strong * strong) / pool_f(strong)
        return strong

    def train(self, epoch):
        # set  train mode
        self.net.train()
        n_train = len(self.train_loader)
        # set batch size
        log_dict = OrderedDict([("loss_total", 0), ("loss_class_strong", 0)])

        tk = tqdm(self.train_loader, total=n_train, leave=False, desc="training processing")

        for batch in tk:
            wav, labels, _, _ = batch
            wav, labels = wav.to(self.device), labels.to(self.device)
            mel, labels = self.preprocess(wav, labels)
            self.optimizer.zero_grad()
            # ==================== forward process ==================
            strong, weak, other_dict = self.net(mel, **self.config["PaSST_CNN"]["train_kwargs"])
            strong = self.frame_pred_pooling(
                strong,
                pooling_size=self.config["PaSST_CNN"]["init_kwargs"]["passt_sed_param"]["decoder_pos_emd_len"] //
                self.config["feature"]["pred_len"],
            )
            pred = self.Pred(strong, weak, other_dict)

            # ==================== calculate loss ====================
            # clip-level prediction from audio tagging branch
            # at_out = pred.other_dict['at_out_specific']

            # classifier loss for audio tagging branch
            # loss_class_at_specific = self.supervised_loss(at_out, labels_weak)

            # supervised_loss for SED branch
            loss_class_strong = self.supervised_loss(pred.strong, labels)

            # at_branch_loss = loss_class_at_specific * self.config["training"]["w_AT"]
            # total loss
            loss_total = loss_class_strong

            if torch.isnan(loss_total).any():
                raise Exception("Get loss value: None")

            # optimizer step
            if self.config["training"]["clip_grad"]:
                torch.nn.utils.clip_grad_norm(self.net.parameters(), max_norm=20, norm_type=2)
            loss_total.backward()
            self.optimizer.step()
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
    def preprocess_eval(self, wav):
        mel = self.net.module.get_feature_extractor().forward(wav)
        return mel

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
        get_weak_f1_seg_macro = torchmetrics.classification.f_beta.MultilabelF1Score(
            len(self.encoder.labels),
            average="macro",
            compute_on_step=False,
        ).to(self.device)

        # buffer for event based scores which we compute using sed-eval
        val_scores_buffer = {}
        n_valid = len(self.val_loader)

        with torch.no_grad():
            tk = tqdm(self.val_loader, total=n_valid, leave=False, desc="validation processing")
            for batch in tk:
                wav, labels, pad_mask, idx, filename, path = batch
                wav, labels = wav.to(self.device), labels.to(self.device)
                mel = self.preprocess_eval(wav)
                # prediction for student
                strong_preds, weak_preds, other_dict = list(self.net(mel, **self.config[self.net.module.get_model_name()]["val_kwargs"]))
                strong_preds = self.frame_pred_pooling(
                    strong_preds,
                    pooling_size=self.config["PaSST_CNN"]["init_kwargs"]["passt_sed_param"]["decoder_pos_emd_len"] //
                    self.config["feature"]["pred_len"],
                )

                # accumulate f1 score for weak labels
                labels_weak = (torch.sum(labels, -1) >= 1).float()
                get_weak_f1_seg_macro(weak_preds, labels_weak.long().to(self.device))

                (
                    scores_raw_strong,
                    _,
                ) = batched_decode_preds(
                    strong_preds=strong_preds,
                    filenames=path,
                    encoder=self.encoder,
                    filter=None,
                )

                val_scores_buffer.update(scores_raw_strong)

        # ================= calculate Segment based metrics =================
        maestro_audio_durations = sed_scores_eval.io.read_audio_durations(self.config["dataset"]["val_dur"])
        maestro_ground_truth_clips = pd.read_csv(self.config["dataset"]["val_tsv"], sep="\t")
        maestro_ground_truth_clips = maestro_ground_truth_clips[maestro_ground_truth_clips.confidence > .5]
        maestro_ground_truth_clips = maestro_ground_truth_clips[maestro_ground_truth_clips.event_label.isin(
            classes_labels_maestro_real_eval)]
        maestro_ground_truth_clips = sed_scores_eval.io.read_ground_truth_events(maestro_ground_truth_clips)
        # the step is equal to restore the ground truth file to the original format of the mastro dataset
        # the resolution of ground truth is 1s
        maestro_ground_truth = merge_maestro_ground_truth(maestro_ground_truth_clips)
        maestro_audio_durations = {file_id: maestro_audio_durations[file_id] for file_id in maestro_ground_truth.keys()}
        maestro_scores = {clip_id: val_scores_buffer[clip_id] for clip_id in maestro_ground_truth_clips.keys()}
        segment_length = 1.
        event_classes_maestro = sorted(classes_labels_maestro_real)
        segment_scores = get_segment_scores_and_overlap_add(
            frame_scores=maestro_scores,
            audio_durations=maestro_audio_durations,
            event_classes=event_classes_maestro,
            segment_length=segment_length,
        )
        event_classes_maestro_eval = sorted(classes_labels_maestro_real_eval)
        keys = ['onset', 'offset'] + event_classes_maestro_eval
        segment_scores = {clip_id: scores_df[keys] for clip_id, scores_df in segment_scores.items()}
        segment_f1_macro_optthres = sed_scores_eval.segment_based.best_fscore(
            segment_scores,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=segment_length,
        )[0]['macro_average']
        segment_mauc = sed_scores_eval.segment_based.auroc(
            segment_scores,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=segment_length,
        )[0]['mean']
        segment_mpauc = sed_scores_eval.segment_based.auroc(
            segment_scores,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=segment_length,
            max_fpr=.1,
        )[0]['mean']

        # calculate F1 score for weak label
        weak_f1 = get_weak_f1_seg_macro.compute()

        obj_metric = segment_mpauc

        # logging
        log_dict = OrderedDict([("segment_mpauc", segment_mpauc), ("segment_mauc", segment_mauc),
                                ("segment_f1_macro_optthres", segment_f1_macro_optthres), ("weak f1", weak_f1)])

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
        score_buffer = dict()
        get_weak_f1_seg_macro = torchmetrics.classification.f_beta.MultilabelF1Score(
            len(self.encoder.labels),
            average="macro",
            compute_on_step=False,
        ).to(self.device)

        # For compute mean value
        mean_computer = ProbMeanValue(n_bins=1000)

        with torch.no_grad():
            tk = tqdm(self.test_loader, total=n_test, leave=False, desc="test processing")
            for batch in tk:
                wav, labels, pad_mask, idx, filename, path = batch
                wav, labels = wav.to(self.device), labels.to(self.device)
                mel = self.preprocess_eval(wav)
                # prediction for student
                pred = list(self.net(mel, **self.config[self.net.module.get_model_name()]["val_kwargs"]))
                pred[0] = torch.nn.functional.avg_pool1d(pred[0], kernel_size=10, stride=10)
                pred = self.Pred(*pred)
                weak_preds = pred.weak
                # accumulate f1 score for weak labels
                labels_weak = (torch.sum(labels, -1) >= 1).float()
                get_weak_f1_seg_macro(weak_preds, labels_weak.long().to(self.device))

                # =========== update mean buffer =================
                labels_weak = (torch.sum(labels, -1) > 0).int()
                mean_computer.update_buffer(preds=pred.strong.detach().cpu().numpy(),
                                            labels=labels.detach().cpu().numpy(),
                                            weak_preds=weak_preds.detach().cpu().numpy(),
                                            labels_weak=labels_weak.detach().cpu().numpy())
                # =========== update psds score buffer ===========
                scores, _ = batched_decode_preds(strong_preds=pred.strong,
                                                 filenames=path,
                                                 encoder=self.encoder,
                                                 filter=None)
                score_buffer.update(scores)

        # calculate mean statistics
        mean_container = mean_computer.compute_mean_prob()
        self.my_logger.logger.info("weak_pos_mean : {0:0.3f}".format(np.mean(mean_container.weak_pos_mean)))
        self.my_logger.logger.info("weak_neg_mean : {0:0.3f}".format(np.mean(mean_container.weak_neg_mean)))
        self.my_logger.logger.info("strong_pos_mean : {0:0.3f}".format(np.mean(mean_container.strong_pos_mean)))
        self.my_logger.logger.info("strong_neg_mean : {0:0.3f}".format(np.mean(mean_container.strong_neg_mean)))

        # calculate psds
        # =============== calculate psds =============================
        test_tsv = self.config["dataset"]["pubeval_tsv"]
        test_dur = self.config["dataset"]["pubeval_dur"]
        psds_folders = self.config["training"]["psds_folders"]

        # calculate F1 score for weak label
        weak_f1 = get_weak_f1_seg_macro.compute()

        # ================= calculate Segment based metrics =================
        maestro_audio_durations = sed_scores_eval.io.read_audio_durations(test_dur)
        maestro_ground_truth_clips = pd.read_csv(test_tsv, sep="\t")
        maestro_ground_truth_clips = maestro_ground_truth_clips[maestro_ground_truth_clips.confidence > .5]
        maestro_ground_truth_clips = maestro_ground_truth_clips[maestro_ground_truth_clips.event_label.isin(
            classes_labels_maestro_real_eval)]
        maestro_ground_truth_clips = sed_scores_eval.io.read_ground_truth_events(maestro_ground_truth_clips)
        # the step is equal to restore the ground truth file to the original format of the mastro dataset
        # the resolution of ground truth is 1s
        maestro_ground_truth = merge_maestro_ground_truth(maestro_ground_truth_clips)
        maestro_audio_durations = {file_id: maestro_audio_durations[file_id] for file_id in maestro_ground_truth.keys()}
        maestro_scores = {clip_id: score_buffer[clip_id] for clip_id in maestro_ground_truth_clips.keys()}
        segment_length = 1.
        event_classes_maestro = sorted(classes_labels_maestro_real)
        segment_scores = get_segment_scores_and_overlap_add(
            frame_scores=maestro_scores,
            audio_durations=maestro_audio_durations,
            event_classes=event_classes_maestro,
            segment_length=segment_length,
        )
        sed_scores_eval.io.write_sed_scores(segment_scores, psds_folders)
        event_classes_maestro_eval = sorted(classes_labels_maestro_real_eval)
        keys = ['onset', 'offset'] + event_classes_maestro_eval
        segment_scores = {clip_id: scores_df[keys] for clip_id, scores_df in segment_scores.items()}
        segment_f1_macro_optthres = sed_scores_eval.segment_based.best_fscore(
            segment_scores,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=segment_length,
        )[0]['macro_average']
        segment_mauc = sed_scores_eval.segment_based.auroc(
            segment_scores,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=segment_length,
        )[0]['mean']
        segment_mpauc = sed_scores_eval.segment_based.auroc(
            segment_scores,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=segment_length,
            max_fpr=.1,
        )[0]['mean']

        # logging
        log_dict = OrderedDict([("segment_mpauc", segment_mpauc), ("segment_mauc", segment_mauc),
                                ("segment_f1_macro_optthres", segment_f1_macro_optthres), ("weak f1", weak_f1)])

        self.test_log(log_dict)

        return
