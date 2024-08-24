import os
from collections import OrderedDict

import pandas as pd
import torch
import sed_scores_eval
from torchaudio.transforms import AmplitudeToDB
from tqdm import tqdm

from recipes.finetune.train import Trainer, ScoreBufferTuple
from src.codec.decoder import batched_decode_preds, decode_pred_batch_fast
from src.evaluation_measures import log_sedeval_metrics
from src.preprocess.scaler import TorchScaler


class ATST_CNN_Trainer(Trainer):

    def __init__(self, *argc, **kwarg):
        super().__init__(*argc, **kwarg)

    def train(self, epoch):
        raise Exception("Unimplemented methods")

    def validation(self, epoch):
        raise Exception("Unimplemented methods")

    def preprocess_eval(self, wav):
        scaler = TorchScaler(
            "instance",
            "minmax",
            [1, 2],
        )
        cnn_mel = self.net.module.cnn_mel_trans(wav)
        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        cnn_feat = scaler(amp_to_db(cnn_mel).clamp(min=-50, max=80))
        atst_feat = self.net.module.atst_mel_trans(wav)
        return cnn_feat, atst_feat

    def test(self):
        self.net.eval()
        self.ema_net.eval()
        n_test = len(self.test_loader)

        score_buffer = ScoreBufferTuple(dict(), dict(), dict(), dict())
        # For compute F1 score
        stud_test_f1_buffer = pd.DataFrame()
        tch_test_f1_buffer = pd.DataFrame()

        # filter type for post-processing
        filter_type = self.config["training"]["filter_type"] if "filter_type" in self.config["training"].keys(
        ) else "median"

        with torch.no_grad():
            tk = tqdm(self.test_loader, total=n_test, leave=False, desc="test processing")
            for batch in tk:
                wav, labels, pad_mask, idx, filename, path = batch
                wav, labels = wav.to(self.device), labels.to(self.device)
                cnn_feat, atst_feat = self.preprocess_eval(wav)
                # prediction for student
                stu_pred = self.net(cnn_feat, atst_feat, **self.config[self.net.module.get_model_name()]["val_kwargs"])
                stu_pred = self.Pred(stu_pred[0], stu_pred[1], dict())
                # prediction for teacher
                tch_pred = self.ema_net(cnn_feat, atst_feat,
                                        **self.config[self.net.module.get_model_name()]["val_kwargs"])
                tch_pred = self.Pred(tch_pred[0], tch_pred[1], dict())

                # =========== update psds score buffer ===========
                stud_raw_scores, stud_scores = batched_decode_preds(strong_preds=stu_pred.strong,
                                                                    filenames=path,
                                                                    encoder=self.encoder,
                                                                    filter=self.median_fiter,
                                                                    weak_preds=stu_pred.weak,
                                                                    need_weak_mask=False,
                                                                    filter_type=filter_type)
                tch_raw_scores, tch_scores = batched_decode_preds(strong_preds=tch_pred.strong,
                                                                  filenames=path,
                                                                  encoder=self.encoder,
                                                                  filter=self.median_fiter,
                                                                  weak_preds=tch_pred.weak,
                                                                  need_weak_mask=False,
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
            sed_scores_eval.io.write_sed_scores(score_buffer.raw_teacher,
                                                os.path.join(self.config["generals"]["save_folder"], "raw_teacher"))
            sed_scores_eval.io.write_sed_scores(score_buffer.raw_student,
                                                os.path.join(self.config["generals"]["save_folder"], "raw_student"))
            sed_scores_eval.io.write_sed_scores(score_buffer.post_teacher,
                                                os.path.join(self.config["generals"]["save_folder"], "post_teacher"))
            sed_scores_eval.io.write_sed_scores(score_buffer.post_student,
                                                os.path.join(self.config["generals"]["save_folder"], "post_student"))
            return

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

        self.test_log(log_dict)
        self.dict_log(stud_psds1_single, head="psds1 single class(student):")
        self.dict_log(tch_psds1_single, head="psds1 single class(teacher):")
        return
