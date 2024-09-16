import argparse
import os.path
import logging
import sys

import pandas as pd

root = "ROOT-PATH"
os.chdir(root)
sys.path.append(root)

from src.evaluation_measures import compute_psds_from_scores


def compute_psds1(input, ground_truth, audio_durations, save_dir):
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


def compute_psds2(input, ground_truth, audio_durations, save_dir):
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


def load_psds_scores(dir):
    res = {}
    for tsv_name in os.listdir(dir):
        wav_name = tsv_name.replace(".tsv", "")
        res[wav_name] = pd.read_csv(os.path.join(dir, tsv_name), sep="\t")
    return res


def test_from_tsv_scores(tsv_scores_folder, test_tsv, test_dur):
    score_buffer = load_psds_scores(tsv_scores_folder)
    # def psds1( input, ground_truth, audio_durations):
    psds1, psds1_single = compute_psds1(score_buffer, test_tsv, test_dur, save_dir=None)
    psds2, psds2_single = compute_psds2(score_buffer, test_tsv, test_dur, save_dir=None)
    logging.info(psds1_single)
    logging.info("psds1:{0}; psds2:{1}".format(psds1, psds2))


if __name__ == "__main__":
    #set configurations
    print(" " * 40 + "<" * 10 + "Ensemble" + ">" * 10)
    logging.basicConfig(level=logging.INFO)
    ##############################                        TEST                        ##############################
    save_folder = "ROOT-PATH/exps/dcase2024/ensemble/val/res/0.7-0.3"
    test_tsv = "ROOT-PATH/meta/validation/validation.tsv"
    test_dur = "ROOT-PATH/meta/validation/validation_durations.tsv"
    test_from_tsv_scores(save_folder, test_tsv, test_dur)

    print("<" * 30 + "DONE!" + ">" * 30)
