import pickle
from collections import namedtuple

import numpy as np


class ProbMeanValue:

    def __init__(self, n_bins,class_num=10) -> None:
        self.n_bins = n_bins
        self.bin = np.linspace(0, 1, n_bins+1)
        self.class_num=class_num
        self.hist_strong_pos = {i: np.zeros(n_bins) for i in range(class_num)}
        self.hist_strong_neg = {i: np.zeros(n_bins) for i in range(class_num)}
        self.hist_weak_pos = {i: np.zeros(n_bins) for i in range(class_num)}
        self.hist_weak_neg = {i: np.zeros(n_bins) for i in range(class_num)}
        # buffer to store mean of probabilities
        self.weak_pos_mean_buffer = []
        self.weak_neg_mean_buffer = []
        self.strong_pos_mean_buffer = []
        self.strong_neg_mean_buffer = []

    def update_buffer(self, preds, labels, weak_preds, labels_weak):
        # ============== Compute mean propability =================
        weak_pos_p, weak_neg_p = self._compute_weak_mean_p(weak_preds, labels_weak)
        self.weak_pos_mean_buffer.append(weak_pos_p)
        self.weak_neg_mean_buffer.append(weak_neg_p)
        strong_pos_p, strong_neg_p = self._compute_strong_mean_p(preds, labels)
        self.strong_pos_mean_buffer.append(strong_pos_p)
        self.strong_neg_mean_buffer.append(strong_neg_p)

        # ============== accumulating outputs histogram ===========
        batch_strong_pos_hist, batch_strong_neg_hist = self._compute_hist(pred=preds, label=labels, bins=self.bin)
        for i in range(self.class_num):
            self.hist_strong_pos[i] += batch_strong_pos_hist[i]
            self.hist_strong_neg[i] += batch_strong_neg_hist[i]

        batch_weak_pos_hist, batch_weak_neg_hist = self._compute_hist(pred=weak_preds, label=labels_weak, bins=self.bin)
        for i in range(self.class_num):
            self.hist_weak_pos[i] += batch_weak_pos_hist[i]
            self.hist_weak_neg[i] += batch_weak_neg_hist[i]

    def compute_mean_prob(self):
        weak_pos_mean = np.nanmean(np.stack(self.weak_pos_mean_buffer, axis=0), axis=0)
        weak_neg_mean = np.nanmean(np.stack(self.weak_neg_mean_buffer, axis=0), axis=0)
        strong_pos_mean = np.nanmean(np.stack(self.strong_pos_mean_buffer, axis=0), axis=0)
        strong_neg_mean = np.nanmean(np.stack(self.strong_neg_mean_buffer, axis=0), axis=0)

        MeanProb = namedtuple("MeanProb", ("weak_pos_mean", "weak_neg_mean", "strong_pos_mean", "strong_neg_mean"))

        return MeanProb(weak_pos_mean, weak_neg_mean, strong_pos_mean, strong_neg_mean)
    
    def save_hist(self, path):
    # ================ compute hist ============================
        with open(path, "wb") as f:
            pickle.dump({"strong_pos": self.hist_strong_pos,
                        "strong_neg": self.hist_strong_neg,
                        "weak_pos": self.hist_weak_pos,
                        "weak_neg": self.hist_weak_neg},f)

    def _compute_hist(self, pred: np.ndarray, label: np.ndarray, bins):
        pos_hist = dict()
        neg_hist = dict()

        for class_id in range(pred.shape[1]):
            class_pred, class_label = pred[:, class_id, ...], label[:, class_id, ...]
            pos, neg = class_pred[class_label == 1], class_pred[class_label == 0]
            pos_hist[class_id], _ = np.histogram(pos, bins=bins)
            neg_hist[class_id], _ = np.histogram(neg, bins=bins)
        return pos_hist, neg_hist

    def _compute_strong_mean_p(self, pred: np.ndarray, label: np.ndarray):
        """ Compute mean probability of strong label
        Both the prediction and label's size is [bs, n_class, frames]
        Args:
            pred (np.ndarray): prediction
            label (np.ndarray): label

        Returns:
            A tuple of positive and negetive mean probability. The shape of mean probability is 
            [n_class]
        """

        pos_pred, neg_pred = pred * label, pred * (1 - label)
        pos_mean_p = np.sum(pos_pred, axis=(0, 2)) / np.sum(label, axis=(0, 2))
        neg_mean_p = np.sum(neg_pred, axis=(0, 2)) / np.sum(1 - label, axis=(0, 2))

        return pos_mean_p, neg_mean_p

    def _compute_weak_mean_p(self, pred: np.ndarray, label: np.ndarray) -> tuple:
        """ Compute mean probability of weak label
            Both the prediction and label's size is [bs, n_class]
        Args:
            pred (np.ndarray): prediction
            label (np.ndarray): label

        Returns:
            tuple: include positive and negetive mean probability with shape [n_class].
        """
        pos_pred, neg_pred = pred * label, pred * (1 - label)
        pos_mean_p = np.sum(pos_pred, axis=0) / np.sum(label, axis=0)
        neg_mean_p = np.sum(neg_pred, axis=0) / np.sum(1 - label, axis=0)

        return pos_mean_p, neg_mean_p
