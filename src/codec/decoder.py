import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from src.postprocess.filter import MyMedianfilterfunc
from sed_scores_eval.base_modules.scores import create_score_dataframe, validate_score_dataframe



def decode_pred_batch_fast(outputs, weak_preds, filenames, encoder, thresholds, median_filter, decode_weak, pad_idx=None):
    pred_dfs = {}
    for threshold in thresholds:
        pred_dfs[threshold] = pd.DataFrame()

    for c_th in thresholds:
        output = copy.deepcopy(outputs.transpose(1, 2).detach()) #output size = [batch,frames, n_class]

        if decode_weak: # if decode_weak = 1 or 2
            weak_pred_negtive_index = torch.where(weak_preds<c_th)
            output[weak_pred_negtive_index[0],:,weak_pred_negtive_index[1]] = 0
            if decode_weak > 1: # use only weak predictions (weakSED)
                weak_pred_positive_index = torch.where(weak_preds > c_th)
                output[weak_pred_positive_index[0],:,weak_pred_positive_index] = 1
        if decode_weak < 2: # weak prediction masking
            output = MyMedianfilterfunc(output, median_filter)
            output = (output > c_th).float()
            #output.shape B,T,C
        output = output.cpu().numpy()
        for batch_idx in range(output.shape[0]):
            output_one = output[batch_idx]
            pred = encoder.decode_strong(output_one)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = Path(filenames[batch_idx]).stem + ".wav"
            pred_dfs[c_th] = pred_dfs[c_th].append(pred, ignore_index=True)

    return pred_dfs

def batched_decode_preds(strong_preds,
                         filenames,
                         encoder,
                         median_filter=7,
                         filter_type="median",
                         pad_indx=None,
                         weak_preds=None,
                         need_weak_mask=None
                         ):
    """ Decode a batch of predictions to dataframes. Each threshold gives a different dataframe and stored in a
    dictionary
    Args:
        strong_preds: torch.Tensor, batch of strong predictions.( #outputs size = [bs, n_class, frames])
        filenames: list, the list of filenames of the current batch.
        encoder: ManyHotEncoder object, object used to decode predictions.
        thresholds: list, the list of thresholds to be used for predictions.
        median_filter: int, the number of frames for which to apply median window (smoothing).
        filter_type: "median" or "max"
        pad_indx: list, the list of indexes which have been used for padding.
    Returns:
        dict of predictions, each keys is a threshold and the value is the DataFrame of predictions.
    """
    # Init a dataframe per threshold
    scores_raw = dict()
    scores_postprocessed = dict()

    for j in range(strong_preds.shape[0]):  # over batches
        audio_id = Path(filenames[j]).stem
        filename = audio_id + ".wav"
        c_scores = strong_preds[j]  # [n_class, frame]
        if pad_indx is not None:
            true_len = int(c_scores.shape[-1] * pad_indx[j].item())
            c_scores = c_scores[:true_len]
        c_scores = c_scores.transpose(0, 1).detach()   # # [frame, n_class]
        if need_weak_mask and (weak_preds != None):
            # >>>>>>>>>>>>>>>> Change mask method here <<<<<<<<<<<<<<<<<<<
            # hard mask
            # for class_idx in range(weak_preds.size(1)):
            #     #  mask using weak prediction
            #     # if weak_preds[j, class_idx] < weak_thresholds[class_idx]:
            #     if weak_preds[j, class_idx] < 0.5:
            #         c_scores[:, class_idx] = 0
            # soft mask
            c_scores = c_scores * weak_preds[j,:]
        c_scores = c_scores.cpu().numpy()
        scores_raw[audio_id] = create_score_dataframe(
            scores=c_scores,
            timestamps=encoder._frame_to_time(np.arange(len(c_scores) + 1)),
            event_classes=encoder.labels,
        )
            
        for idx in range(len(median_filter)):
            # median filter or max filter
            if filter_type == "median":
                c_scores[:, idx] = ndimage.filters.median_filter(c_scores[:, idx], (median_filter[idx]))
            elif filter_type == "max":
                c_scores[:, idx] = ndimage.filters.maximum_filter(c_scores[:, idx], (median_filter[idx]))
        scores_postprocessed[audio_id] = create_score_dataframe(
            scores=c_scores,
            timestamps=encoder._frame_to_time(np.arange(len(c_scores) + 1)),
            event_classes=encoder.labels,
        )

    return  scores_raw, scores_postprocessed