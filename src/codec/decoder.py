import copy
from collections import defaultdict
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from src.postprocess.filter import MyMedianfilterfunc
from sed_scores_eval.base_modules.scores import create_score_dataframe, validate_score_dataframe
""""
decode_pred_batch_fast vs batched_decode_preds
    decode_pred_batch_fast: 生成离散的，二值化的预测值；
    batched_decode_preds: 生成连续的预测；
"""


def decode_pred_batch_fast(outputs,
                           weak_preds,
                           filenames,
                           encoder,
                           thresholds,
                           median_filter,
                           decode_weak):
    pred_dfs = {}
    for threshold in thresholds:
        pred_dfs[threshold] = pd.DataFrame()

    for c_th in thresholds:
        output = copy.deepcopy(outputs.transpose(1, 2).detach())  #output size = [batch,frames, n_class]

        if decode_weak:  # if decode_weak = 1 or 2
            weak_pred_negtive_index = torch.where(weak_preds < c_th)
            output[weak_pred_negtive_index[0], :, weak_pred_negtive_index[1]] = 0
            if decode_weak > 1:  # use only weak predictions (weakSED)
                weak_pred_positive_index = torch.where(weak_preds > c_th)
                output[weak_pred_positive_index[0], :, weak_pred_positive_index] = 1
        if decode_weak < 2:  # weak prediction masking
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
                         filter=7,
                         filter_type="median",
                         pad_indx=None,
                         weak_preds=None,
                         need_weak_mask=None):
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
        c_scores = c_scores.transpose(0, 1).detach()  # # [frame, n_class]
        if need_weak_mask and (weak_preds != None):
            # >>>>>>>>>>>>>>>> Change mask method here <<<<<<<<<<<<<<<<<<<
            # hard mask
            # for class_idx in range(weak_preds.size(1)):
            #     #  mask using weak prediction
            #     # if weak_preds[j, class_idx] < weak_thresholds[class_idx]:
            #     if weak_preds[j, class_idx] < 0.5:
            #         c_scores[:, class_idx] = 0
            # soft mask
            c_scores = c_scores * weak_preds[j, :]
        c_scores = c_scores.cpu().numpy()
        scores_raw[audio_id] = create_score_dataframe(
            scores=c_scores,
            timestamps=encoder._frame_to_time(np.arange(len(c_scores) + 1)),
            event_classes=encoder.labels,
        )

        if filter:
            for idx in range(len(filter)):
                # median filter or max filter
                if filter_type == "median":
                    c_scores[:, idx] = ndimage.filters.median_filter(c_scores[:, idx], (filter[idx]))
                elif filter_type == "max":
                    c_scores[:, idx] = ndimage.filters.maximum_filter(c_scores[:, idx], (filter[idx]))
            scores_postprocessed[audio_id] = create_score_dataframe(
                scores=c_scores,
                timestamps=encoder._frame_to_time(np.arange(len(c_scores) + 1)),
                event_classes=encoder.labels,
            )
        else:
            scores_postprocessed[audio_id] = scores_raw[audio_id]

    return scores_raw, scores_postprocessed


def merge_maestro_ground_truth(clip_ground_truth):
    ground_truth = defaultdict(list)
    for clip_id in clip_ground_truth:
        file_id, clip_onset_time, clip_offset_time = clip_id.rsplit('-', maxsplit=2)
        clip_onset_time = int(clip_onset_time) // 100
        ground_truth[file_id].extend([
            (clip_onset_time + event_onset_time, clip_onset_time + event_offset_time, event_class)
            for event_onset_time, event_offset_time, event_class in clip_ground_truth[clip_id]
        ])
    return merge_overlapping_events(ground_truth)


def merge_overlapping_events(ground_truth_events):
    for clip_id, events in ground_truth_events.items():
        per_class_events = defaultdict(list)
        for event in events:
            per_class_events[event[2]].append(event)
        ground_truth_events[clip_id] = []
        for event_class, events in per_class_events.items():
            events = sorted(events)
            merged_events = []
            current_offset = -1e6
            for event in events:
                if event[0] > current_offset:
                    merged_events.append(list(event))
                else:
                    merged_events[-1][1] = max(current_offset, event[1])
                current_offset = merged_events[-1][1]
            ground_truth_events[clip_id].extend(merged_events)
    return ground_truth_events


def get_segment_scores_and_overlap_add(frame_scores, audio_durations, event_classes, segment_length=1.):
    """
    Calculate segment scores by overlapping and adding frame scores.
    It has two functions：
    (1) change the resolution of the prediction (upsample to "segment_length");
    (2) aggregate the results from different files at the logit level over sliding windows;

    Args:
        frame_scores (dict): A dictionary containing frame scores for each clip.
        audio_durations (dict): A dictionary containing the duration of each audio file.
        event_classes (list): A list of event classes.
        segment_length (float, optional): The length of each segment. Defaults to 1.

    Returns:
        dict: A dictionary containing segment scores for each audio file.

    Example:
        >>> event_classes = ['a', 'b', 'c']
        >>> audio_durations = {'f1': 201.6, 'f2':133.1, 'f3':326}
        >>> frame_scores = {\
            f'{file_id}-{int(100*onset)}-{int(100*(onset+10.))}': create_score_dataframe(np.random.rand(156,3), np.arange(157.)*0.064, event_classes)\
            for file_id in audio_durations for onset in range(int((audio_durations[file_id]-9.)))\
        }
        >>> frame_scores.keys()
        >>> seg_scores = _get_segment_scores_and_overlap_add(frame_scores, audio_durations, event_classes, segment_length=1.)
        >>> [(key, validate_score_dataframe(value)[0][-3:]) for key, value in seg_scores.items()]
    """
    segment_scores_file = {}
    summand_count = {}
    keys = ['onset', 'offset'] + event_classes
    for clip_id in frame_scores:
        file_id, clip_onset_time, clip_offset_time = clip_id.rsplit('-', maxsplit=2)
        clip_onset_time = float(clip_onset_time) / 100
        clip_offset_time = float(clip_offset_time) / 100
        if file_id not in segment_scores_file:
            segment_scores_file[file_id] = np.zeros(
                (ceil(audio_durations[file_id] / segment_length), len(event_classes)))
            summand_count[file_id] = np.zeros_like(segment_scores_file[file_id])
        segment_scores_clip = get_segment_scores(frame_scores[clip_id][keys],
                                                 clip_length=(clip_offset_time - clip_onset_time),
                                                 segment_length=1.)[event_classes].to_numpy()
        seg_idx = int(clip_onset_time // segment_length)
        segment_scores_file[file_id][seg_idx:seg_idx + len(segment_scores_clip)] += segment_scores_clip
        summand_count[file_id][seg_idx:seg_idx + len(segment_scores_clip)] += 1
    return {
        file_id: create_score_dataframe(
            segment_scores_file[file_id] / np.maximum(summand_count[file_id], 1),
            np.minimum(np.arange(0., audio_durations[file_id] + segment_length, segment_length),
                       audio_durations[file_id]),
            event_classes,
        )
        for file_id in segment_scores_file
    }


def get_segment_scores(scores_df, clip_length, segment_length=1.):
    """
    Calculate segment scores for a given scores dataframe.

    Args:
        scores_df (pd.DataFrame): Dataframe containing scores for each event class.
        clip_length (float): Length of the audio clip in seconds.
        segment_length (float, optional): Length of each segment in seconds. Defaults to 1.

    Returns:
        pd.DataFrame: Dataframe containing segment scores for each event class.

    Example:
        >>> scores_arr = np.random.rand(156,3)
        >>> timestamps = np.arange(157)*0.064
        >>> event_classes = ['a', 'b', 'c']
        >>> scores_df = create_score_dataframe(scores_arr, timestamps, event_classes)
        >>> seg_scores_df = _get_segment_scores(scores_df, clip_length=10., segment_length=1.)
    """
    frame_timestamps, event_classes = validate_score_dataframe(scores_df)
    scores_arr = scores_df[event_classes].to_numpy()
    segment_scores = []
    segment_timestamps = []
    seg_onset_idx = 0
    seg_offset_idx = 0
    for seg_onset in np.arange(0., clip_length, segment_length):
        seg_offset = seg_onset + segment_length
        while frame_timestamps[seg_onset_idx + 1] <= seg_onset:
            seg_onset_idx += 1
        while seg_offset_idx < len(scores_arr) and frame_timestamps[seg_offset_idx] < seg_offset:
            seg_offset_idx += 1
        seg_weights = (np.minimum(frame_timestamps[seg_onset_idx + 1:seg_offset_idx + 1], seg_offset) -
                       np.maximum(frame_timestamps[seg_onset_idx:seg_offset_idx], seg_onset))
        segment_scores.append(
            (seg_weights[:, None] * scores_arr[seg_onset_idx:seg_offset_idx]).sum(0) / seg_weights.sum())
        segment_timestamps.append(seg_onset)
    segment_timestamps.append(clip_length)
    return create_score_dataframe(np.array(segment_scores), np.array(segment_timestamps), event_classes)
