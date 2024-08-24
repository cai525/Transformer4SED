import math

import numpy as np
import pandas as pd


class Encoder:

    def __init__(self, labels, audio_len, frame_len, frame_hop, net_pooling=1, sr=16000):
        if type(labels) in [np.ndarray, np.array]:
            labels = labels.tolist()
        self.labels = labels
        self.audio_len = audio_len
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.sr = sr
        self.net_pooling = net_pooling
        n_samples = self.audio_len * self.sr
        self.n_frames = int(math.ceil(n_samples / 2 / self.frame_hop) * 2 / self.net_pooling)

    def _time_to_frame(self, time):
        sample = time * self.sr
        frame = sample / self.frame_hop
        return np.clip(frame / self.net_pooling, a_min=0, a_max=self.n_frames)

    def _frame_to_time(self, frame):
        time = frame * self.net_pooling * self.frame_hop / self.sr
        return np.clip(time, a_min=0, a_max=self.audio_len)

    def encode_strong_df(self, events_df):
        # from event dict, generate strong label tensor sized as [n_frame, n_class]
        true_labels = np.zeros((self.n_frames, len(self.labels)))
        for _, row in events_df.iterrows():
            if not pd.isna(row['event_label']):
                label_idx = self.labels.index(row["event_label"])
                onset = int(self._time_to_frame(row["onset"]))
                offset = int(np.ceil(self._time_to_frame(row["offset"])))
                true_labels[onset:offset, label_idx] = 1
        return true_labels

    def encode_weak(self, events):
        # from event dict, generate weak label tensor sized as [n_class]
        labels = np.zeros((len(self.labels)))
        if len(events) == 0:
            return labels
        else:
            for event in events:
                labels[self.labels.index(event)] = 1
            return labels

    def decode_strong(self, outputs):
        #from the network output sized [n_frame, n_class], generate the label/onset/offset lists
        pred = []
        for i, label_column in enumerate(outputs.T):  #outputs size = [n_class, frames]
            change_indices = self.find_contiguous_regions(label_column)
            for row in change_indices:
                onset = self._frame_to_time(row[0])
                offset = self._frame_to_time(row[1])
                onset = np.clip(onset, a_min=0, a_max=self.audio_len)
                offset = np.clip(offset, a_min=0, a_max=self.audio_len)
                pred.append([self.labels[i], onset, offset])
        return pred

    def decode_weak(self, outputs):
        result_labels = []
        for i, value in enumerate(outputs):
            if value == 1:
                result_labels.append(self.labels[i])
        return result_labels

    def find_contiguous_regions(self, array):
        #find at which frame the label changes in the array
        change_indices = np.logical_xor(array[1:], array[:-1]).nonzero()[0]
        #shift indices to focus the frame after
        change_indices += 1
        if array[0]:
            #if first element of array is True(1), add 0 in the beggining
            #change_indices = np.append(0, change_indices)
            change_indices = np.r_[0, change_indices]
        if array[-1]:
            #if last element is True, add the length of array
            change_indices = np.r_[change_indices, array.size]
        #reshape the result into two columns
        return change_indices.reshape((-1, 2))


class Encoder_v2:
    """ Not different to the above one, except to add the decoder_ratio """

    def __init__(self, labels, audio_len, frame_len, frame_hop, net_pooling=1, sr=16000, decoder_ratio=1):
        if type(labels) in [np.ndarray, np.array]:
            labels = labels.tolist()
        self.labels = labels
        self.audio_len = audio_len
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.sr = sr
        self.net_pooling = net_pooling
        n_samples = self.audio_len * self.sr
        self.n_frames = math.ceil(math.ceil(n_samples / 2 / self.frame_hop) * 2 / self.net_pooling) * decoder_ratio
        self.decoder_ratio = decoder_ratio

    def _time_to_frame(self, time):
        sample = time * self.sr
        frame = sample / self.frame_hop
        return np.clip(frame / self.net_pooling * self.decoder_ratio, a_min=0, a_max=self.n_frames)

    def _frame_to_time(self, frame):
        time = frame / self.decoder_ratio * self.net_pooling * self.frame_hop / self.sr
        return np.clip(time, a_min=0, a_max=self.audio_len)

    def encode_strong_df(self, events_df):
        # from event dict, generate strong label tensor sized as [n_frame, n_class]
        true_labels = np.zeros((self.n_frames, len(self.labels)))
        for _, row in events_df.iterrows():
            if not pd.isna(row['event_label']):
                label_idx = self.labels.index(row["event_label"])
                onset = int(self._time_to_frame(row["onset"]))  #버림 -> 해당 time frame에 걸쳐있으면 true
                offset = int(np.ceil(self._time_to_frame(row["offset"])))  #올림 -> 해당 time frame에 걸쳐있으면 true
                true_labels[onset:offset, label_idx] = 1
        return true_labels

    def encode_weak(self, events):
        # from event dict, generate weak label tensor sized as [n_class]
        labels = np.zeros((len(self.labels)))
        if len(events) == 0:
            return labels
        else:
            for event in events:
                labels[self.labels.index(event)] = 1
            return labels

    def decode_strong(self, outputs):
        #from the network output sized [n_frame, n_class], generate the label/onset/offset lists
        pred = []
        for i, label_column in enumerate(outputs.T):  #outputs size = [n_class, frames]
            change_indices = self.find_contiguous_regions(label_column)
            for row in change_indices:
                onset = self._frame_to_time(row[0])
                offset = self._frame_to_time(row[1])
                onset = np.clip(onset, a_min=0, a_max=self.audio_len)
                offset = np.clip(offset, a_min=0, a_max=self.audio_len)
                pred.append([self.labels[i], onset, offset])
        return pred

    def decode_weak(self, outputs):
        result_labels = []
        for i, value in enumerate(outputs):
            if value == 1:
                result_labels.append(self.labels[i])
        return result_labels

    def find_contiguous_regions(self, array):
        #find at which frame the label changes in the array
        change_indices = np.logical_xor(array[1:], array[:-1]).nonzero()[0]
        #shift indices to focus the frame after
        change_indices += 1
        if array[0]:
            #if first element of array is True(1), add 0 in the beggining
            #change_indices = np.append(0, change_indices)
            change_indices = np.r_[0, change_indices]
        if array[-1]:
            #if last element is True, add the length of array
            change_indices = np.r_[change_indices, array.size]
        #reshape the result into two columns
        return change_indices.reshape((-1, 2))
