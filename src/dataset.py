#Some codes are adopted from https://github.com/DCASE-REPO/DESED_task
import os.path

from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, Sampler

from src.preprocess.feats_extraction import waveform_modification

#dataset classes
class StronglyLabeledDataset(Dataset):

    def __init__(self, tsv_read, dataset_dir, return_name, encoder):
        #refer: https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html
        self.dataset_dir = dataset_dir
        self.encoder = encoder
        self.pad_to = encoder.audio_len * encoder.sr
        self.return_name = return_name

        #construct clip dictionary with filename = {path, events} where events = {label, onset and offset}
        clips = {}

        tk = tqdm(tsv_read.iterrows(), total=len(tsv_read), leave=False, desc="strong dataset")
        for _, row in tk:
            if row["filename"] not in clips.keys():
                clips[row["filename"]] = {"path": os.path.join(dataset_dir, row["filename"]), "events": []}
            
            if not np.isnan(row["onset"]):
                clips[row["filename"]]["events"].append({
                    "event_label": row["event_label"],
                    "onset": row["onset"],
                    "offset": row["offset"]
                })

        self.clips = clips  #dictionary for each clip
        self.clip_list = list(clips.keys())  # list of all clip names

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.clip_list[idx]
        clip = self.clips[filename]
        path = clip["path"]

        # get wav
        wav, pad_mask = waveform_modification(path, self.pad_to, self.encoder)

        # get labels
        events = clip["events"]
        if not len(events):  #label size = [frames, nclass]
            label = torch.zeros(self.encoder.n_frames, len(self.encoder.labels)).float()
        else:
            label = self.encoder.encode_strong_df(pd.DataFrame(events))
            label = torch.from_numpy(label).float()
        label = label.transpose(0, 1)

        # return
        out_args = [wav, label, pad_mask, idx]
        if self.return_name:
            out_args.extend([filename, path])
        return out_args


class WeaklyLabeledDataset(Dataset):

    def __init__(self, tsv_read, dataset_dir, return_name, encoder):
        self.dataset_dir = dataset_dir
        self.encoder = encoder
        self.pad_to = encoder.audio_len * self.encoder.sr
        self.return_name = return_name

        #construct clip dictionary with file name, path, label, onset and offset
        clips = {}
        for _, row in tsv_read.iterrows():
            if row["filename"] not in clips.keys():
                clips[row["filename"]] = {
                    "path": os.path.join(dataset_dir, row["filename"]),
                    "events": row["event_labels"].split(",")
                }
        #dictionary for each clip
        self.clips = clips
        self.clip_list = list(clips.keys())  # all file names

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.clip_list[idx]
        clip = self.clips[filename]
        path = clip["path"]

        #get labels
        events = clip["events"]
        label = torch.zeros(self.encoder.n_frames, len(self.encoder.labels))
        if len(events):
            label_encoded = self.encoder.encode_weak(events)  # label size: [n_class]
            label[0, :] = torch.from_numpy(label_encoded).float()  # label size: [n_frames, n_class]
        label = label.transpose(0, 1)

        # get wav
        wav, pad_mask = waveform_modification(path, self.pad_to, self.encoder)
        # return
        out_args = [wav, label, pad_mask, idx]
        if self.return_name:
            out_args.extend([filename, path])
        return out_args


class UnlabeledDataset(Dataset):

    def __init__(self, dataset_dir, return_name, encoder):
        self.encoder = encoder
        self.pad_to = encoder.audio_len * self.encoder.sr
        self.return_name = return_name

        #list of clip directories
        self.clips = glob(os.path.join(dataset_dir, '*.wav'))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.clips[idx]
        filename = os.path.split(path)[-1]

        #produce empty label
        label = torch.zeros(self.encoder.n_frames, len(self.encoder.labels)).float()
        label = label.transpose(0, 1)

        # get wav
        wav, pad_mask = waveform_modification(path, self.pad_to, self.encoder)
        # return
        out_args = [wav, label, pad_mask, idx]
        if self.return_name:
            out_args.extend([filename, path])
        return out_args

class ConcatDatasetBatchSampler(Sampler):
    def __init__(self, samplers, batch_sizes, epoch=0):
        self.batch_sizes = batch_sizes
        self.samplers = samplers
        self.offsets = [0] + np.cumsum([len(x) for x in self.samplers]).tolist()[:-1]

        self.epoch = epoch
        self.set_epoch(self.epoch)

    def _iter_one_dataset(self, c_batch_size, c_sampler, c_offset):
        batch = []
        for idx in c_sampler:
            batch.append(c_offset + idx)
            if len(batch) == c_batch_size:
                yield batch

    def set_epoch(self, epoch):
        if hasattr(self.samplers[0], "epoch"):
            for s in self.samplers:
                s.set_epoch(epoch)

    def __iter__(self):
        iterators = [iter(i) for i in self.samplers]
        tot_batch = []
        for b_num in range(len(self)):
            for samp_idx in range(len(self.samplers)):
                c_batch = []
                while len(c_batch) < self.batch_sizes[samp_idx]:
                    c_batch.append(self.offsets[samp_idx] + next(iterators[samp_idx]))
                tot_batch.extend(c_batch)
            yield tot_batch
            tot_batch = []

    def __len__(self):
        min_len = float("inf")
        for idx, sampler in enumerate(self.samplers):
            c_len = (len(sampler)) // self.batch_sizes[idx]
            min_len = min(c_len, min_len)
        return min_len