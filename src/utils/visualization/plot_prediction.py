import os
import pandas as pd
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math


classes=['Blender', 
         'Dog', 
         'Vacuum_cleaner',
         'Running_water', 
         'Alarm_bell_ringing', 
         'Speech', 
         'Dishes',
         'Electric_shaver_toothbrush', 
         'Frying', 
         'Cat']

sample_rate = 16000
n_window = 2048
hop_length = 313
n_mels = 128
max_len_seconds = 10.
max_frames = math.ceil(max_len_seconds * sample_rate / hop_length)
pooling_time_ratio=2

class ManyHotEncoder:
    """"
        Adapted after DecisionEncoder.find_contiguous_regions method in
        https://github.com/DCASE-REPO/dcase_util/blob/master/dcase_util/data/decisions.py

        Encode labels into numpy arrays where 1 correspond to presence of the class and 0 absence.
        Multiple 1 can appear on the same line, it is for multi label problem.
    Args:
        labels: list, the classes which will be encoded
        n_frames: int, (Default value = None) only useful for strong labels. The number of frames of a segment.
    Attributes:
        labels: list, the classes which will be encoded
        n_frames: int, only useful for strong labels. The number of frames of a segment.
    """
    def __init__(self, labels, n_frames=None):
        if type(labels) in [np.ndarray, np.array]:
            labels = labels.tolist()
        self.labels = labels
        self.n_frames = n_frames

    def encode_strong_df(self, label_df):
        """Encode a list (or pandas Dataframe or Serie) of strong labels, they correspond to a given filename

        Args:
            label_df: pandas DataFrame or Series, contains filename, onset (in frames) and offset (in frames)
                If only filename (no onset offset) is specified, it will return the event on all the frames
                onset and offset should be in frames
        Returns:
            numpy.array
            Encoded labels, 1 where the label is present, 0 otherwise
        """

        assert self.n_frames is not None, "n_frames need to be specified when using strong encoder"
        if type(label_df) is str:
            if label_df == 'empty':
                y = np.zeros((self.n_frames, len(self.labels))) - 1
                return y
        y = np.zeros((self.n_frames, len(self.labels)))
        if type(label_df) is pd.DataFrame:
            if {"onset", "offset", "event_label"}.issubset(label_df.columns):
                for _, row in label_df.iterrows():
                    if not pd.isna(row["event_label"]):
                        i = self.labels.index(row["event_label"])
                        # onset = int(row["onset"])
                        # offset = int(row["offset"])
                        onset = int(row["onset"]/10.0*self.n_frames)
                        offset = int(row["offset"]/10.0*self.n_frames)
                        y[onset:offset, i] = 1  # means offset not included (hypothesis of overlapping frames, so ok)




        elif type(label_df) in [pd.Series, list, np.ndarray]:  # list of list or list of strings
            if type(label_df) is pd.Series:
                if {"onset", "offset", "event_label"}.issubset(label_df.index):  # means only one value
                    if not pd.isna(label_df["event_label"]):
                        i = self.labels.index(label_df["event_label"])
                        # onset = int(label_df["onset"])
                        # offset = int(label_df["offset"])
                        onset = int(label_df["onset"]/10.0*self.n_frames)
                        offset = int(label_df["offset"]/10.0*self.n_frames)
                        y[onset:offset, i] = 1
                    return y

            for event_label in label_df:
                # List of string, so weak labels to be encoded in strong
                if type(event_label) is str:
                    if event_label is not "":
                        i = self.labels.index(event_label)
                        y[:, i] = 1

                # List of list, with [label, onset, offset]
                elif len(event_label) == 3:
                    if event_label[0] is not "":
                        i = self.labels.index(event_label[0])
                        # onset = int(event_label[1])
                        # offset = int(event_label[2])
                        onset = int(event_label[1]/10.0*self.n_frames)
                        offset = int(event_label[2]/10.0*self.n_frames)
                        y[onset:offset, i] = 1

                else:
                    raise NotImplementedError("cannot encode strong, type mismatch: {}".format(type(event_label)))

        else:
            raise NotImplementedError("To encode_strong, type is pandas.Dataframe with onset, offset and event_label"
                                      "columns, or it is a list or pandas Series of event labels, "
                                      "type given: {}".format(type(label_df)))
        return y



def plot_events(labels, out_path,file):
    print(file)
    y_ticks = classes
    plt.figure(figsize=(15,8))
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(10))
    x_scale = np.arange(256) / 256 * 10
    
    #colors=['r','b','g']
    #types=['ground_truth','crnn_pred','cnn_transformer_pred']
    colors=['r']
    types=['ground_truth']
    
    for k,label in enumerate(labels):
        label = np.pad(label, ((1, 1), (0, 0)))
        begin_end = np.int64(np.logical_xor(label[1:, :], label[:-1, :]))
        # import pdb
        # pdb.set_trace()
        begin_end = np.argwhere(begin_end > 0)
        begin_end = sorted(begin_end,key=lambda x:(x[1],x[0]))
        #begin_end = begin_end[np.argsort(begin_end[:, 1])]
        if len(begin_end)==0:
            continue
        begin_end=np.array(begin_end)
        #print(begin_end)
        begin = begin_end[0::2]
        end = begin_end[1::2]
        end[:, 0] -= 1

        begin[:,1]=begin[:,1]*3+k
        end[:, 1] = end[:, 1] * 3 + k


        lines = []
        for j in range(len(begin)):
            #print("begin:", begin[j][0])
            #print("end:", end[j][0])
            if j == len(begin) - 1:

                line, = plt.plot([x_scale[begin[j][0]], x_scale[end[j][0]]], [begin[j][1], end[j][1]], color=colors[k],
                                 linewidth=6.0, label=types[k])

            else:
                line, = plt.plot([x_scale[begin[j][0]], x_scale[end[j][0]]], [begin[j][1], end[j][1]], color=colors[k],
                                 linewidth=6.0)

            lines.append(line)

    plt.legend()
    plt.title(file)
    plt.ylim([0, 30])
    plt.xlim([0, 10])
    plt.yticks(ticks=np.arange(0,30,3), labels=y_ticks,rotation=30)

    plt.savefig(out_path)




if __name__ == '__main__':
    
    # paths
    ground_truth = '../metadata/validation/validation.csv'
    #crnn_pred = '../RNNvsTransformer/main_mean_teacher_2022_04_12__03_18_46_supervised_train_train_modelNet_for_DA_vcnn11base_synth21_official_max_consistency_cost2.0_snr1_mixup_typeinter_scalerown/predictions/pred_validation.csv'
    #cnn_transformer_pred='../RNNvsTransformer/main_mean_teacher_2022_04_12__06_09_09_supervised_train_train_modelNet_for_DA_vcnn11base_1Transformer_noGRU_synth21_official_max_consistency_cost2.0_snr1_mixup_typeinter_scalerown/predictions/pred_validation.csv'

    # read files
    ground_truth = pd.read_csv(ground_truth,sep='\t')
    # crnn_pred = pd.read_csv(crnn_pred,sep='\t')
    # cnn_transformer_pred = pd.read_csv(cnn_transformer_pred,sep='\t')

    # set encoder
    strong_encoder = ManyHotEncoder(classes, n_frames=max_frames // pooling_time_ratio)
    file_list = ground_truth['filename'].tolist()
    file_list_no_repeat = []
    for id in file_list:
        if id not in file_list_no_repeat:
            file_list_no_repeat.append(id)


    total_plot_num=10000
    out_dir = 'label_figures'
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i,file in enumerate(file_list_no_repeat):
        out_path = os.path.join(out_dir,file[:-4]+'.png')

        this_ground_truth=ground_truth[ground_truth['filename']==file]
        ground_truth_label = strong_encoder.encode_strong_df(this_ground_truth)

        # this_crnn_pred = crnn_pred[crnn_pred['filename']==file]
        # crnn_pred_label = strong_encoder.encode_strong_df(this_crnn_pred)

        # this_cnn_transformer_pred = cnn_transformer_pred[cnn_transformer_pred['filename'] == file]
        # cnn_transformer_pred_label = strong_encoder.encode_strong_df(this_cnn_transformer_pred)

        plot_events([ground_truth_label], out_path,file)

        if i==total_plot_num:
            break

    print('Done ! Total plot num:{}'.format(total_plot_num))