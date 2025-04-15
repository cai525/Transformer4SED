import os
import pandas as pd
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from utils.dataset import *
from utils.utils import *
from utils.settings import *
from utils.data_aug import *
from utils.evaluation_measures import compute_per_intersection_macro_f1,compute_sed_eval_metrics


def plot_events(labels,names, out_path,file):
    print(file)

    classes = [
        "Vacuum_cleaner",
        "Speech",
        "Running_water",
        "Frying",
        "Electric_shaver_toothbrush",
        "Dog",
        "Dishes",
        "Cat",
        "Blender",
        "Alarm_bell_ringing",
    ]

    t_len =labels[0].shape[0]
    y_ticks = classes
    plt.figure(figsize=(15,8))
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(10))
    x_scale = np.arange(t_len) / t_len * 10
    if len(names) == 3:
        colors=['r','b','g']
    elif len(names) == 2:
        colors=['r','b']
    else:
        raise ValueError('only support 2 or 3')
    types=names
    for k,label in enumerate(labels):
        ##
        label = np.flip(label,axis=1)
        ##
        label = np.pad(label, ((1, 1), (0, 0)))
        begin_end = np.int64(np.logical_xor(label[1:, :], label[:-1, :]))
        begin_end = np.argwhere(begin_end > 0)
        begin_end = sorted(begin_end,key=lambda x:(x[1],x[0]))
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
    plt.close()
input_paths = {
    "ground_truth":'/home/mnt/likang/AST-SED/meta/validation/validation.tsv',
    "pred1": 'exps/config_p_ast_gru_lgd_specaugf_multitask.yaml_2022_12_11_19_03_39_new_exp_gpu=0_iter_0/psds_teacher/predictions_dtc0.1_gtc0.1_cttc0.3/predictions_th_0.49.tsv',
}

##load sed output
input_data ={}
for name,path in input_paths.items():
    input_data[name] = pd.read_csv(path,sep='\t')

feature_cfg = {
    "n_mels": 128,
    "n_fft": 1024,
    "hopsize": 320,
    "win_length": 800,
    "fmin": 0.0,
    "fmax": None,
    "audio_max_len": 10,
    "sr": 32000,
    "net_subsample": 1
}
LabelDict = get_labeldict()
strong_encoder = get_encoder_passt(LabelDict, feature_cfg, feature_cfg["audio_max_len"])


##get filename_list
file_list=input_data['ground_truth']['filename'].tolist()
file_list_no_repeat = []
for id in file_list:
    if id not in file_list_no_repeat:
        file_list_no_repeat.append(id)


#set config
total_plot_num=10000
out_dir = 'label_figures'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#start plot
for i,file in enumerate(file_list_no_repeat):
    out_path = os.path.join(out_dir,file[:-4]+'.png')
    plot_func_input=[]
    plot_func_input_name=[]
    for name,data in input_data.items():
        predict=data[data['filename']==file]
        strong_label = strong_encoder.encode_strong_df(predict)
        plot_func_input.append(strong_label)
        plot_func_input_name.append(name)
    plot_events(plot_func_input,plot_func_input_name,out_path,file)
    if i==total_plot_num:
        break

print('Done ! Total plot num:{}'.format(total_plot_num))