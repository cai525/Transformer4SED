import copy
import os

import matplotlib.pyplot as plt
import torch


def get_pred_statistic(device):
    pred_statistic = {}
    pred_statistic['mean'] = {}
    pred_statistic['mean']['positive'] = []
    pred_statistic['mean']['negtive'] = []
    pred_statistic['mean']['positive_ema'] = torch.tensor([0.5] * 10).to(device)
    pred_statistic['mean']['negtive_ema'] = torch.tensor([0.5] * 10).to(device)
    pred_statistic['std'] = {}
    pred_statistic['std']['positive'] = []
    pred_statistic['std']['negtive'] = []
    pred_statistic['std']['positive_ema'] = torch.tensor([0.] * 10).to(device)
    pred_statistic['std']['negtive_ema'] = torch.tensor([0.] * 10).to(device)
    return pred_statistic


def plot_statistic(statistic_input, output_name, configs):
    #statistic_input shape, (steps,Class)
    color_list = ['red', 'peru', 'darkorange', 'gold', 'palegreen', 'cyan', 'dodgerblue', 'blue', 'm', 'pink']
    label_list = [
        "Alarm_bell_ringing", "Blender", "Cat", "Dishes", "Dog", "Electric_shaver_toothbrush", "Frying",
        "Running_water", "Speech", "Vacuum_cleaner"
    ]
    plt.figure()
    for i in range(10):
        plt.plot(statistic_input[:, i], color=color_list[i], label=label_list[i])
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(configs['generals']['save_folder'], output_name))
    plt.close()
    pass


def plot_pred_statistic(pred_statistic, configs, type='SED'):

    pred_mean_positive = pred_statistic['mean']['positive']
    pred_std_positive = pred_statistic['std']['positive']
    pred_mean_negtive = pred_statistic['mean']['negtive']
    pred_std_negtive = pred_statistic['std']['negtive']

    if len(pred_mean_positive) > 0:
        pred_mean_positive = torch.cat(pred_mean_positive, dim=0).cpu().numpy()
        plot_statistic(pred_mean_positive, '{}_pred_mean_positive.png'.format(type), configs)

        pred_std_positive = torch.cat(pred_std_positive, dim=0).cpu().numpy()
        plot_statistic(pred_std_positive, '{}_pred_std_positive.png'.format(type), configs)

        pred_mean_negtive = torch.cat(pred_mean_negtive, dim=0).cpu().numpy()
        plot_statistic(pred_mean_negtive, '{}_pred_mean_negtive.png'.format(type), configs)

        pred_std_negtive = torch.cat(pred_std_negtive, dim=0).cpu().numpy()
        plot_statistic(pred_std_negtive, '{}_pred_std_negtive.png'.format(type), configs)

    pass


def update_pred_statistic(pred, labels, old_pred_statistic, step, ema_factor=0.999):
    # update pred statistic
    alpha = min(1 - 1 / step, ema_factor)
    if len(pred.shape) == 3:
        for i in range(10):
            #positive
            this_class_index = (labels[:, i, :] == 1)
            if len(this_class_index[0]) == 0:
                this_class_pred_mean = old_pred_statistic['mean']['positive_ema'][i]
                this_class_pred_std = old_pred_statistic['std']['positive_ema'][i]
            else:
                this_class_pred = pred[:, i, :][this_class_index]
                this_class_pred_mean = torch.mean(this_class_pred)
                this_class_pred_std = torch.std(this_class_pred)

            old_pred_statistic['mean']['positive_ema'][i] = alpha * old_pred_statistic['mean']['positive_ema'][i] + (
                1 - alpha) * this_class_pred_mean
            old_pred_statistic['std']['positive_ema'][i] = alpha * old_pred_statistic['std']['positive_ema'][i] + (
                1 - alpha) * this_class_pred_std

            #old_pred_statistic[i]['mean']['positive'].append(this_class_pred_mean)
            #old_pred_statistic[i]['std']['positive'].append(this_class_pred_std)
            #negtive
            this_class_index = torch.where(labels[:, i, :] == 0)
            if len(this_class_index[0]) == 0:
                this_class_pred_mean = old_pred_statistic['mean']['negtive_ema'][i]
                this_class_pred_std = old_pred_statistic['std']['negtive_ema'][i]
            else:
                this_class_pred = pred[:, i, :][this_class_index]
                this_class_pred_mean = torch.mean(this_class_pred)
                this_class_pred_std = torch.std(this_class_pred)

            old_pred_statistic['mean']['negtive_ema'][i] = alpha * old_pred_statistic['mean']['negtive_ema'][i] + (
                1 - alpha) * this_class_pred_mean
            old_pred_statistic['std']['negtive_ema'][i] = alpha * old_pred_statistic['std']['negtive_ema'][i] + (
                1 - alpha) * this_class_pred_std

        old_pred_statistic['mean']['positive'].append(
            copy.deepcopy(old_pred_statistic['mean']['positive_ema']).unsqueeze(0))
        old_pred_statistic['std']['positive'].append(
            copy.deepcopy(old_pred_statistic['std']['positive_ema']).unsqueeze(0))
        old_pred_statistic['mean']['negtive'].append(
            copy.deepcopy(old_pred_statistic['mean']['negtive_ema']).unsqueeze(0))
        old_pred_statistic['std']['negtive'].append(
            copy.deepcopy(old_pred_statistic['std']['negtive_ema']).unsqueeze(0))
    elif len(pred.shape) == 2:  #for weak pred
        for i in range(10):
            # positive
            this_class_index = torch.where(labels[:, i] == 1)
            if len(this_class_index[0]) == 0:
                this_class_pred_mean = old_pred_statistic['mean']['positive_ema'][i]
                this_class_pred_std = old_pred_statistic['std']['positive_ema'][i]
            else:
                this_class_pred = pred[:, i][this_class_index]
                this_class_pred_mean = torch.mean(this_class_pred)
                this_class_pred_std = torch.std(this_class_pred)

            old_pred_statistic['mean']['positive_ema'][i] = alpha * old_pred_statistic['mean']['positive_ema'][i] + (
                1 - alpha) * this_class_pred_mean
            old_pred_statistic['std']['positive_ema'][i] = alpha * old_pred_statistic['std']['positive_ema'][i] + (
                1 - alpha) * this_class_pred_std

            # old_pred_statistic[i]['mean']['positive'].append(this_class_pred_mean)
            # old_pred_statistic[i]['std']['positive'].append(this_class_pred_std)
            # negtive
            this_class_index = torch.where(labels[:, i] == 0)
            if len(this_class_index[0]) == 0:
                this_class_pred_mean = old_pred_statistic['mean']['negtive_ema'][i]
                this_class_pred_std = old_pred_statistic['std']['negtive_ema'][i]
            else:
                this_class_pred = pred[:, i][this_class_index]
                this_class_pred_mean = torch.mean(this_class_pred)
                this_class_pred_std = torch.std(this_class_pred)

            old_pred_statistic['mean']['negtive_ema'][i] = alpha * old_pred_statistic['mean']['negtive_ema'][i] + (
                1 - alpha) * this_class_pred_mean
            old_pred_statistic['std']['negtive_ema'][i] = alpha * old_pred_statistic['std']['negtive_ema'][i] + (
                1 - alpha) * this_class_pred_std

        old_pred_statistic['mean']['positive'].append(
            copy.deepcopy(old_pred_statistic['mean']['positive_ema']).unsqueeze(0))
        old_pred_statistic['std']['positive'].append(
            copy.deepcopy(old_pred_statistic['std']['positive_ema']).unsqueeze(0))
        old_pred_statistic['mean']['negtive'].append(
            copy.deepcopy(old_pred_statistic['mean']['negtive_ema']).unsqueeze(0))
        old_pred_statistic['std']['negtive'].append(
            copy.deepcopy(old_pred_statistic['std']['negtive_ema']).unsqueeze(0))

    else:

        raise ValueError('invalid pred shape, must be 3')
    return old_pred_statistic