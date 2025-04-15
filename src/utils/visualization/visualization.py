import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchaudio.transforms as T


def plot_waveform(waveform, sample_rate, title):
    waveform = waveform.numpy()

    if waveform.ndim == 1:
        waveform = np.array([waveform])

    num_channels, num_frames = waveform.shape
    time_axis = np.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].set_xlabel("time")
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show(block=False)


def plot_spectrogram(specgram, xlabel, ylabel, extend=None, title=None):
    fig, ax = plt.subplots(1, 1)
    ax.set_title(title or "Spectrogram (db)")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    im = ax.imshow(T.AmplitudeToDB(stype="magnitude")(specgram),
                   origin="lower", aspect="auto", extent=extend)
    fig.colorbar(im, ax=ax)
    plt.show(block=False)


def plot_timestamp(audio_df: pd.DataFrame, total_time):
    event_list = list(set(audio_df["event_label"]))
    event_heights = dict()
    for i, name in enumerate(event_list):
        event_heights[name] = i + 0.25
    fig, ax = plt.subplots()
    for _, event in audio_df.iterrows():
        class_height = event_heights[event["event_label"]]
        begin_time = event["onset"]
        end_time = event["offset"]
        rect = plt.Rectangle((begin_time, class_height),
                             end_time-begin_time, 0.5)
        ax.add_patch(rect)
    ax.set_xlim(0, total_time)
    ax.set_ylim(0, max(event_heights.values()) + 0.5)
    ax.set_yticks([i + 0.25 for i in event_heights.values()])
    ax.set_yticklabels(event_list)
    plt.show(block=False)
