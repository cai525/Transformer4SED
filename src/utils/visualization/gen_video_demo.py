import argparse
import os
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from moviepy.editor import AudioFileClip, ImageSequenceClip

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_name', type=str)
    parser.add_argument('--wav_dir', type=str)
    parser.add_argument('--tsv_dir', type=str)
    parser.add_argument('--save_dir', type=str, default="./")
    args = parser.parse_args()

    # read data
    wav_name = args.wav_name
    df = pd.read_csv(os.path.join(args.tsv_dir, wav_name + ".tsv"), sep='\t')
    audio_clip = AudioFileClip(os.path.join(args.wav_dir, wav_name + ".wav"))

    if not os.path.exists("./.frames"):
        os.mkdir("./.frames")  # make temporary directory for saving frames

    sound_events = df.columns[2:]

    # color mapping
    color_palette = sns.color_palette("hsv", len(sound_events))
    color_map = dict(zip(sound_events, color_palette))

    # video parameter setting
    fps = 30
    duration = df['offset'].iloc[-1]
    time_steps = np.arange(0, duration, 1 / fps)

    frames = []

    for t in time_steps:
        row = df[(df['onset'] <= t) & (df['offset'] > t)].iloc[0]

        probabilities = row[sound_events].values

        # draw a histogram for frame t
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(sound_events, probabilities, color=[color_map[event] for event in sound_events])
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability', fontsize=10, color="#333333")
        ax.set_title(
            '{wav_name}   {t:.2f}s'.format(
                wav_name=wav_name,
                t=t,
            ),
            fontsize=12,
            color="#333333",
        )
        ax.tick_params(axis='y', labelsize=10, colors="#333333")
        ax.tick_params(axis='x', labelsize=10, colors="#333333")

        plt.tight_layout()
        frame_path = f'./.frames/frame_{t:.2f}.png'
        plt.savefig(frame_path)
        plt.close(fig)
        frames.append(frame_path)

    # make a video clip
    video_clip = ImageSequenceClip(frames, fps=fps).set_audio(audio_clip)
    video_clip.write_videofile(
        os.path.join(args.save_dir, '{wav_name}.mp4'.format(wav_name=wav_name)),
        codec='libx264',
    )

    # delete the temporary directory
    shutil.rmtree("./.frames")
