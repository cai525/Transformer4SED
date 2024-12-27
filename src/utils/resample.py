# Import librosa library
import argparse
import os

import librosa
import soundfile as sf
from tqdm import tqdm


def resample_audio(src_file_path, des_file_path, target_sr):
    """ Function to resample and save audio files """
    y, sr = librosa.load(src_file_path, sr=None)
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    sf.write(des_file_path, y_resampled, target_sr)


def ensure_dir(file_path):
    """ Function to create directory if it doesn't exist """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="resampe .wav files")
    parser.add_argument("--src_folder", type=str)
    parser.add_argument("--des_folder", type=str)
    parser.add_argument("--target_sr", type=int)
    args = parser.parse_args()
    # Define source and destination directories
    src_dir = args.src_folder
    des_dir = args.des_folder

    # Define target sample rate
    target_sr = args.target_sr

    # Recursively process all subdirectories and files
    for root, dirs, files in os.walk(src_dir):
        subdir_name = os.path.relpath(root, start=src_dir)
        for file in tqdm(files, desc=subdir_name):
            if file.endswith(".wav") and not (file[0] == '.'):
                # Construct full file path
                src_file_path = os.path.join(root, file)
                # Replace source directory with destination in the path
                des_file_path = os.path.join(des_dir, subdir_name, file)
                # Ensure the destination directory exists
                # ensure_dir(des_file_path)
                # Resample and save the audio file
                resample_audio(src_file_path, des_file_path, target_sr)
