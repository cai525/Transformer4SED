# Import librosa library
import os
import librosa
import soundfile as sf
from tqdm import tqdm

# Define source and destination directories
src_dir = "/home/mnt/mydataset/dcase2021/audio/validation/synthetic_validation/soundscapes/"
des_dir = "/home/mnt/mydataset/dcase2021/audio/validation/synthetic_validation/soundscapes_32k/"

# Define target sample rate
target_sr = 32_000

# Loop through all the files in the source directory
for file in tqdm(os.listdir(src_dir)):
    if os.path.splitext(file)[-1] == ".wav":
        # Load the audio file using librosa.load
        y, sr = librosa.load(os.path.join(src_dir, file), sr=None)
        # Resample the audio to the target sample rate using librosa.resample
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        # Save the resampled audio to the destination directory using librosa.output.write_wav
        sf.write(os.path.join(des_dir, file), y_resampled, target_sr)