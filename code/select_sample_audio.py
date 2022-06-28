from conf import *
import os
import shutil
import numpy as np
import soundfile

SAMPLE_AUDIO_DIR = os.path.join(DATA_DIR, "sample_audios")
try:
    os.mkdir(SAMPLE_AUDIO_DIR)
except FileExistsError:
    pass

SYNC_DIR, REAL_DIR = os.path.join(SAMPLE_AUDIO_DIR, "sync"), os.path.join(SAMPLE_AUDIO_DIR, "real")

try:
    shutil.rmtree(SYNC_DIR)
    shutil.rmtree(REAL_DIR)
except FileNotFoundError:
    pass

os.mkdir(SYNC_DIR)
os.mkdir(REAL_DIR)


def extract_audios(src_path, target_dir):
    data = np.load(src_path)
    soundfile.write(os.path.join(target_dir, "ref.wav"), data["ref"], SR)
    soundfile.write(os.path.join(target_dir, "truth.wav"), data["truth"], SR)
    soundfile.write(os.path.join(target_dir, "mixed_input.wav"), data["input"], SR)


for i in range(13):
    sync_dir = os.path.join(SYNC_DIR, str(i))
    real_dir = os.path.join(REAL_DIR, str(i))
    try:
        shutil.rmtree(sync_dir)
        shutil.rmtree(real_dir)
    except FileNotFoundError:
        pass
    os.mkdir(sync_dir)
    os.mkdir(real_dir)

    extract_audios(os.path.join(DATA_DIR, "train", f"{i:06}.npz"), sync_dir)
    extract_audios(os.path.join(DATA_DIR, "test", f"{i:06}.npz"), real_dir)
