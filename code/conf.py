import os
import librosa
import numpy as np

CODE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(CODE_DIR, "..")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "validation")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
BASELINE_MODEL_DIR = os.path.join(ROOT_DIR, "baseline_model")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoint")

SR = 44100

SAVE_AUDIO = True
SAVE_SPECTROGRAM = True
SAVE_16K = True
ADD_NOISE = False

URMP_VIOLIN_CLARINET_PIECES = {17: (0, 2), 19: (1, 0), 37: (1, 3)}
URMP_VIOLIN_FLUTE_PIECE = {8: (1, 0), 17: (0, 1), 18: (0, 1), 37: (1, 0)}

VIOLIN_PROGRAM_NUM = 40
CLARINET_PROGRAM_NUM = 71
FLUTE_PROGRAM_NUM = 73

# Length of each clip for training/evaluating, in seconds
AUDIO_CLIP_LENGTH = 5
AUDIO_CLIP_HOP = 2.5
BATCH_SIZE = 16
LEARNING_RATE = 1e-4


STFT_16K_PARAMS = {"n_fft": 512, "hop_length": 256, "window": "hann", "center": True}
STFT_44K_PARAMS = {"n_fft": 4096, "hop_length": 1024, "window": "hann", "center": True}


def stft_routine(wav, sr):
    params = STFT_44K_PARAMS if sr == SR else STFT_16K_PARAMS
    spec = librosa.stft(wav, **params).T
    return np.abs(spec), np.angle(spec)


def istft_routine(mag, phase, sr):
    params = STFT_44K_PARAMS if sr == SR else STFT_16K_PARAMS
    spec = (mag * np.exp(phase * 1j)).T
    return librosa.istft(spec, **params)
