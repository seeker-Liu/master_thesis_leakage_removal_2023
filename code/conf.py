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

SR = 44100

SAVE_AUDIO = True
SAVE_SPECTOGRAM = True

URMP_VIOLIN_CLARINET_PIECES = {17: (0, 2), 19: (1, 0), 37: (1, 3)}
URMP_VIOLIN_FLUTE_PIECE = {8: (1, 0), 17: (0, 1), 18: (0, 1), 37: (1, 0)}

VIOLIN_PROGRAM_NUM = 40
CLARINET_PROGRAM_NUM = 71


def stft_routine(wav):
    spec = librosa.stft(wav, n_fft=4096, hop_length=1024, window="hann", center=True).T
    return np.abs(spec), np.angle(spec)


def istft_routine(mag, phase):
    spec = mag * np.exp(phase * 1j).T
    return librosa.istft(spec, n_fft=4096, hop_length=1024, window="hann", center=True)
