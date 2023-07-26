import os
import librosa
import numpy as np
import tensorflow as tf

CODE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(CODE_DIR, "..")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "validation")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
BASELINE_MODEL_DIR = os.path.join(ROOT_DIR, "baseline_model")
WAVE_U_NET_MODEL_DIR = os.path.join(ROOT_DIR, "wave_u_net_model")
WAVE_U_NET_BASELINE_MODEL_DIR = os.path.join(ROOT_DIR, "wave_u_net_baseline_model")

MODEL_DIRS = {
    "original": MODEL_DIR,
    "baseline": BASELINE_MODEL_DIR,
    "wave-u-net": WAVE_U_NET_MODEL_DIR,
    "wave-u-net-baseline": WAVE_U_NET_BASELINE_MODEL_DIR,
}

SR = 44100

SAVE_AUDIO = True
SAVE_SPECTROGRAM = True
SAVE_16K = False
ADD_NOISE = False

URMP_VIOLIN_CLARINET_PIECES = {17: (0, 2), 19: (1, 0), 37: (1, 3)}
URMP_VIOLIN_FLUTE_PIECE = {8: (1, 0), 17: (0, 1), 18: (0, 1), 37: (1, 0)}

PIANO_PROGRAM_NUM = 0
VIOLIN_PROGRAM_NUM = 40
CLARINET_PROGRAM_NUM = 71
FLUTE_PROGRAM_NUM = 73

# Length of each clip for training/evaluating, in seconds
AUDIO_CLIP_LENGTH = 5
AUDIO_CLIP_HOP = 2.5
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

WAVE_U_NET_INPUT_LENGTH = 217075
WAVE_U_NET_OUTPUT_LENGTH = 86021

STFT_16K_PARAMS = {"n_fft": 512, "hop_length": 256, "window": "hann", "center": True}
STFT_44K_PARAMS = {"n_fft": 4096, "hop_length": 1024, "window": "hann", "center": True}

DATASET_PARAMS = {
    "original": {
        "use_spectrogram": True,
        "use_irm": False,
        "sr": SR,
        "sr_postfix_str": "",
        "target_input_length": None,
        "target_output_length": None,
        "batch_size": BATCH_SIZE,
        "output_data_mapper": lambda x1, x2, y: ((x1, x2), y)
    },
    "baseline": {
        "use_spectrogram": False,
        "use_irm": True,
        "sr": 16000,
        "sr_postfix_str": "_16k",
        "target_input_length": None,
        "target_output_length": None,
        "batch_size": BATCH_SIZE,
        "output_data_mapper": lambda x1, x2, y: ((x1, x2), y)
    },
    "wave-u-net": {
        "use_spectrogram": False,
        "use_irm": False,
        "sr": SR,
        "sr_postfix_str": "",
        "target_input_length": WAVE_U_NET_INPUT_LENGTH,
        "target_output_length": WAVE_U_NET_OUTPUT_LENGTH,
        "batch_size": 8,  # GPU memory limit
        "output_data_mapper": lambda x1, x2, y:
            ((tf.expand_dims(x1, -1), tf.expand_dims(x2, -1)), tf.expand_dims(y, -1))
    },
    "wave-u-net-baseline": {
        "use_spectrogram": False,
        "use_irm": False,
        "sr": SR,
        "sr_postfix_str": "",
        "target_input_length": WAVE_U_NET_INPUT_LENGTH,
        "target_output_length": WAVE_U_NET_OUTPUT_LENGTH,
        "batch_size": 8,
        "output_data_mapper": lambda x1, x2, y: ((tf.expand_dims(x1, -1),), tf.expand_dims(y, -1))
    }
}


def stft_routine(wav, sr):
    params = STFT_44K_PARAMS if sr == SR else STFT_16K_PARAMS
    spec = librosa.stft(wav, **params).T
    return np.abs(spec), np.angle(spec)


def istft_routine(mag, phase, sr):
    params = STFT_44K_PARAMS if sr == SR else STFT_16K_PARAMS
    spec = (mag * np.exp(phase * 1j)).T
    return librosa.istft(spec, **params)
