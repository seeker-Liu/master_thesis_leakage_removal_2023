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
U_NET_MODEL_DIR = os.path.join(ROOT_DIR, "u_net_model")
U_NET_BASELINE_MODEL_DIR = os.path.join(ROOT_DIR, "u_net_baseline_model")

MODEL_DIRS = {
    "original": MODEL_DIR,
    "baseline": BASELINE_MODEL_DIR,
    "wave-u-net": WAVE_U_NET_MODEL_DIR,
    "wave-u-net-baseline": WAVE_U_NET_BASELINE_MODEL_DIR,
    "u-net": U_NET_MODEL_DIR,
    "u-net-baseline": U_NET_BASELINE_MODEL_DIR
}

SR = 44100

SAVE_SPECTROGRAM = True
SAVE_16K = False
ADD_NOISE = True

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

STFT_PARAMS = {16000: {"n_fft": 512, "hop_length": 256, "window": "hann", "center": True},
               44100: {"n_fft": 4096, "hop_length": 1024, "window": "hann", "center": True},
               8192: {"n_fft": 1024, "hop_length": 768, "window": "hann", "center": False}}

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
    },
    "u-net": {
        "use_spectrogram": True,
        "use_irm": False,
        "sr": 8192,
        "sr_postfix_str": "_8k",
        "target_input_length": None,
        "target_output_length": None,
        "batch_size": 16,
        "output_data_mapper":
            lambda x1, x2, y: ((tf.expand_dims(x1, -1), tf.expand_dims(x2, -1)), tf.expand_dims(y, -1))
    },
    "u-net-baseline": {
        "use_spectrogram": True,
        "use_irm": False,
        "sr": 8192,
        "sr_postfix_str": "_8k",
        "target_input_length": None,
        "target_output_length": None,
        "batch_size": 16,
        "output_data_mapper": lambda x1, x2, y: ((tf.expand_dims(x1, -1), ), tf.expand_dims(y, -1))
    },
}


def stft_routine(wav, sr):
    params = STFT_PARAMS[sr]
    spec = librosa.stft(wav, **params).T

    if sr == 8192:  # Cut the last freq bin
        spec = spec[:, 0:512]
    return librosa.magphase(spec)


def istft_routine(mag, phase, sr):
    params = STFT_PARAMS[sr]
    spec = (mag * phase).T

    if sr == 8192:  # Append zero to the cut last freq bin
        temp = np.zeros((513, spec.shape[1]), dtype=np.complex64)
        temp[0:512, :] = spec
        spec = temp
    return librosa.istft(spec, **params)


def grow_array(src, target):
    tmp = np.zeros_like(target)
    tmp[0:src.size] = src
    return tmp


def mix_on_given_snr(snr, signal, noise):
    if signal.size / noise.size > 1.05 or signal.size / noise.size < 0.95:
        raise ValueError("Two input array length mismatch seriously.")
    if signal.size < noise.size:
        signal = grow_array(signal, noise)
    else:
        noise = grow_array(noise, signal)

    signal_energy = np.mean(signal ** 2)
    noise_energy = np.mean(noise ** 2)

    g = np.sqrt(10.0 ** (-snr / 10) * signal_energy / noise_energy)
    a = np.sqrt(1 / (1 + g ** 2))
    b = np.sqrt(g ** 2 / (1 + g ** 2))

    return a * signal + b * noise, a, b


def get_si_sdr(est, ref):
    # as provided by @Jonathan-LeRoux and slightly adapted for the case of just one reference
    # and one estimate.
    # see original code here: https://github.com/sigsep/bsseval/issues/3#issuecomment-494995846
    eps = np.finfo(est[0].dtype).eps
    reference = np.expand_dims(ref, -1)
    estimate = np.expand_dims(est, -1)
    Rss = np.dot(reference.T, reference)

    # get the scaling factor for clean sources
    a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)

    e_true = a * reference
    e_res = estimate - e_true

    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()

    return 10 * np.log10((eps+ Sss)/(eps + Snn))