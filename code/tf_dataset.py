from conf import *

import tensorflow as tf
from baseline_model import build_ideal_mask
import numpy as np


def get_dataset(t: str, use_spectrogram, use_irm, sr, sr_postfix_str):
    """
    :param t:  "train", "validation" or "test'
    :param use_spectrogram: Use spectrogram or not.
    :param use_irm: Return Ideal Ratio Mask or not. Overwrite use_spectrogram
    :param sr sample rate used
    :param sr_postfix_str Postfix string of sr
    :return: Required datasets.
    """
    dataset = tf.data.Dataset.list_files(os.path.join(DATA_DIR, t, "*.npz"))
    dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)

    if use_irm:
        def dataset_mapper(filepath):
            data = np.load(filepath)
            input_mag = data["input_mag" + sr_postfix_str]
            input_phase = data["input_phase" + sr_postfix_str]
            ref_mag = data["ref_mag" + sr_postfix_str]
            truth_mag = data["truth_mag" + sr_postfix_str]
            truth_phase = data["truth_phase" + sr_postfix_str]
            input_complex = input_mag * np.exp(input_phase * 1j)
            truth_complex = truth_mag * np.exp(truth_phase * 1j)

            return input_mag, ref_mag, build_ideal_mask(input_complex, truth_complex)

    elif use_spectrogram:
        def dataset_mapper(filepath):
            data = np.load(filepath)
            if SAVE_SPECTROGRAM:
                return data["input_mag" + sr_postfix_str], \
                       data["ref_mag" + sr_postfix_str], data["truth_mag" + sr_postfix_str]
            else:
                input_mag = stft_routine(data["input" + sr_postfix_str], sr)
                ref_mag = stft_routine(data["ref" + sr_postfix_str], sr)
                truth_mag = stft_routine(data["truth" + sr_postfix_str], sr)
                return input_mag, ref_mag, truth_mag
    else:
        def dataset_mapper(filepath):
            data = np.load(filepath)
            return data["input" + sr_postfix_str], data["ref" + sr_postfix_str], data["truth" + sr_postfix_str]

    def dataset_mapper_wrapper(filepath):
        x1, x2, y = tf.numpy_function(dataset_mapper, [filepath], (tf.float32, tf.float32, tf.float32))
        return (x1, x2), y

    dataset = dataset.map(dataset_mapper_wrapper).batch(BATCH_SIZE)
    return dataset


if __name__ == "__main__":
    ds = get_dataset("train", True, False, SR, "")
    i = iter(ds)
    x, y = next(i)
    x1, x2 = x
    print(x1.shape, x2.shape, y.shape)

    ds = get_dataset("train", False, True, 16000, "_16k")
    i = iter(ds)
    x, y = next(i)
    x1, x2 = x
    print(x1.shape, x2.shape, y.shape)
