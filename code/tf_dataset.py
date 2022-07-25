from conf import *

import tensorflow as tf
import numpy as np


def get_dataset(t: str, use_spectrogram, sr, sr_postfix_str):
    """
    :param t:  "train", "validation" or "test'
    :param use_spectrogram: Use spectrogram or not.
    :param sr sample rate used
    :param sr_postfix_str Postfix string of sr
    :return: Required datasets.
    """
    dataset = tf.data.Dataset.list_files(os.path.join(DATA_DIR, t, "*.npz"))
    dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)

    if use_spectrogram:
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
    ds = get_dataset("train", True, SR, "")
    i = iter(ds)
    d = next(i)
    print(d)
