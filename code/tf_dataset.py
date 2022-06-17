from conf import *

import tensorflow as tf
import numpy as np


def get_dataset(t: str, use_spectrogram):
    """
    :param t:  "train", "validation" or "test'
    :param use_spectrogram: Use spectrogram or not.
    :param epoch:  Positive int or None, None means repeat indefinitely
    :return: Required datasets.
    """
    dataset = tf.data.Dataset.list_files(os.path.join(DATA_DIR, t, "*.npz"))
    dataset = dataset.shuffle(buffer_size=10000)

    if use_spectrogram:
        def dataset_mapper(filepath):
            data = np.load(filepath)
            return np.hstack((data["input_mag"], data["ref_mag"])), data["truth_mag"]
    else:
        def dataset_mapper(filepath):
            data = np.load(filepath)
            return np.vstack((data["input"], data["ref"])), data["truth"]

    def dataset_mapper_wrapper(filepath):
        return tf.numpy_function(dataset_mapper, [filepath], (tf.float32, tf.float32))

    dataset = dataset.map(dataset_mapper_wrapper).batch(1)
    return dataset


if __name__ == "__main__":
    ds = get_dataset("train", True)
    i = iter(ds)
    d = next(i)
    print(d)
