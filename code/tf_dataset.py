from conf import *

import tensorflow as tf
import numpy as np


def get_dataset(t: str, use_spectrogram):
    """
    :param t:  "train", "validation" or "test'
    :param use_spectrogram: Use spectrogram or not.
    :return: Required datasets.
    """
    dataset = tf.data.Dataset.list_files(os.path.join(DATA_DIR, t, "*.npz"))
    dataset = dataset.shuffle(buffer_size=10000)

    if use_spectrogram:
        def dataset_mapper(filepath):
            data = np.load(filepath)
            return data["input_mag"], data["ref_mag"], data["truth_mag"]
    else:
        def dataset_mapper(filepath):
            data = np.load(filepath)
            return data["input"], data["ref"], data["truth"]

    def dataset_mapper_wrapper(filepath):
        x1, x2, y = tf.numpy_function(dataset_mapper, [filepath], (tf.float32, tf.float32, tf.float32))
        return (x1, x2), y

    dataset = dataset.map(dataset_mapper_wrapper).batch(1)
    return dataset


if __name__ == "__main__":
    ds = get_dataset("train", True)
    i = iter(ds)
    d = next(i)
    print(d)
