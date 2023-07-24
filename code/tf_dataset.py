from conf import *

import tensorflow as tf
from baseline_model import build_ideal_mask
import numpy as np
from typing import Callable


def get_dataset(t: str, use_spectrogram, use_irm, sr, sr_postfix_str, target_output_length, batch_size,
                output_data_mapper: Callable):
    """
    :param t:  "train", "validation" or "test'
    :param use_spectrogram: Use spectrogram or not.
    :param use_irm: Return Ideal Ratio Mask or not. Overwrite use_spectrogram
    :param sr sample rate used
    :param sr_postfix_str Postfix string of sr
    :param target_output_length Only meaningful to Wave-u-net model, wave-u-net model use a context setting and only
    the middle part is kept as prediction.
    :param batch_size
    :param output_data_mapper It is a mapper that maps raw input (spectrogram or waveform without extra dimension) to
    the shape that downstream models need. It should take 3 arguments (mixture, leakage, ground_truth) and
    transform them into proper type and shape.
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
        begin_idx = ((sr * AUDIO_CLIP_LENGTH) - target_output_length) // 2
        end_idx = sr * AUDIO_CLIP_LENGTH - target_output_length - begin_idx

        def dataset_mapper(filepath):
            data = np.load(filepath)
            return data["input" + sr_postfix_str], data["ref" + sr_postfix_str], \
                data["truth" + sr_postfix_str][begin_idx:-end_idx]

    def dataset_mapper_wrapper(filepath):
        x1, x2, y = tf.numpy_function(dataset_mapper, [filepath], (tf.float32, tf.float32, tf.float32))
        return output_data_mapper(x1, x2, y)

    dataset = dataset.map(dataset_mapper_wrapper).batch(batch_size)
    return dataset


if __name__ == "__main__":
    ds = get_dataset("train", True, False, SR, "", None)
    i = iter(ds)
    x, y = next(i)
    x1, x2 = x
    print(x1.shape, x2.shape, y.shape)

    ds = get_dataset("train", False, True, 16000, "_16k", None)
    i = iter(ds)
    x, y = next(i)
    x1, x2 = x
    print(x1.shape, x2.shape, y.shape)
