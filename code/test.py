from conf import *

import tensorflow as tf
import numpy as np
import sys
import mir_eval
import soundfile

if __name__ == "__main__":
    no_gpu = False
    for arg in sys.argv[1:]:
        if arg == "-no-gpu":
            no_gpu = True
    if no_gpu:
        tf.config.set_visible_devices([], 'GPU')

    data_files = [os.path.join(TEST_DIR, np_f) for np_f in os.listdir(TEST_DIR)]

    def avg_mse(y_true, y_pred):
        return (y_true - y_pred) * (y_true - y_pred) / tf.cast(tf.shape(y_true)[1], tf.float32)
    ckpt_folder = os.path.join(MODEL_DIR, "checkpoint")
    last_model_path = os.listdir(ckpt_folder)[-1]
    model = tf.keras.models.load_model(os.path.join(ckpt_folder, last_model_path),
                                       custom_objects={"avg_mse": avg_mse})

    for i, path in enumerate(data_files):
        print(f"{i}-th input, path: {path}")

        data = np.load(path)

        input_mag = np.expand_dims(np.hstack((data["input_mag"], data["ref_mag"])), axis=0)
        res_mag = np.squeeze(model(input_mag, training=False).numpy(), axis=0)
        print(f"MSE value of spectrogram: {np.sum((res_mag - data['truth_mag']) ** 2) / res_mag.shape[0]}")

        result_wav = istft_routine(res_mag, data["input_phase"]).copy()
        truth_wav = data["truth"]
        result_wav.resize(truth_wav.shape)

        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
            np.vstack((truth_wav, data["input"] * 2 - data["truth"])),
            np.vstack((result_wav, data["input"] * 2 - result_wav))
        )
        print(f"SDR: {sdr[0]}, SIR: {sir[0]}, SAR: {sar[0]}")

        output_dir = os.path.join(MODEL_DIR, "test_output")
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass
        soundfile.write(os.path.join(output_dir, os.path.splitext(os.path.basename(path))[0] + ".wav"),
                        result_wav,
                        SR
                        )


