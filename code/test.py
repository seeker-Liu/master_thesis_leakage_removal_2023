from conf import *

import tensorflow as tf
import numpy as np
import sys
import shutil
import mir_eval
import soundfile

if __name__ == "__main__":
    no_gpu = False
    for arg in sys.argv[1:]:
        if arg == "-no-gpu":
            no_gpu = True
    if no_gpu:
        tf.config.set_visible_devices([], 'GPU')
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    data_files = [os.path.join(TEST_DIR, np_f) for np_f in os.listdir(TEST_DIR)]

    ckpt_folder = os.path.join(MODEL_DIR, "checkpoint")
    last_model_name = sorted(os.listdir(ckpt_folder))[-1]
    model_path = os.path.join(ckpt_folder, last_model_name)
    print(f"Model path: {model_path}")
    model = tf.keras.models.load_model(model_path)

    baseline_sdrs = []
    algo_sdrs = []
    for i, path in enumerate(data_files):
        print()
        print(f"{i}-th input:")
        data_no = os.path.splitext(os.path.basename(path))[0]

        data = np.load(path)
        print(f"Leakage audio path: {data['leak_path']}")
        print(f"Target audio path: {data['truth_path']}")
        print(f"Starting point: {data['starting_seconds']}")

        input_mag = np.expand_dims(data["input_mag"], axis=0)
        ref_mag = np.expand_dims(data["ref_mag"], axis=0)
        res_mag = np.squeeze(model((input_mag, ref_mag), training=False).numpy(), axis=0)

        result_wav = istft_routine(res_mag, data["input_phase"], SR).copy()
        truth_wav = data["truth"]
        leak_wav = data["leak"]
        result_wav.resize(truth_wav.shape)

        def get_metric(wav):
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                np.vstack((truth_wav, leak_wav)),
                np.vstack((wav, data["input"] * 2 - wav))
            )
            return sdr[0], sir[0], sar[0]
        sdr, sir, sar = get_metric(result_wav)
        algo_sdrs.append(sdr)
        print(f"Result metrics: SDR: {sdr:.5f}, SIR: {sir:.5f}, SAR: {sar:.5f}")
        baseline_wav = istft_routine(data["input_mag"] * 2 - data["ref_mag"], data["input_phase"], SR).copy()
        baseline_wav.resize(truth_wav.shape)
        sdr, sir, sar = get_metric(baseline_wav)
        print(f"Baseline metrics: SDR: {sdr:.5f}, SIR: {sir:.5f}, SAR: {sar:.5f}")
        baseline_sdrs.append(sdr)

        output_dir = os.path.join(MODEL_DIR, "test_output")
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass
        output_dir = os.path.join(output_dir, data_no)
        try:
            shutil.rmtree(output_dir)
        except FileNotFoundError:
            pass
        os.mkdir(output_dir)
        soundfile.write(os.path.join(output_dir, data_no + "_output.wav"),
                        result_wav, SR)
        soundfile.write(os.path.join(output_dir, data_no + "_input.wav"),
                        data["input"], SR)
        soundfile.write(os.path.join(output_dir, data_no + "_ref.wav"),
                        data["ref"], SR)
        soundfile.write(os.path.join(output_dir, data_no + "_truth.wav"),
                        data["truth"], SR)
        soundfile.write(os.path.join(output_dir, data_no + "_baseline.wav"),
                        baseline_wav, SR)

    print(f"Average SDR: proposed algo: {np.mean(algo_sdrs)}, baseline: {np.mean(baseline_sdrs)}")


