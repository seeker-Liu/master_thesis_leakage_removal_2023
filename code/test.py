import librosa

from conf import *

import tensorflow as tf
import numpy as np
import sys
import shutil
import mir_eval
import soundfile


def inference_original(model, data):
    input_mag = np.expand_dims(data["input_mag"], axis=0)
    ref_mag = np.expand_dims(data["ref_mag"], axis=0)
    res_mag = np.squeeze(model((input_mag, ref_mag), training=False).numpy(), axis=0)

    result_wav = istft_routine(res_mag, data["input_phase"], SR).copy()
    truth_wav = data["truth"]
    result_wav.resize(truth_wav.shape)
    return result_wav


def inference_wave_u_net(model, data):
    input_wav = data["input"]
    ref_wav = data["ref"]
    out_wav = np.zeros(shape=input_wav.shape)
    assert input_wav.size == ref_wav.size

    left_margin = (WAVE_U_NET_INPUT_LENGTH - WAVE_U_NET_OUTPUT_LENGTH) // 2
    for i in range(0, input_wav.size, WAVE_U_NET_OUTPUT_LENGTH):
        input_seg = np.zeros((WAVE_U_NET_INPUT_LENGTH, ), np.float32)
        ref_seg = np.zeros((WAVE_U_NET_INPUT_LENGTH, ), np.float32)

        if i < left_margin:
            input_seg[left_margin - i: left_margin] = input_wav[0:i]
            ref_seg[left_margin - i: left_margin] = ref_wav[0:i]
        else:
            input_seg[0:left_margin] = input_wav[i-left_margin:i]
            ref_seg[0:left_margin] = ref_wav[i-left_margin:i]

        if WAVE_U_NET_INPUT_LENGTH - left_margin >= input_wav.size - i:
            input_seg[left_margin:left_margin + input_wav.size - i] = input_wav[i:]
            ref_seg[left_margin:left_margin + input_wav.size - i] = ref_wav[i:]
        else:
            input_seg[left_margin:] = input_wav[i: i + WAVE_U_NET_INPUT_LENGTH - left_margin]
            ref_seg[left_margin:] = ref_wav[i: i + WAVE_U_NET_INPUT_LENGTH - left_margin]

        input_seg = np.expand_dims(input_seg, (0, -1))
        ref_seg = np.expand_dims(ref_seg, (0, -1))
        out_seg = model((input_seg, ref_seg), training=False)["target"].numpy()
        out_seg = np.squeeze(out_seg)
        if i + WAVE_U_NET_OUTPUT_LENGTH <= out_wav.size:
            out_wav[i: i + WAVE_U_NET_OUTPUT_LENGTH] = out_seg
        else:
            out_wav[i:] = out_seg[0: out_wav.size - i]

    return out_wav


def inference_u_net(model, data):
    input_mag = data["input_mag_8k"]
    ref_mag = data["ref_mag_8k"]
    ans_mag = np.empty_like(input_mag)

    valid_frames = 64  # We only use the center 64 frames as valid output and discard the boundary.
    padding_edge = (128 - valid_frames) // 2  # Data outside is padded by zero
    for i in range(0, input_mag.shape[0], valid_frames):
        def fetch_segment(src_mag):
            result_mag = np.zeros((128, 512), src_mag.dtype)
            if i != 0:
                result_mag[0:padding_edge, :] = src_mag[i - padding_edge:i, :]
            if i + valid_frames + padding_edge <= src_mag.shape[0]:
                result_mag[padding_edge:, :] = src_mag[i: i + padding_edge + valid_frames, :]
            else:
                result_mag[padding_edge: src_mag.shape[0] - i + padding_edge, :] = src_mag[i:, :]
            return np.expand_dims(result_mag, (0, -1))
        input_seg = fetch_segment(input_mag)
        ref_seg = fetch_segment(ref_mag)

        if i + valid_frames <= input_mag.shape[0]:
            ans_mag[i: i+valid_frames, :] = \
                np.squeeze(model((input_seg, ref_seg)))[padding_edge: padding_edge + valid_frames, :]
        else:
            ans_mag[i:] = np.squeeze(model((input_seg, ref_seg)))[padding_edge: padding_edge + input_mag.shape[0] - i, :]

    result_wav = istft_routine(ans_mag, data["input_phase_8k"], 8192)
    result_wav = librosa.resample(result_wav, orig_sr=8192, target_sr=44100)
    return result_wav


def inference(target: str, model, data):
    """
    return tuples (truth_waveform, leak_waveform, result_waveform)
    """
    if target == "original":
        return inference_original(model, data)
    elif target == "baseline":
        return 0
    elif target == "wave-u-net":
        return inference_wave_u_net(model, data)
    elif target == "wave-u-net-baseline":
        return 0
    elif target == "u-net":
        return inference_u_net(model, data)
    elif target == "u-net-baseline":
        return 0
    else:
        raise ValueError("Unknown target type.")


if __name__ == "__main__":
    no_gpu = False
    target = None
    for arg in sys.argv[1:]:
        if arg == "--no-gpu":
            no_gpu = True
        elif arg == "--original":
            target = "original"
        elif arg == "--baseline":
            target = "baseline"
        elif arg == "--wave-u-net":
            target = "wave-u-net"
        elif arg == "--wave-u-net-baseline":
            target = "wave-u-net-baseline"
        elif arg == "--u-net":
            target = "u-net"
        elif arg == "--u-net-baseline":
            target = "u-net-baseline"
        else:
            print(f"unknown arg {arg}")

    if no_gpu:
        tf.config.set_visible_devices([], 'GPU')
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    test_data_dir = os.path.join(TEST_DIR, "regular" if not target.startswith("u-net") else "u_net")
    data_files = [os.path.join(test_data_dir, np_f) for np_f in os.listdir(test_data_dir)]

    ckpt_folder = os.path.join(MODEL_DIRS[target], "checkpoint")
    last_model_name = sorted(os.listdir(ckpt_folder))[-1]
    model_path = os.path.join(ckpt_folder, last_model_name)
    print(f"Model path: {model_path}")
    model = tf.keras.models.load_model(model_path)

    algo_sdrs = []
    for i, path in enumerate(data_files):
        print()
        print(f"{i}-th input:")
        data_index = os.path.splitext(os.path.basename(path))[0]

        data = np.load(path)
        print(f"Leakage audio path: {data['leak_path']}")
        print(f"Target audio path: {data['truth_path']}")
        print(f"Starting point: {data['starting_seconds']}")

        result_wav = inference(target, model, data)
        truth_wav = data["truth"]
        leak_wav = data["leak"]
        input_wav = data["input"]

        def get_metric(wav):
            if wav.size > input_wav.size:
                wav.resize(input_wav.shape)
            elif wav.size < input_wav.size:
                wav = np.pad(wav, (0, input_wav.size - wav.size), "constant", constant_values=0)
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                np.vstack((truth_wav, leak_wav)),
                np.vstack((wav, input_wav * 2 - wav))
            )
            return sdr[0], sir[0], sar[0]

        sdr, sir, sar = get_metric(result_wav)
        algo_sdrs.append(sdr)
        print(f"Result metrics: SDR: {sdr:.5f}, SIR: {sir:.5f}, SAR: {sar:.5f}")

        output_dir = os.path.join(MODEL_DIRS[target], "test_output")
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass
        output_dir = os.path.join(output_dir, data_index)
        try:
            shutil.rmtree(output_dir)
        except FileNotFoundError:
            pass
        os.mkdir(output_dir)
        soundfile.write(os.path.join(output_dir, data_index + "_output.wav"),
                        result_wav, SR)
        soundfile.write(os.path.join(output_dir, data_index + "_input.wav"),
                        data["input"], SR)
        soundfile.write(os.path.join(output_dir, data_index + "_ref.wav"),
                        data["ref"], SR)
        soundfile.write(os.path.join(output_dir, data_index + "_truth.wav"),
                        data["truth"], SR)

    print(f"Average SDR of tested algo: {np.mean(algo_sdrs)}")
