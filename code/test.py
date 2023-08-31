from conf import *

import tensorflow as tf
import numpy as np
import sys
import shutil
import soundfile
import time
import baseline_model


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
        input_seg = np.zeros((WAVE_U_NET_INPUT_LENGTH,), np.float32)
        ref_seg = np.zeros((WAVE_U_NET_INPUT_LENGTH,), np.float32)

        if i < left_margin:
            input_seg[left_margin - i: left_margin] = input_wav[0:i]
            ref_seg[left_margin - i: left_margin] = ref_wav[0:i]
        else:
            input_seg[0:left_margin] = input_wav[i - left_margin:i]
            ref_seg[0:left_margin] = ref_wav[i - left_margin:i]

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

        result_seg = np.squeeze(model((input_seg, ref_seg), training=False))
        if i + valid_frames <= input_mag.shape[0]:
            ans_mag[i: i + valid_frames, :] = result_seg[padding_edge: padding_edge + valid_frames, :]
        else:
            ans_mag[i:] = result_seg[padding_edge: padding_edge + input_mag.shape[0] - i, :]

    result_wav = istft_routine(ans_mag, data["input_phase_8k"], 8192)
    result_wav = librosa.resample(result_wav, orig_sr=8192, target_sr=44100)
    return result_wav


def inference_baseline(model, data):
    input_mag = data["input_mag_16k"]
    ref_mag = data["ref_mag_16k"]
    ans_mask = np.empty(input_mag.shape + (2,), dtype=np.complex64)

    input_frame_length = 313
    frame_hop = 300
    for i in range(0, input_mag.shape[0], frame_hop):
        def fetch_segment(src_mag):
            seg = np.zeros((313, 257), dtype=src_mag.dtype)
            seg[0: input_mag.shape[0] - i, :] = src_mag[i:i + input_frame_length, :]
            return np.expand_dims(seg, 0)
        input_seg = fetch_segment(input_mag)
        ref_seg = fetch_segment(ref_mag)
        mask = model((input_seg, ref_seg), training=False)[0, :, :, :]
        ans_mask[i:i+frame_hop, :, :] = mask[0: min(input_mag.shape[0] - i, frame_hop), :, :]

    input_spec = input_mag * data["input_phase_16k"]
    result_wav = istft_routine_with_spec(baseline_model.result_from_mask(input_spec, ans_mask).T, 16000)
    result_wav = librosa.resample(result_wav, orig_sr=16000, target_sr=SR)
    return result_wav


def inference(target: str, model, data):
    if target == "original":
        return inference_original(model, data)
    elif target == "baseline":
        return inference_baseline(model, data)
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
    assert ADD_NOISE, "Now test routine is rewrite and no longer support no noise case."

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

    # test_data_dir = os.path.join(TEST_DIR, "regular" if not target.startswith("u-net") else "u_net")
    test_data_dir = TEST_DIR
    data_files = [os.path.join(test_data_dir, np_f) for np_f in os.listdir(test_data_dir)]

    # ckpt_folder = os.path.join(MODEL_DIRS[target], "checkpoint")
    # last_model_name = sorted(os.listdir(ckpt_folder))[-1]
    # model_path = os.path.join(ckpt_folder, last_model_name)
    model_path = os.path.join(MODEL_DIRS[target], "trained_model")
    print(f"Model path: {model_path}")
    model = tf.keras.models.load_model(model_path)

    output_dir = os.path.join(MODEL_DIRS[target], "test_output")
    try:
        shutil.rmtree(output_dir)
    except FileNotFoundError:
        pass
    os.mkdir(output_dir)

    algo_sdrs = []
    process_times = []
    for i, path in enumerate(data_files):
        print()
        i = f"{i:06}"
        print(f"{i}-th input:")

        data = dict(np.load(path))
        print(f"Leakage audio path: {data['leak_path']}")
        print(f"Target audio path: {data['truth_path']}")
        print(f"Starting point: {data['starting_seconds']}")

        sub_output_dir = os.path.join(output_dir, i)
        try:
            shutil.rmtree(sub_output_dir)
        except FileNotFoundError:
            pass
        os.mkdir(sub_output_dir)
        soundfile.write(os.path.join(sub_output_dir, i + "_ref.wav"), data["ref"], SR)
        soundfile.write(os.path.join(sub_output_dir, i + "_truth.wav"), data["truth"], SR)

        truth_wav = data["truth"]
        remainder_wav = data["leak"]

        def get_metric(wav):
            if abs(wav.size - truth_wav.size) > 10000:
                raise ValueError("Audio lengths do not match.")
            if wav.size > truth_wav.size:
                wav.resize(truth_wav.shape)
            elif wav.size < truth_wav.size:
                wav = np.pad(wav, (0, truth_wav.size - wav.size), "constant", constant_values=0)
            return get_si_sdr(wav, truth_wav)

        si_sdr_matrix = []
        for snr in TEST_SNRS:
            si_sdrs = []
            for noise_snr in TEST_NOISE_SNRS:
                def prepare_data(sr, sr_str):
                    data["input" + sr_str], _, _ = mix_on_given_snr(snr, data["truth" + sr_str], data["leak" + sr_str])
                    data["input" + sr_str], _, _ = \
                        mix_on_given_snr(noise_snr, data["input" + sr_str], data["noise" + sr_str])

                def prepare_spectrogram(sr, sr_str):
                    for t in ["ref", "input"]:
                        mag, phase = stft_routine(data[t + sr_str], sr)
                        data[t + "_mag" + sr_str] = mag
                        if t == "input":
                            data[t + "_phase" + sr_str] = phase

                if target == "original":
                    prepare_data(SR, "")
                elif target == "baseline":
                    prepare_data(16000, "_16k")
                elif target == "u-net":
                    prepare_data(8192, "_8k")
                elif target == "wave-u-net":
                    prepare_data(SR, "")

                tic = time.perf_counter()

                if target == "original":
                    prepare_spectrogram(SR, "")
                elif target == "baseline":
                    prepare_spectrogram(16000, "_16k")
                elif target == "u-net":
                    prepare_spectrogram(8192, "_8k")
                result_wav = inference(target, model, data)

                toc = time.perf_counter()
                process_times.append(toc - tic)

                sdr = get_metric(result_wav)
                si_sdrs.append(sdr)

                soundfile.write(os.path.join(sub_output_dir,  f"SNR{snr}_NoiseSNR{noise_snr}_output.wav"),
                                result_wav, SR)
                input_file_name = os.path.join(sub_output_dir, f"SNR{snr}_NoiseSNR{noise_snr}_input.wav")
                if target == "original" or target == "wave-u-net":
                    soundfile.write(input_file_name, data["input"], SR)
                elif target == "baseline":
                    soundfile.write(input_file_name, data["input_16k"], 16000)
                elif target == "u-net":
                    soundfile.write(input_file_name, data["input_8k"], 8192)

            si_sdr_matrix.append(si_sdrs)

        algo_sdrs.append(si_sdr_matrix)
        # print(f"Result metrics: SI-SDR: {si_sdr:.5f}")

        # output_dir = os.path.join(output_dir, i)
        # try:
        #     shutil.rmtree(output_dir)
        # except FileNotFoundError:
        #     pass
        # os.mkdir(output_dir)

    # print(f"Average SI-SDR of tested algo: {np.mean(algo_sdrs)}")
    algo_sdrs = np.array(algo_sdrs)
    np.save(os.path.join(output_dir, "SDRMatrix.npy"), algo_sdrs)
    process_times = np.array(process_times)

    print(f"Finished, average SI-SDR of all cases: {np.mean(algo_sdrs)}, result shape: {algo_sdrs.shape}")
    print(f"Average process time for each instance: {np.mean(process_times)}, "
          f"variance of time usage: {np.var(process_times)}")
