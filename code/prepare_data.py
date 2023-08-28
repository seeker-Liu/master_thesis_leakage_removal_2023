from conf import *

import os
import sys
import shutil
import random
import numpy as np
import scipy.signal
import pretty_midi
import librosa


def get_mango_list() -> list:
    ans = []
    mango_dir = os.path.join(DATASET_DIR, "MangoFuture", "midi")
    for piece in os.listdir(mango_dir):
        piece_path = os.path.join(mango_dir, piece)
        if os.path.isdir(piece_path):
            truth_midi_path = os.path.join(piece_path, "独奏乐谱_violin_0.midi")
            leak_midi_path = os.path.join(piece_path, "伴奏乐谱_piano_0.midi")
            ans.append({"truth": (truth_midi_path, 0, VIOLIN_PROGRAM_NUM),
                        "leak": (leak_midi_path, 1, PIANO_PROGRAM_NUM)})

    return ans


def get_sync_list() -> list:
    ans = []
    bach10_dir = os.path.join(DATASET_DIR, "Bach10_v1.1")
    for root, dirs, files in os.walk(bach10_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".mid":
                midi_path = os.path.join(root, file)
                ans.append({"truth": (midi_path, 0, VIOLIN_PROGRAM_NUM), "leak": (midi_path, 1, CLARINET_PROGRAM_NUM)})

    urmp_dir = os.path.join(DATASET_DIR, "URMP")
    for piece in os.listdir(urmp_dir):
        if os.path.isdir(os.path.join(urmp_dir, piece)):
            try:
                piece_num = int(piece.split('_')[0])
            except ValueError:
                continue
            if piece_num in URMP_VIOLIN_CLARINET_PIECES:
                vn_c, cl_c = URMP_VIOLIN_CLARINET_PIECES[piece_num]
                midi_path = os.path.join(urmp_dir, piece, f"Sco_{piece}.mid")
                ans.append({"truth": (midi_path, vn_c, VIOLIN_PROGRAM_NUM),
                            "leak": (midi_path, cl_c, CLARINET_PROGRAM_NUM)})
            if piece_num in URMP_VIOLIN_FLUTE_PIECE:
                vn_c, fl_c = URMP_VIOLIN_FLUTE_PIECE[piece_num]
                midi_path = os.path.join(urmp_dir, piece, f"Sco_{piece}.mid")
                ans.append({"truth": (midi_path, vn_c, VIOLIN_PROGRAM_NUM),
                            "leak": (midi_path, fl_c, FLUTE_PROGRAM_NUM)})

    return ans + get_mango_list()


def get_test_list() -> list:
    ans = []
    bach10_dir = os.path.join(DATASET_DIR, "Bach10_v1.1")
    for piece in os.listdir(bach10_dir):
        if os.path.isdir(os.path.join(bach10_dir, piece)):
            if piece[0] != "0" and piece[0] != "1":
                continue
            violin_path = os.path.join(bach10_dir, piece, piece + "-violin.wav")
            clarinet_path = os.path.join(bach10_dir, piece, piece + "-clarinet.wav")
            ans.append({"truth": (violin_path, None, None), "leak": (clarinet_path, None, None)})

    urmp_dir = os.path.join(DATASET_DIR, "URMP")
    for piece in os.listdir(urmp_dir):
        if os.path.isdir(os.path.join(urmp_dir, piece)):
            try:
                piece_num = int(piece.split('_')[0])
            except ValueError:
                continue
            piece_name = piece.split('_')[1]
            if piece_num in URMP_VIOLIN_CLARINET_PIECES:
                vn_c, cl_c = URMP_VIOLIN_CLARINET_PIECES[piece_num]
                violin_path = os.path.join(urmp_dir, piece, f"AuSep_{vn_c + 1}_vn_{piece_num:02}_{piece_name}.wav")
                clarinet_path = os.path.join(urmp_dir, piece, f"AuSep_{cl_c + 1}_cl_{piece_num:02}_{piece_name}.wav")
                ans.append({"truth": (violin_path, None, None), "leak": (clarinet_path, None, None)})
            if piece_num in URMP_VIOLIN_FLUTE_PIECE:
                vn_c, fl_c = URMP_VIOLIN_FLUTE_PIECE[piece_num]
                violin_path = os.path.join(urmp_dir, piece, f"AuSep_{vn_c + 1}_vn_{piece_num:02}_{piece_name}.wav")
                flute_path = os.path.join(urmp_dir, piece, f"AuSep_{fl_c + 1}_fl_{piece_num:02}_{piece_name}.wav")
                ans.append({"truth": (violin_path, None, None), "leak": (flute_path, None, None)})

    return ans


def get_data_list() -> dict[str: list]:
    sync_list = get_sync_list()
    random.shuffle(sync_list)
    return {"train": sync_list[len(sync_list) // 10:],
            "valid": sync_list[:len(sync_list) // 10],
            "test": get_test_list()}


def get_ir_list() -> list:
    ans = []
    aec_challenge_dir = os.path.join(DATASET_DIR, "ACE_challenge", "Mobile")
    for place in os.listdir(aec_challenge_dir):
        data_dir = os.path.join(aec_challenge_dir, place, "1")
        data_path = os.path.join(data_dir, os.listdir(data_dir)[0])
        ans.append(data_path)

    return ans


def get_noise_list() -> list:
    ans = []
    openslr_noise_dir = os.path.join(DATASET_DIR, "OpenSLR-RIR-Noises", "pointsource_noises")
    with open(os.path.join(openslr_noise_dir, "ANNOTATIONS"), "r", encoding="utf-8") as notation_f:
        for line in notation_f.readlines()[1:]:
            ans.append(os.path.join(openslr_noise_dir, line.rstrip('\n') + ".wav"))
    return ans


def get_random_snr():
    snr_values = [-6, -3, 0, 3, 6]
    return random.choice(snr_values)


def get_random_noise_snr():
    return random.random() * 18 - 3  # [-3, 15] dB


def shake_midi(midi_channel: pretty_midi.Instrument):
    # Shake the onsets of notes to simulate unstable performance from learners.
    # TODO
    return midi_channel


def sync_midi(midi_path, channel, program=None, /, *, sf_path=None, onset_shake=True):
    # Channel is 0-indexed.
    midi = pretty_midi.PrettyMIDI(midi_file=midi_path)
    channel = midi.instruments[channel]
    if onset_shake:
        channel = shake_midi(channel)
    if program:
        channel.program = program
    if DEBUG:
        midi.instruments = [channel]
        midi.write(os.path.join(DATA_DIR, "test.mid"))
    return np.array(channel.fluidsynth(fs=SR, sf2_path=sf_path) / 32768.0, dtype=np.float32)


def load_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=SR, mono=True)
    return audio


def add_reverb(audio, ir):
    audio = scipy.signal.fftconvolve(audio, ir)
    if np.max(np.abs(audio)) >= 1:
        audio /= np.max(np.abs(audio))
    return audio



def sync_audio(data_type: str,
               src_info: dict[str: tuple[str, int, int]],
               clip_length: float = AUDIO_CLIP_LENGTH,
               clip_hop: float = AUDIO_CLIP_HOP,
               sync_for_u_net: bool = False) -> list[dict[str: np.array]]:
    temp = {}
    for t, src in src_info.items():
        # t == "truth" or "leak"
        path, channel, program = src
        if os.path.splitext(path)[1].startswith(".mid"):
            temp[t] = sync_midi(path, channel, program)
        else:
            temp[t] = load_audio(path)
        temp[t + "_path"] = path

    ir = random.choice(IRs)

    # Defensive code for the situation that both lengths do not match
    if len(temp["leak"]) < len(temp["truth"]):
        temp["leak"] = grow_array(temp["leak"], temp["truth"])
    else:
        temp["truth"] = grow_array(temp["truth"], temp["leak"])

    temp["truth"] = add_reverb(temp["truth"], ir)
    temp["leak_convoluted"] = add_reverb(temp["leak"], ir)
    temp["leak"] = grow_array(temp["leak"], temp["truth"])
    # leak is not conv'ed so is slightly shorter, compensate that.

    if sync_for_u_net:
        temp["truth_8k"] = librosa.resample(temp["truth"], orig_sr=SR, target_sr=8192)
        temp["leak_8k"] = librosa.resample(temp["leak"], orig_sr=SR, target_sr=8192)
        temp["leak_convoluted_8k"] = librosa.resample(temp["leak_convoluted"], orig_sr=SR, target_sr=8192)
    elif SAVE_16K:
        temp["truth_16k"] = librosa.resample(temp["truth"], orig_sr=SR, target_sr=16000)
        temp["leak_16k"] = librosa.resample(temp["leak"], orig_sr=SR, target_sr=16000)
        temp["leak_convoluted_16k"] = librosa.resample(temp["leak_convoluted"], orig_sr=SR, target_sr=16000)

    answers = []
    if data_type == "test":
        # Save whole audio in one.
        ans = {"truth_path": temp["truth_path"], "leak_path": temp["leak_path"], "starting_seconds": 0,
               "snr": get_random_snr()}
        if ADD_NOISE:
            ans["noise"] = random.choice(noises).copy()
            if sync_for_u_net:
                ans["noise_8k"] = librosa.resample(ans["noise"], orig_sr=SR, target_sr=8192)
            elif SAVE_16K:
                ans["noise_16k"] = librosa.resample(ans["noise"], orig_sr=SR, target_sr=16000)
            ans["noise_snr"] = get_random_noise_snr()

        def save_content(sr, sr_str, save_spectrogram=SAVE_SPECTROGRAM):
            ans["truth" + sr_str] = temp["truth" + sr_str].copy()
            ans["ref" + sr_str] = temp["leak" + sr_str].copy()
            ans["leak" + sr_str] = temp["leak_convoluted" + sr_str].copy()
            ans["input" + sr_str], _, _ = mix_on_given_snr(ans["snr"], ans["truth" + sr_str], ans["leak" + sr_str])
            if ADD_NOISE:
                if ans["noise" + sr_str].size < ans["input" + sr_str].size:
                    # If the noise is not long enough we repeat it.
                    rep_times = ans["input" + sr_str].size // ans["noise" + sr_str].size + 1
                    ans["noise" + sr_str] = np.tile(ans["noise" + sr_str], rep_times)
                # Then if noise is longer, chop it
                ans["noise" + sr_str] = np.resize(ans["noise" + sr_str], ans["input" + sr_str].shape)

                ans["input" + sr_str], _, _ = \
                    mix_on_given_snr(ans["noise_snr"], ans["input" + sr_str], ans["noise" + sr_str])

            if save_spectrogram:
                for t in ["truth", "ref", "input"]:
                    ans[t + "_mag" + sr_str], ans[t + "_phase" + sr_str] = stft_routine(ans[t + sr_str], sr)

        if sync_for_u_net:
            save_content(8192, "_8k")
            save_content(SR, "", False)
        else:
            save_content(SR, "")
            if SAVE_16K:
                save_content(16000, "_16k")
        answers.append(ans)

    else:
        # for i in range(0, temp["truth"].size // SR - AUDIO_CLIP_HOP, AUDIO_CLIP_HOP):
        i = 0
        while i < temp["truth"].size / SR - clip_hop:
            # Public information shared among different sample rate and stft configs.
            ans = {"truth_path": temp["truth_path"], "leak_path": temp["leak_path"], "starting_seconds": i,
                   "snr": get_random_snr()}
            if ADD_NOISE:
                noise = random.choice(noises)
                # Randomly select a continuous fragment
                noise_fragment_index = random.randrange(0, len(noise) - int(clip_length * SR))
                ans["noise"] = noise[noise_fragment_index: noise_fragment_index + int(clip_length * SR)].copy()
                if sync_for_u_net:
                    ans["noise_8k"] = librosa.resample(ans["noise"], orig_sr=SR, target_sr=8192)
                if SAVE_16K:
                    ans["noise_16k"] = librosa.resample(ans["noise"], orig_sr=SR, target_sr=16000)
                ans["noise_snr"] = get_random_noise_snr()

            def save_content(sr, sr_str):
                index_range = slice(int(i * sr), int((i + clip_length) * sr))
                ans["truth" + sr_str] = temp["truth" + sr_str][index_range].copy()
                ans["ref" + sr_str] = temp["leak" + sr_str][index_range].copy()
                ans["leak" + sr_str] = temp["leak_convoluted" + sr_str][index_range].copy()

                # Zero-padding if audio clip is shorter than AUDIO_CLIP_LENGTH seconds
                ans["truth" + sr_str].resize((int(sr * clip_length), ), refcheck=False)
                ans["ref" + sr_str].resize((int(sr * clip_length), ), refcheck=False)
                ans["leak" + sr_str].resize((int(sr * clip_length), ), refcheck=False)

                ans["input" + sr_str], _, _ = mix_on_given_snr(ans["snr"], ans["truth" + sr_str], ans["leak" + sr_str])
                if ADD_NOISE:
                    ans["input" + sr_str], _, _ = \
                        mix_on_given_snr(ans["noise_snr"], ans["input" + sr_str], ans["noise" + sr_str])

                if SAVE_SPECTROGRAM:
                    for t in ["truth", "ref", "input"]:
                        ans[t + "_mag" + sr_str], ans[t + "_phase" + sr_str] = stft_routine(ans[t + sr_str], sr)

            if sync_for_u_net:
                save_content(8192, "_8k")
            else:
                save_content(SR, "")
                if SAVE_16K:
                    save_content(16000, "_16k")
            answers.append(ans)

            i += clip_hop

    return answers


if __name__ == '__main__':
    TYPES = ["train", "valid", "test"]
    DIRS = [TRAIN_DIR, VALID_DIR, TEST_DIR]
    DIRS = {x: y for x, y in zip(TYPES, DIRS)}

    DEBUG = False
    for_u_net = False
    for_regular = True
    targets = []
    for arg in sys.argv[1:]:
        if arg == "--debug":
            DEBUG = True
        elif arg == "--all":
            targets += TYPES
        elif arg == "--train":
            targets.append("train")
        elif arg == "--valid":
            targets.append("valid")
        elif arg == "--test":
            targets.append("test")
        elif arg == "--u-net":
            for_u_net = True
        elif arg == "--no-regular":
            for_regular = False
        else:
            print(f"Unknown option: {arg}")
    DIRS = {k: v for k, v in DIRS.items() if k in targets}
    if not for_regular and not for_u_net:
        raise ValueError("At least do something?")

    del targets

    # Clean dirs
    for d in DIRS.values():
        # Clean up regular content
        try:
            os.mkdir(d)
        except FileExistsError:
            pass

        def clean_dir(path):
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass
            os.mkdir(path)
        if for_regular:
            regular_dir = os.path.join(d, "regular")
            clean_dir(regular_dir)
        if for_u_net:
            clean_dir(os.path.join(d, "u_net"))

    # Set-up IRs
    print("Setting up IRs' data")
    IRs = [load_audio(ir) for ir in get_ir_list()]

    # Set-up noises
    print("Setting up noises' data")
    noises = [load_audio(noise) for noise in get_noise_list()]
    noises = [noise for noise in noises if len(noise) >= 15 * SR]  # Get long enough noise

    # Actually synth
    data_lists = get_data_list()
    for t in DIRS.keys():
        print(f"Preparing {t} data")
        # Save two types of data

        def sync_and_save(target_dir, data_list, sync_audio_params):
            j = 0
            for i, info in enumerate(data_list):
                if i % 10 == 0:
                    print(f"{i} / {len(data_list)}")
                audios = sync_audio(t, info, **sync_audio_params)
                for audio in audios:
                    out_path = os.path.join(target_dir, f"{j:06}.npz")
                    j += 1
                    np.savez(out_path, **audio)

        if for_regular:
            print("Regular part.")
            sync_and_save(os.path.join(DIRS[t], "regular"), data_lists[t], {})
        if for_u_net:
            print("Special for u-net.")
            sync_and_save(os.path.join(DIRS[t], "u_net"), data_lists[t],
                          {"clip_length": 12.1, "clip_hop": 12.1/2, "sync_for_u_net": True})

