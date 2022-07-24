from conf import *

import os
import sys
import shutil
import random
import numpy as np
import scipy.signal
import pretty_midi
import librosa
import soundfile


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
    return ans

# These 3 functions gives a list of pairs of midi or audio files.
# Modify them to add/remove/modify data sources.


def get_train_list() -> list:
    ans = get_sync_list()
    ans *= 10
    random.shuffle(ans)
    return ans


def get_valid_list() -> list:
    ans = get_sync_list()
    random.shuffle(ans)
    return ans


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
                violin_path = os.path.join(urmp_dir, piece, f"AuSep_{vn_c+1}_vn_{piece_num:02}_{piece_name}.wav")
                clarinet_path = os.path.join(urmp_dir, piece, f"AuSep_{cl_c+1}_cl_{piece_num:02}_{piece_name}.wav")
                ans.append({"truth": (violin_path, None, None), "leak": (clarinet_path, None, None)})
            if piece_num in URMP_VIOLIN_FLUTE_PIECE:
                vn_c, fl_c = URMP_VIOLIN_FLUTE_PIECE[piece_num]
                violin_path = os.path.join(urmp_dir, piece, f"AuSep_{vn_c+1}_vn_{piece_num:02}_{piece_name}.wav")
                flute_path = os.path.join(urmp_dir, piece, f"AuSep_{fl_c+1}_fl_{piece_num:02}_{piece_name}.wav")
                ans.append({"truth": (violin_path, None, None), "leak": (flute_path, None, None)})

    return ans


def get_ir_list() -> list:
    ans = []
    aec_challenge_dir = os.path.join(DATASET_DIR, "ACE_challenge", "Mobile")
    for place in os.listdir(aec_challenge_dir):
        data_dir = os.path.join(aec_challenge_dir, place, "1")
        data_path = os.path.join(data_dir, os.listdir(data_dir)[0])
        ans.append(data_path)

    return ans


def get_data_list(data_type) -> list:
    if data_type == "train":
        return get_train_list()
    elif data_type == "valid":
        return get_valid_list()
    elif data_type == "test":
        return get_test_list()
    else:
        raise ValueError("Unknown data type")


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


def sync_audio(data_type: str, src_info: dict[str: tuple[str, int]]) -> list[dict[str: np.array]]:
    temp = {}
    ans = {}
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
        temp["leak"].resize(temp["truth"].shape, refcheck=False)
    else:
        temp["truth"].resize(temp["leak"].shape, refcheck=False)

    temp["truth"] = add_reverb(temp["truth"], ir)
    temp["leak_convoluted"] = add_reverb(temp["leak"], ir)
    temp["leak"] = np.resize(temp["leak"], temp["truth"].size)
    # leak is not conv'ed so is slightly shorter, compensate that.

    if SAVE_16K:
        temp["truth_16k"] = librosa.resample(temp["truth"], orig_sr=SR, target_sr=16000)
        temp["leak_16k"] = librosa.resample(temp["leak"], orig_sr=SR, target_sr=16000)
        temp["leak_convoluted_16k"] = librosa.resample(temp["leak_convoluted"], orig_sr=SR, target_sr=16000)

    answers = []
    for i in range(0, temp["truth"].size // SR - AUDIO_CLIP_HOP, AUDIO_CLIP_HOP):
        ans = {"truth_path": temp["truth_path"], "leak_path": temp["leak_path"], "starting_seconds": i}

        def save_content(sr, sr_str):
            index_range = slice(i * sr, (i + AUDIO_CLIP_LENGTH) * sr)
            ans["truth" + sr_str] = temp["truth" + sr_str][index_range].copy()
            ans["ref" + sr_str] = temp["leak" + sr_str][index_range].copy()
            ans["leak" + sr_str] = temp["leak_convoluted" + sr_str][index_range].copy()

            # Zero-padding if audio clip is shorter than AUDIO_CLIP_LENGTH seconds
            ans["truth" + sr_str].resize(sr * AUDIO_CLIP_LENGTH, refcheck=False)
            ans["ref" + sr_str].resize(sr * AUDIO_CLIP_LENGTH, refcheck=False)
            ans["leak" + sr_str].resize(sr * AUDIO_CLIP_LENGTH, refcheck=False)

            ans["input" + sr_str] = (ans["leak" + sr_str] + ans["truth" + sr_str]) * 0.5

            if SAVE_SPECTOGRAM:
                for t in ["truth", "ref", "input"]:
                    ans[t + "_mag" + sr_str], ans[t + "_phase" + sr_str] = stft_routine(ans[t + sr_str], sr)

            if not SAVE_AUDIO:
                ans.pop("truth" + sr_str)
                ans.pop("leak" + sr_str)
                ans.pop("truth" + sr_str)

        save_content(SR, "")
        if SAVE_16K:
            save_content(16000, "_16k")
        answers.append(ans)

    return answers


if __name__ == '__main__':
    TYPES = ["train", "valid", "test"]
    DIRS = [TRAIN_DIR, VALID_DIR, TEST_DIR]
    DIRS = {x: y for x, y in zip(TYPES, DIRS)}

    DEBUG = False
    targets = []
    for arg in sys.argv[1: ]:
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
        else:
            print(f"Unknown option: {arg}")
    DIRS = {k: v for k, v in DIRS.items() if k in targets}
    del targets

    # Clean dirs
    for d in DIRS.values():
        try:
            shutil.rmtree(d)
        except FileNotFoundError:
            pass
        os.mkdir(d)

    # Set-up IRs
    IRs = [load_audio(ir) for ir in get_ir_list()]

    # Actually synth
    for t in DIRS.keys():
        print(f"Preparing {t} data")
        j = 0
        data_list = get_data_list(t)
        for i, info in enumerate(data_list):
            if i % 10 == 0:
                print(f"{i} / {len(data_list)}")
            audios = sync_audio(t, info)
            for audio in audios:
                out_path = os.path.join(DIRS[t], f"{j:06}.npz")
                j += 1
                np.savez(out_path, **audio)
                if DEBUG:
                    for k, v in audio.items():
                        soundfile.write(os.path.join(DATA_DIR, k + "-test.wav"), v, SR)
