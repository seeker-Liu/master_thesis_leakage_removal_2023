import time
import numpy as np
import librosa

if __name__ == "__main__":
    wav = (np.random.rand(44100 * 10) + 1) / 2
    his = []
    for i in range(5):
        start = time.perf_counter()
        for j in range(100):
            spec = librosa.stft(wav, n_fft=4096, hop_length=1024, window="hann", center=True).T
            mag, pha = np.abs(spec), np.angle(spec)
        his.append(time.perf_counter() - start)

    print(his)
    print(f"Average: {np.mean(his)}")

