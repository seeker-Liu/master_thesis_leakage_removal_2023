import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from conf import *

if __name__ == "__main__":
    for k, v in MODEL_DIRS.items():
        if k.endswith("-baseline"):
            continue

        sdr_matrix = np.load(os.path.join(v, "test_output", "SDRMatrix.npy"))
        sdr_matrix = np.mean(sdr_matrix, axis=0)
        ax = sns.heatmap(sdr_matrix, vmin=-25, vmax=10, annot=True, fmt=".1f",
                         xticklabels=TEST_NOISE_SNRS, yticklabels=TEST_SNRS)
        plt.xlabel("Mixture input vs. Noise SNR")
        plt.ylabel("Violin vs. Non-violin SNR")
        if k == "original":
            title_str = "RNN"
        elif k == "baseline":
            title_str = "FullSubNet"
        else:
            title_str = k
        plt.title(title_str + " model")
        plt.savefig(os.path.join(ROOT_DIR, k + "_heatmap.jpg"))
        plt.close()
        print(k, sdr_matrix.mean())
