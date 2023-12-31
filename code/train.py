import shutil

from conf import *

import tf_dataset
import tf_model
import baseline_model
import tensorflow as tf
import sys
from wave_u_net.wave_u_net import wave_u_net
from wave_u_net.wave_u_net_AEC import wave_u_net_aec
from u_net.u_net_baseline import u_net_baseline
from u_net.u_net import u_net

if __name__ == "__main__":
    no_gpu = False
    continue_train = False
    target = "None"  # "original", "baseline", "u-net" or "wave-u-net"
    for arg in sys.argv[1:]:
        if arg == "--no-gpu":
            no_gpu = True
        elif arg == "-c" or arg == "--continue":
            continue_train = True
        elif arg == "--rnn" or arg == "--original":
            target = "original"
        elif arg == "-b" or arg == "--baseline" or arg == "--fullsubnet":
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
            print(f"Unknown arg: {arg}")

    if no_gpu:
        tf.config.set_visible_devices([], 'GPU')
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    try:
        os.mkdir(MODEL_DIR)
    except FileExistsError:
        pass
    model_dir = MODEL_DIRS[target]
    try:
        os.mkdir(model_dir)
    except FileExistsError:
        pass
    ckpt_folder = os.path.join(model_dir, "checkpoint")
    try:
        os.mkdir(ckpt_folder)
    except FileExistsError:
        pass

    if continue_train:
        last_model_folder_name = os.listdir(ckpt_folder)[-1]
        model = tf.keras.models.load_model(os.path.join(ckpt_folder, last_model_folder_name))
    else:
        if target == "baseline":
            model = baseline_model.get_baseline_model()
            optimizer = tf.keras.optimizers.Adam()
            model.compile(optimizer=optimizer,
                          loss="mse")
        elif target == "original":
            model = tf_model.get_model()
            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=optimizer,
                          loss="mse")
        elif target == "wave-u-net" or target == "wave-u-net-baseline":
            wave_u_net_params = {
                "num_initial_filters": 24,
                "num_layers": 12,
                "kernel_size": 15,
                "merge_filter_size": 5,
                "source_names": ["target"],
                "num_channels": 1,
                "output_filter_size": 1,
                "padding": "valid",
                "input_size": WAVE_U_NET_INPUT_LENGTH,
                "context": True,
                "upsampling_type": "linear",  # "learned" or "linear"
                "output_activation": "linear",  # "linear" or "tanh"
                "output_type": "direct",  # "direct" or "difference"
            }
            model = wave_u_net(**wave_u_net_params) \
                if target == "wave-u-net-baseline" else wave_u_net_aec(**wave_u_net_params)
            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=optimizer,
                          loss="mse")
        elif target == "u-net":
            model = u_net()
            model.compile(optimizer="adam", loss="mean_absolute_error")
        elif target == "u-net-baseline":
            model = u_net_baseline()
            model.compile(optimizer="adam", loss="mean_absolute_error")

    shutil.rmtree(ckpt_folder)
    try:
        os.mkdir(ckpt_folder)
    except FileExistsError:
        pass

    dataset_param = DATASET_PARAMS[target]
    train_dataset = tf_dataset.get_dataset("train", **dataset_param)
    valid_dataset = tf_dataset.get_dataset("validation", **dataset_param)

    early_stop_min_delta = 1e-7 if target.startswith("wave-u-net") else 1e-5
    history = model.fit(train_dataset, epochs=100, validation_data=valid_dataset,
                        callbacks=(tf.keras.callbacks.TerminateOnNaN(),
                                   tf.keras.callbacks.EarlyStopping(
                                       patience=2,
                                       min_delta=early_stop_min_delta,
                                       verbose=1),
                                   tf.keras.callbacks.ModelCheckpoint(
                                       os.path.join(ckpt_folder, "model_{epoch:03d}"),
                                       save_weights_only=False,
                                       save_best_only=False,
                                       verbose=1
                                   ))
                        )

    model.save(os.path.join(model_dir, "trained_model"))
    print(history.history)
    np.save(os.path.join(model_dir, "history.npy"), history.history)
    # To load: history = np.load('history.npy', allow_pickle='TRUE').item()
