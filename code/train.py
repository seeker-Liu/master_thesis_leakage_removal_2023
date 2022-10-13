from conf import *

import tf_dataset
import tf_model
import baseline_model
import tensorflow as tf
import sys

if __name__ == "__main__":
    no_gpu = False
    continue_train = False
    train_baseline = False
    for arg in sys.argv[1:]:
        if arg == "-no-gpu":
            no_gpu = True
        if arg == "-c" or arg == "-continue":
            continue_train = True
        if arg == "-b" or arg == "-baseline":
            train_baseline = True
        else:
            print(f"Unknown arg: {arg}")

    if no_gpu:
        tf.config.set_visible_devices([], 'GPU')
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    model_dir = BASELINE_MODEL_DIR if train_baseline else MODEL_DIR
    if continue_train:
        ckpt_folder = os.path.join(model_dir, "checkpoint")
        last_model_path = os.listdir(ckpt_folder)[-1]
        model = tf.keras.models.load_model(os.path.join(ckpt_folder, last_model_path))
    else:
        if train_baseline:
            model = baseline_model.BaselineModel()
            optimizer = tf.keras.optimizers.Adam()
            model.compile(optimizer=optimizer,
                          loss="mse")
        else:
            model = tf_model.get_model()
            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=optimizer,
                          loss="mse")

    try:
        os.mkdir(CHECKPOINT_DIR)
    except FileExistsError:
        pass

    dataset_param = {"use_spectrogram": not train_baseline,
                     "use_irm": train_baseline,
                     "sr": 16000 if train_baseline else SR,
                     "sr_postfix_str": "_16k" if train_baseline else ""}
    train_dataset = tf_dataset.get_dataset("train", **dataset_param)
    valid_dataset = tf_dataset.get_dataset("validation", **dataset_param)

    history = model.fit(train_dataset, epochs=100, validation_data=valid_dataset,
                        callbacks=(tf.keras.callbacks.TerminateOnNaN(),
                                   tf.keras.callbacks.EarlyStopping(patience=3, min_delta=1e-5, verbose=1),
                                   tf.keras.callbacks.ModelCheckpoint(
                                       os.path.join(CHECKPOINT_DIR, "model_{epoch:03d}"),
                                       save_weights_only=False,
                                       save_best_only=False,
                                       verbose=1
                                   ))
                        )

    model.save(os.path.join(model_dir, "trained_model"))
    print(history.history)
    np.save(os.path.join(model_dir, "history.npy"), history.history)
    # To load: history = np.load('history.npy', allow_pickle='TRUE').item()
