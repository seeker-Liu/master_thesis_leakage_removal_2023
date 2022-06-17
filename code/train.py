from conf import *

import tf_dataset
import tf_model
import tensorflow as tf

if __name__ == "__main__":
    def avg_mse(y_true, y_pred):
        return (y_true - y_pred) * (y_true - y_pred) / tf.cast(tf.shape(y_true)[1], tf.float32)

    model = tf_model.get_model()
    train_dataset = tf_dataset.get_dataset("train", True)
    valid_dataset = tf_dataset.get_dataset("validation", True)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
    model.compile(optimizer=optimizer,
                  metrics=[avg_mse],
                  loss="mse")
    history = model.fit(train_dataset, epochs=100, validation_data=valid_dataset,
                        callbacks=(tf.keras.callbacks.TerminateOnNaN(),
                                   tf.keras.callbacks.EarlyStopping()
                                   ))

    model.save(os.path.join(MODEL_DIR, "trained_model"))
    print(history.history)
    np.save(os.path.join(MODEL_DIR, "history.npy"), history.history)
    # To load: history = np.load('history.npy', allow_pickle='TRUE').item()
