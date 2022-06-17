import tensorflow as tf


def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(1000, return_sequences=True, input_shape=(None, 2049*2)))
    model.add(tf.keras.layers.GRU(1000, return_sequences=True))
    model.add(tf.keras.layers.Dense(2049))

    return model


if __name__ == "__main__":
    model = get_model()
    model.summary()
