import tensorflow as tf


def get_model():
    mixed_spec = tf.keras.layers.Input(shape=(None, 2049), name="mixed_spec")
    ref_spec = tf.keras.layers.Input(shape=(None, 2049), name="ref_spec")
    concated = tf.keras.layers.Concatenate(axis=-1)([mixed_spec, ref_spec])
    d = tf.keras.layers.Dense(512, activation="relu")(concated)
    g1 = tf.keras.layers.GRU(256, return_sequences=True)(d)
    g2 = tf.keras.layers.GRU(256, return_sequences=True)(g1)
    mask = tf.keras.layers.Dense(2049, activation="sigmoid")(g2)
    output = mask * mixed_spec
    model = tf.keras.Model(inputs=[mixed_spec, ref_spec], outputs=output)
    return model


if __name__ == "__main__":
    model = get_model()
    model.summary()
