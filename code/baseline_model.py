from conf import *

import tensorflow as tf


class Unfold(tf.keras.layers.Layer):
    """
    Unfold layer.
    Unfold the input spectrogram to generate sub-band matrix
    """
    def __init__(self, num_neighbor, **kwargs):
        super().__init__(**kwargs)
        self.N = num_neighbor
        self.paddings = tf.constant([[0, 0], [0, 0], [self.N, self.N]])

    def call(self, x, *args, **kwargs):
        """
        :param x: 3D tensor, shape: (Batch_index, Time_index, Freq_index)
        :return: unfolded_tensor: 4D, shape: (Batch, Time, Freq, Freq_band)
        """
        b, t, f = x.get_shape().as_list()
        x = tf.pad(x, self.paddings, mode="reflect")
        # From the paper, correct mode should be "circular"? (not supported by tf now)
        # But the implementation says "reflect" XD
        x = tf.expand_dims(x, -1)
        x = tf.image.extract_patches(x,
                                     sizes=[1, 1, f, 1],
                                     strides=[1, 1, 1, 1],
                                     rates=[1, 1, 1, 1],
                                     padding="VALID")
        x = tf.transpose(x, perm=[0, 1, 3, 2])
        return x


class NormLayer3D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NormLayer3D, self).__init__(**kwargs)

    def call(self, x, *args, **kwargs):
        mu = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
        return x / (mu + 1e-5)


class NormLayer4D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NormLayer4D, self).__init__(**kwargs)

    def call(self, x, *args, **kwargs):
        mu = tf.reduce_mean(x, axis=(1, 2, 3), keepdims=True)
        return x / (mu + 1e-5)


class BaselineModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(BaselineModel, self).__init__(**kwargs)
        self.norm_3d = NormLayer3D()
        self.norm_4d = NormLayer4D()
        self.unfold = Unfold(15)
        self.fb_lstm_1 = tf.keras.layers.LSTM(512, return_sequences=True)
        self.fb_lstm_2 = tf.keras.layers.LSTM(512, return_sequences=True)
        self.fb_dense = tf.keras.layers.Dense(257, activation="relu")
        self.sb_lstm_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.LSTM(384, return_sequences=True))
        self.sb_lstm_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.LSTM(384, return_sequences=True))
        self.sb_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2, activation=None))

    def call(self, inputs, training=False, mask=None):
        mag, ref_mag = inputs
        # Pad 2 frames of zeros at the end for 2-frame look ahead
        mag = tf.pad(mag, [[0, 0], [0, 2], [0, 0]])
        ref_mag = tf.pad(ref_mag, [[0, 0], [0, 2], [0, 0]])

        # Full band
        norm_mag = self.norm_3d(mag)
        norm_ref_mag = self.norm_3d(ref_mag)
        fb_input = tf.concat([norm_mag, norm_ref_mag], axis=-1)
        fb1 = self.fb_lstm_1(fb_input)
        fb2 = self.fb_lstm_2(fb1)
        fb_out = self.fb_dense(fb2)

        # Unfold
        mag_unfold = self.norm_4d(self.unfold(mag))
        ref_mag_unfold = self.norm_4d(self.unfold(ref_mag))
        fb_out_unfold = self.norm_4d(tf.expand_dims(fb_out, axis=-1))

        # Sub band
        sb_input = tf.concat([fb_out_unfold, mag_unfold, ref_mag_unfold], axis=-1)
        # [B, T, F, F_s] -> [B, F, T, F_s]
        sb_input = tf.transpose(sb_input, perm=(0, 2, 1, 3))
        sb1 = self.sb_lstm_1(sb_input)
        sb2 = self.sb_lstm_2(sb1)
        sb_out = self.sb_dense(sb2)

        # Remove first two frames because 2-frame look-ahead
        sb_out = sb_out[:, :, 2:, :]
        sb_out = tf.transpose(sb_out, perm=(0, 2, 1, 3))

        # Output is complex ideal ratio mask.
        return sb_out

    def summary(self):
        i1 = tf.keras.Input(shape=(313, 257), dtype=np.float32)
        i2 = tf.keras.Input(shape=(313, 257), dtype=np.float32)
        model = tf.keras.Model(inputs=[i1, i2], outputs=self.call([i1, i2]))
        return model.summary()

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


def compress_cIRM(mask, K=10, C=0.1):
    """
        Compress from (-inf, +inf) to [-K ~ K]
    """
    mask = tf.math.maximum(-100, mask)
    if tf.is_tensor(mask):
        mask = K * (1 - tf.math.exp(-C * mask)) / (1 + tf.math.exp(-C * mask))
    else:
        mask = K * (1 - np.exp(-C * mask)) / (1 + np.exp(-C * mask))
    return mask


def decompress_cIRM(mask, K=10, limit=9.9):
    mask = np.maximum(np.minimum(mask, limit), -limit)
    mask = -K * np.log((K - mask) / (K + mask))
    return mask


def build_ideal_mask(mixed, truth):
    denominator = tf.math.square(mixed.real) + tf.math.square(mixed.imag) + 1e-6

    mask_real = (mixed.real * truth.real + mixed.imag * truth.imag) / denominator
    mask_imag = (mixed.real * truth.imag - mixed.imag * truth.real) / denominator

    complex_ratio_mask = tf.stack((mask_real, mask_imag), axis=-1)

    return compress_cIRM(complex_ratio_mask, K=10, C=0.1)


def result_from_mask(mixed, mask):
    mask = decompress_cIRM(mask)
    enhanced_real = mask[..., 0] * mixed.real - mask[..., 1] * mixed.imag
    enhanced_imag = mask[..., 1] * mixed.real + mask[..., 0] * mixed.imag
    return enhanced_real + enhanced_imag*1j


def get_baseline_model():
    model = BaselineModel()
    i1 = tf.keras.Input(shape=(313, 257), dtype=np.float32)
    i2 = tf.keras.Input(shape=(313, 257), dtype=np.float32)
    model = tf.keras.Model(inputs=[i1, i2], outputs=model.call([i1, i2]))
    return model


if __name__ == "__main__":
    # tf.config.set_visible_devices([], 'GPU')

    model = BaselineModel()
    model.summary()
    model([tf.constant(0, shape=(4, 313, 257), dtype=np.float32),
           tf.constant(0, shape=(4, 313, 257), dtype=np.float32)], training=True)
