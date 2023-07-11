# -*- coding: utf-8 -*-
"""
Modified Wave-U-Net implementation on AEC problem.
"""

import tensorflow as tf
from wave_u_net.wave_u_net import InterpolationLayer, IndependentOutputLayer, DiffOutputLayer, AudioClipLayer, CropLayer

"""# Define the Network"""


def wave_u_net_aec(num_initial_filters=24, num_layers=12, kernel_size=15, merge_filter_size=5,
                   source_names=["bass", "drums", "other", "vocals"], num_channels=1, output_filter_size=1,
                   padding="same", input_size=16384 * 4, context=False, upsampling_type="learned",
                   output_activation="linear", output_type="difference"):
    # `enc_outputs` stores the downsampled outputs to re-use during upsampling.
    enc_outputs = []
    far_end_enc_outputs = []

    mixture_input = tf.keras.layers.Input(shape=(input_size, num_channels), name="Mixture_input")
    X = mixture_input
    inp_mix = mixture_input

    far_end_input = tf.keras.layers.Input(shape=(input_size, num_channels), name="Far_End_input")
    Y = far_end_input
    inp_far_end = far_end_input

    # Down sampling
    for i in range(num_layers):
        X = tf.keras.layers.Conv1D(filters=num_initial_filters + (num_initial_filters * i),
                                   kernel_size=kernel_size, strides=1,
                                   padding=padding, name="Mixture_Down_Conv_" + str(i))(X)
        X = tf.keras.layers.LeakyReLU(name="Mixture_Down_Conv_Activ_" + str(i))(X)

        enc_outputs.append(X)

        X = tf.keras.layers.Lambda(lambda x: x[:, ::2, :], name="Mixture_Decimate_" + str(i))(X)

    X = tf.keras.layers.Conv1D(filters=num_initial_filters + (num_initial_filters * num_layers),
                               kernel_size=kernel_size, strides=1,
                               padding=padding, name="Mixture_Down_Conv_" + str(num_layers))(X)
    X = tf.keras.layers.LeakyReLU(name="Mixture_Down_Conv_Activ_" + str(num_layers))(X)


    for i in range(num_layers):
        Y = tf.keras.layers.Conv1D(filters=num_initial_filters + (num_initial_filters * i),
                                   kernel_size=kernel_size, strides=1,
                                   padding=padding, name="Far_End_Down_Conv_" + str(i))(Y)
        Y = tf.keras.layers.LeakyReLU(name="Far_End_Down_Conv_Activ_" + str(i))(Y)

        far_end_enc_outputs.append(Y)

        Y = tf.keras.layers.Lambda(lambda x: x[:, ::2, :], name="Far_End_Decimate_" + str(i))(Y)

    Y = tf.keras.layers.Conv1D(filters=num_initial_filters + (num_initial_filters * num_layers),
                               kernel_size=kernel_size, strides=1,
                               padding=padding, name="Far_End_Down_Conv_" + str(num_layers))(Y)
    Y = tf.keras.layers.LeakyReLU(name="Far_End_Down_Conv_Activ_" + str(num_layers))(Y)

    X = tf.keras.layers.Concatenate(axis=2, name="concat_two_downsampling_subnetwork")([X, Y])

    # Up sampling
    for i in range(num_layers):
        X = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1), name="exp_dims_" + str(i))(X)

        if upsampling_type == "learned":
            X = InterpolationLayer(name="IntPol_" + str(i), padding=padding)(X)

        else:
            if context:
                X = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [1, x.shape.as_list()[2] * 2 - 1]),
                                           name="bilinear_interpol_" + str(i))(X)
                # current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2] * 2 - 1], align_corners=True)
            else:
                X = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [1, x.shape.as_list()[2] * 2]),
                                           name="bilinear_interpol_" + str(i))(X)
                # current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2]*2]) # out = in + in - 1

        X = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1), name="sq_dims_" + str(i))(X)

        c_layer1 = CropLayer(X, False, name="crop_layer_mixture_" + str(i))(enc_outputs[-i - 1])
        c_layer2 = CropLayer(X, False, name="crop_layer_far_end_" + str(i))(far_end_enc_outputs[-i - 1])
        X = tf.keras.layers.Concatenate(axis=2, name="concatenate_" + str(i))([X, c_layer1, c_layer2])

        X = tf.keras.layers.Conv1D(filters=num_initial_filters + (num_initial_filters * (num_layers - i - 1)),
                                   kernel_size=merge_filter_size, strides=1,
                                   padding=padding, name="Up_Conv_" + str(i))(X)
        X = tf.keras.layers.LeakyReLU(name="Up_Conv_Activ_" + str(i))(X)

    c_layer1 = CropLayer(X, False, name="crop_layer_mixture" + str(num_layers))(inp_mix)
    c_layer2 = CropLayer(X, False, name="crop_layer_far_end_" + str(num_layers))(inp_far_end)
    X = tf.keras.layers.Concatenate(axis=2, name="concatenate_" + str(num_layers))([X, c_layer1, c_layer2])
    X = AudioClipLayer(name="audio_clip_" + str(0))(X)

    if output_type == "direct":
        X = IndependentOutputLayer(source_names, num_channels, output_filter_size, padding=padding,
                                   name="independent_out")(X)

    else:
        # Difference Output
        cropped_input = CropLayer(X, False, name="crop_layer_" + str(num_layers + 1))(inp_mix)
        X = DiffOutputLayer(source_names, num_channels, output_filter_size, padding=padding, name="diff_out")(
            [X, cropped_input])

    o = X
    model = tf.keras.Model(inputs=[mixture_input, far_end_input], outputs=o)
    return model


# Parameters for the Wave-U-net

params = {
    "num_initial_filters": 24,
    "num_layers": 12,
    "kernel_size": 15,
    "merge_filter_size": 5,
    "source_names": ["bass", "drums", "other", "vocals"],
    "num_channels": 2,
    "output_filter_size": 1,
    "padding": "valid",
    "input_size": 147443,
    "context": True,
    "upsampling_type": "learned",  # "learned" or "linear"
    "output_activation": "linear",  # "linear" or "tanh"
    "output_type": "difference",  # "direct" or "difference"
}

