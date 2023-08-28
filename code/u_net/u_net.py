from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, LeakyReLU, Conv2DTranspose as Deconv2D, Activation, Concatenate


def u_net(inputs=(Input((128, 512, 1), ), Input((128, 512, 1), ))):
    conv1 = Conv2D(16, 5, strides=2, padding='same')(inputs[0])
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)

    conv2 = Conv2D(32, 5, strides=2, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)

    conv3 = Conv2D(64, 5, strides=2, padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)

    conv4 = Conv2D(128, 5, strides=2, padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.2)(conv4)

    conv5 = Conv2D(256, 5, strides=2, padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.2)(conv5)

    conv6 = Conv2D(512, 5, strides=2, padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=0.2)(conv6)

    ref_conv1 = Conv2D(16, 5, strides=2, padding='same')(inputs[1])
    ref_conv1 = BatchNormalization()(ref_conv1)
    ref_conv1 = LeakyReLU(alpha=0.2)(ref_conv1)

    ref_conv2 = Conv2D(32, 5, strides=2, padding='same')(ref_conv1)
    ref_conv2 = BatchNormalization()(ref_conv2)
    ref_conv2 = LeakyReLU(alpha=0.2)(ref_conv2)

    ref_conv3 = Conv2D(64, 5, strides=2, padding='same')(ref_conv2)
    ref_conv3 = BatchNormalization()(ref_conv3)
    ref_conv3 = LeakyReLU(alpha=0.2)(ref_conv3)

    ref_conv4 = Conv2D(128, 5, strides=2, padding='same')(ref_conv3)
    ref_conv4 = BatchNormalization()(ref_conv4)
    ref_conv4 = LeakyReLU(alpha=0.2)(ref_conv4)

    ref_conv5 = Conv2D(256, 5, strides=2, padding='same')(ref_conv4)
    ref_conv5 = BatchNormalization()(ref_conv5)
    ref_conv5 = LeakyReLU(alpha=0.2)(ref_conv5)

    ref_conv6 = Conv2D(512, 5, strides=2, padding='same')(ref_conv5)
    ref_conv6 = BatchNormalization()(ref_conv6)
    ref_conv6 = LeakyReLU(alpha=0.2)(ref_conv6)

    conv6 = Concatenate(axis=3)([conv6, ref_conv6])
    deconv7 = Deconv2D(256, 5, strides=2, padding='same')(conv6)
    deconv7 = BatchNormalization()(deconv7)
    deconv7 = Dropout(0.5)(deconv7)
    deconv7 = Activation('relu')(deconv7)

    deconv8 = Concatenate(axis=3)([deconv7, conv5, ref_conv5])
    deconv8 = Deconv2D(128, 5, strides=2, padding='same')(deconv8)
    deconv8 = BatchNormalization()(deconv8)
    deconv8 = Dropout(0.5)(deconv8)
    deconv8 = Activation('relu')(deconv8)

    deconv9 = Concatenate(axis=3)([deconv8, conv4, ref_conv4])
    deconv9 = Deconv2D(64, 5, strides=2, padding='same')(deconv9)
    deconv9 = BatchNormalization()(deconv9)
    deconv9 = Dropout(0.5)(deconv9)
    deconv9 = Activation('relu')(deconv9)

    deconv10 = Concatenate(axis=3)([deconv9, conv3, ref_conv3])
    deconv10 = Deconv2D(32, 5, strides=2, padding='same')(deconv10)
    deconv10 = BatchNormalization()(deconv10)
    deconv10 = Activation('relu')(deconv10)

    deconv11 = Concatenate(axis=3)([deconv10, conv2, ref_conv2])
    deconv11 = Deconv2D(16, 5, strides=2, padding='same')(deconv11)
    deconv11 = BatchNormalization()(deconv11)
    deconv11 = Activation('relu')(deconv11)

    deconv12 = Concatenate(axis=3)([deconv11, conv1, ref_conv1])
    deconv12 = Deconv2D(1, 5, strides=2, padding='same')(deconv12)
    deconv12 = Activation('sigmoid')(deconv12)

    output = deconv12 * inputs[0]

    model = Model(inputs=inputs, outputs=output)
    # model.summary()
    return model

