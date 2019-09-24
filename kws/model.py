"""
Models
"""
from tensorflow import keras


def conv2d(x, filters):
    y = keras.layers.Conv2D(filters, (3, 3), padding='same',
                            kernel_initializer='he_uniform')(x)
    return keras.activations.relu(y, max_value=6.0)


def conv2d_bn(x, filters):
    y = keras.layers.Conv2D(filters, (3, 3), padding='same',
                            kernel_initializer='he_uniform')(x)
    y = keras.layers.BatchNormalization()(y)
    return keras.activations.relu(y)


def pool(x):
    return keras.layers.MaxPool2D((2, 2))(x)


def cnn(batch_norm=False):
    if batch_norm:
        conv = conv2d_bn
    else:
        conv = conv2d
    x = keras.Input(shape=(None, 40, 1), name='input')
    y = keras.layers.Conv2D(4, (3, 40), padding='same',
                            kernel_initializer='he_uniform')(x)
    y = conv(y, 16)
    y = conv(y, 16)
    y = pool(y)
    y = conv(y, 32)
    y = conv(y, 32)
    y = pool(y)
    y = conv(y, 64)
    y = conv(y, 30)
    y = keras.layers.GlobalAveragePooling2D()(y)

    return keras.Model(x, y)
