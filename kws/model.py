"""
Models
"""
from tensorflow import keras
from replearn.models import genc_model


def conv2d(x, filters):
    y = keras.layers.Conv2D(filters, (3, 3), padding='same',
                            kernel_initializer='he_uniform')(x)
    return keras.activations.relu(y)


def conv2d_bn(x, filters):
    y = keras.layers.Conv2D(filters, (3, 3), padding='same',
                            kernel_initializer='he_uniform')(x)
    y = keras.layers.BatchNormalization()(y)
    return keras.activations.relu(y)


def pool(x):
    return keras.layers.MaxPool2D((2, 2))(x)


def cnn(config):
    if config['batch_norm']:
        conv = conv2d_bn
    else:
        conv = conv2d
    if config['ds_type'] == 'samples':
        x = keras.Input(shape=(16000,), name='input')
        genc = genc_model(dim_z=40)
        y = genc(x)
        y = keras.layers.Reshape((-1, 40, 1))(y)
        y = keras.layers.Cropping2D(((5, 0), (0, 0)))(y)
        y = conv(y, 16)
    else:
        x = keras.Input(shape=(None, 40, 1), name='input')
        y = conv(x, 16)
    y = conv(y, 16)
    y = pool(y)
    y = conv(y, 32)
    y = conv(y, 32)
    y = pool(y)
    y = conv(y, 64)
    y = conv(y, 30)
    y = keras.layers.GlobalAveragePooling2D()(y)

    return keras.Model(x, y)


def dense_conv(config):
    if config['ds_type'] == 'samples':
        x = keras.Input(shape=(16000,), name='input')
        genc = genc_model(dim_z=40)
        y = genc(x)
        y = keras.layers.Reshape((-1, 40, 1))(y)
        y = keras.layers.Cropping2D(((5, 0), (0, 0)))(y)
        y = keras.layers.Conv2D(80, (3, 40), (2, 1), padding='valid',
                                kernel_initializer='he_uniform',
                                activation='relu')(y)
    else:
        x = keras.Input(shape=(None, 40, 1), name='input')
        y = keras.layers.Conv2D(80, (3, 40), (2, 1), padding='valid',
                                kernel_initializer='he_uniform',
                                activation='relu')(x)
    y = keras.layers.Conv2D(160, (3, 1), (2, 1), padding='valid',
                            kernel_initializer='he_uniform',
                            activation='relu')(y)
    y = keras.layers.Conv2D(160, (3, 1), (2, 1), padding='valid',
                            kernel_initializer='he_uniform',
                            activation='relu')(y)
    if config['dense_final']:
        y = keras.layers.Flatten()(y)
        y = keras.layers.Dense(90, 'relu', kernel_initializer='he_uniform')(y)
        y = keras.layers.Dense(30, kernel_initializer='he_uniform')(y)
    else:
        y = keras.layers.Conv2D(30, (3, 1), (2, 1), padding='valid',
                                kernel_initializer='he_uniform')(y)
        y = keras.layers.GlobalAveragePooling2D()(y)

    return keras.Model(x, y)
