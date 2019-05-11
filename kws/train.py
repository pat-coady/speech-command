#! /usr/bin/env python3
"""
CNN-based audio keyword detector.

The dataset consists of ~2,000 1 second recordings per keyword.
There are 30 total keywords. Dataset available here:
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge

The audio .wav files are converted to a 30-bin log-mel-spectrogram
with 32ms frames and 16ms overlap. The resulting spectrogram is
a 61x30 "image". The spectrograms are fed into a CNN, with global
average pooling as the final output layer.

Very first training attempt reached 90% accuracy on the validation
set with no bells or whistles (i.e. batch norm, data augmentation,
residual layers, ...). Just a plain-old CNN. Trained on a macbook
in about 10 minutes.

by: Patrick Coady
"""
from tensorflow import keras
from build_dataset import *
import time
import os
import shutil
import glob


def conv(x, filters):
    return keras.layers.Conv2D(filters, (3, 3), padding='same',
                               activation='relu6',
                               kernel_initializer='he_uniform')(x)


def pool(x):
    return keras.layers.MaxPool2D((2, 2))(x)


def build_model():
    # TODO - Remove hard-coded shape:
    x = keras.Input(shape=(61, 40, 1), name='input')
    y = conv(x, 16)
    y = conv(y, 16)
    y = pool(y)
    y = conv(y, 32)
    y = conv(y, 32)
    y = pool(y)
    y = conv(y, 30)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)

    return keras.Model(x, y)


def logpaths():
    """Build training log paths for checkpoints and TensorBoard data

    Returns:
        (ckpt_path, tblog_path)
    """
    model_name = 'simple_conv'
    tag = 'default'
    logdir = time.strftime("run_%Y_%m_%d-%H_%M_%S/")

    ckpt_path = ('../outputs/checkpoints/' + model_name + '/'
                 + tag)
    tblog_path = ('../outputs/tb_logs/' + model_name + '/'
                  + tag + '/' + logdir)

    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    filenames = glob.glob('*.py')
    for filename in filenames:
        shutil.copy(filename, ckpt_path)

    return ckpt_path, tblog_path


def build_callbacks():
    """Various callbacks, e.g. Tensorboard, checkpoints"""
    ckpt_path, tblog_path = logpaths()
    callbacks = []
    cb = keras.callbacks.ModelCheckpoint(ckpt_path + '/cp-{epoch:04d}.ckpt',
                                         save_weights_only=True,
                                         verbose=True)
    callbacks.append(cb)
    cb = keras.callbacks.TensorBoard(tblog_path, histogram_freq=1,
                                     update_freq=1000)
    callbacks.append(cb)

    return callbacks


def main():
    model = build_model()
    with tf.device('/cpu:0'):  # put data pipeline on CPU
        ds_train = build_dataset('train')
        ds_val = build_dataset('val')
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.Adam()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    callbacks = build_callbacks()
    model.fit(x=ds_train, validation_data=ds_val, epochs=10,
              callbacks=callbacks)


if __name__ == "__main__":
    main()
