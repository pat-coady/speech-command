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
"""
import tensorflow as tf
from tensorflow import keras
from dataset import build_dataset
from model import cnn
import time
import os
import shutil
import glob
import argparse


def logpaths(config):
    """Build training log paths for checkpoints and TensorBoard data

    Returns:
        (ckpt_path, tblog_path)
    """
    tag = config['tag']
    logdir = time.strftime("run_%Y_%m_%d-%H_%M_%S/")

    ckpt_path = '../outputs/checkpoints/{}'.format(tag)
    tblog_path = '../outputs/tb_logs/{}/{}'.format(tag, logdir)

    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    filenames = glob.glob('*.py')
    for filename in filenames:
        shutil.copy(filename, ckpt_path)

    return ckpt_path, tblog_path


def build_callbacks(config):
    """Various callbacks, e.g. Tensorboard, checkpoints"""
    ckpt_path, tblog_path = logpaths(config)
    callbacks = []
    cb = keras.callbacks.ModelCheckpoint(ckpt_path + '/cp-{epoch:04d}.ckpt',
                                         save_weights_only=True,
                                         verbose=True)
    callbacks.append(cb)
    cb = keras.callbacks.TensorBoard(tblog_path, histogram_freq=1,
                                     update_freq=100)
    callbacks.append(cb)

    return callbacks


def main(config):
    model = cnn(config)
    with tf.device('/cpu:0'):  # put data pipeline on CPU
        ds_train = build_dataset(config, 'train')
        ds_val = build_dataset(config, 'val')
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=config['optimizer'], metrics=['accuracy'])
    callbacks = build_callbacks(config)
    model.fit(x=ds_train, validation_data=ds_val, epochs=config['epochs'],
              callbacks=callbacks, validation_steps=90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument(
        '-t', '--tag', type=str,
        help='Tag for the training run (default = "default")',
        default='default')
    parser.add_argument(
        '-e', '--epochs', type=int,
        help='Number of epochs (default=15)',
        default=15)
    parser.add_argument(
        '-b', '--batch_sz', type=int,
        help='Batch size (default=64)',
        default=64)
    parser.add_argument(
        '-ds', '--ds_type', type=str,
        help='Choose "log-mel-spec" or "cpc-enc".',
        default='log-mel-spec')
    parser.add_argument(
        '--optimizer', type=str,
        help='Choose optimizer: "adam" or "sgd_mom". (default=adam)',
        default='adam')
    parser.add_argument(
        '-bn', '--batch_norm', action='store_true',
        help='Enable batch normalization.')
    parser.set_defaults(batch_norm=False)
    parser.add_argument(
        '-tc', '--tall_conv', action='store_true',
        help='Enable 40x3x4 first layer convolution.')
    parser.set_defaults(batch_norm=False)

    main(vars(parser.parse_args()))
