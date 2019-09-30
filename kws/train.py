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

See tfrecords.py for dataset preparation.

Update: add provision to take audio encoded using network trained with
    Contrastive Predictive Coding.
"""
import tensorflow as tf
from tensorflow import keras
from dataset import build_dataset
from model import cnn, dense_conv
import time
import os
import shutil
import glob
import argparse


def logpaths(config):
    """
    Build training log paths for checkpoints and TensorBoard data. TensorBoard
    tb_logs directories are time-stamped to keep them unique.

    Returns:
        ckpt_path (str)
        tblog_path (str)
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


def get_optimizer(config):
    if config['optimizer'] == 'adam':
        return tf.optimizers.Adam(config['lr'])
    if config['optimizer'] == 'sgd':
        return tf.optimizers.SGD(config['lr'],
                                 momentum=config['momentum'],
                                 nesterov=True)


def main(config):
    if config['dense_conv']:
        model = dense_conv(config)
    else:
        model = cnn(config)
    with tf.device('/cpu:0'):  # put data pipeline on CPU
        ds_train = build_dataset(config, 'train')
        ds_val = build_dataset(config, 'val')
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = get_optimizer(config)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    callbacks = build_callbacks(config)
    model.fit(x=ds_train, validation_data=ds_val, epochs=config['epochs'],
              callbacks=callbacks, validation_steps=90)
    print(model.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument(
        '-t', '--tag', type=str,
        help='Tag for the training run (default = "default")',
        default='default')
    parser.add_argument(
        '-lr', '--lr', type=float,
        help='Learning rate (default=0.001)',
        default=0.001)
    parser.add_argument(
        '-m', '--momentum', type=float,
        help='Momentum (default=0.9)',
        default=0.9)
    parser.add_argument(
        '-e', '--epochs', type=int,
        help='Number of epochs (default=15)',
        default=25)
    parser.add_argument(
        '-b', '--batch_sz', type=int,
        help='Batch size (default=64)',
        default=64)
    parser.add_argument(
        '-ds', '--ds_type', type=str,
        help='Choose "log-mel-spec", "cpc-enc" or "samples".',
        default='log-mel-spec')
    parser.add_argument(
        '--optimizer', type=str,
        help='Choose optimizer: "adam" or "sgd". (default=adam)',
        default='adam')
    parser.add_argument(
        '-bn', '--batch_norm', action='store_true',
        help='Enable batch normalization.')
    parser.set_defaults(batch_norm=False)
    parser.add_argument(
        '-dc', '--dense_conv', action='store_true',
        help='Convolve with fully connected filters covering T=3 input steps.')
    parser.set_defaults(dense_conv=False)
    parser.add_argument(
        '-df', '--dense_final', action='store_true',
        help='Use dense layer as final output instead of Global Average Pool.')
    parser.set_defaults(dense_final=False)

    main(vars(parser.parse_args()))
