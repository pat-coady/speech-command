#! /usr/bin/env python3
"""
CNN-based audio keyword detector.

The dataset consists of ~2,000 1 second recordings per keyword.
There are 30 total keywords. Dataset available here:
https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html

The audio .wav files are converted to a 30-bin log-mel-spectrogram
with 32ms frames and 16ms overlap. The resulting spectrogram is
a 61x30 "image". The spectrograms are fed into a CNN, with global
average pooling as the final output layer.

See tfrecords.py for dataset preparation.

Update: Added provision to take audio encoded using network trained with
    Contrastive Predictive Coding. Create dataset using cpc_tfrecords.py
"""
import tensorflow as tf
import argparse
import json
import shutil
from pathlib import Path

from dataset import build_dataset
from model import cnn, tall_kernel
from callbacks import build_callbacks


def add_logpaths(config):
    """Add paths for checkpoints and TensorBoard data to the config dictionary."""
    config['ckpt_path'] = str(Path.cwd() / 'outputs' / 'checkpoints')
    config['tblog_path'] = str(Path.cwd() / 'logs')
    Path(config['ckpt_path']).mkdir(parents=True, exist_ok=True)


def save_state(config):
    """Save configuration of training run: all *.py files and config dictionary."""
    filenames = Path.cwd().glob('*.py')
    for filename in filenames:
        shutil.copy(str(filename), config['ckpt_path'])
    with (Path(config['ckpt_path']) / 'config.json').open('w') as f:
        json.dump(config, f)


def init_azure_logging(config):
    """Get Azure run logger and log all configuration settings."""
    from azureml.core.run import Run
    run_logger = Run.get_context()
    for k in config:
        run_logger.log(k, config[k])

    return run_logger


def get_optimizer(config):
    if config['optimizer'] == 'adam':
        return tf.optimizers.Adam(config['lr'])
    if config['optimizer'] == 'sgd':
        return tf.optimizers.SGD(config['lr'],
                                 momentum=config['momentum'],
                                 nesterov=True)


def main(config):
    run_logger = None
    if config['azure_ml']:
        run_logger = init_azure_logging(config)
    if config['tall_kernel']:
        model = tall_kernel(config)
    else:
        model = cnn(config)
    with tf.device('/cpu:0'):  # put data pipeline on CPU
        ds_train = build_dataset(config, 'train')
        ds_val = build_dataset(config, 'val')
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = get_optimizer(config)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    add_logpaths(config)
    save_state(config)
    callbacks = build_callbacks(config, run_logger)
    model.fit(x=ds_train, validation_data=ds_val, epochs=config['epochs'],
              callbacks=callbacks, validation_steps=90, verbose=config['verbose'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument(
        '-lr', '--lr', type=float,
        help='Learning rate (default=0.001)',
        default=0.01)
    parser.add_argument(
        '-m', '--momentum', type=float,
        help='Momentum (default=0.9)',
        default=0.9)
    parser.add_argument(
        '-e', '--epochs', type=int,
        help='Number of epochs (default=25)',
        default=25)
    parser.add_argument(
        '-b', '--batch_sz', type=int,
        help='Batch size (default=64)',
        default=64)
    parser.add_argument(
        '--data_dir', type=str,
        help='Path to training data (default = ~/Data/kws/tfrecords).',
        default=str(Path.home() / 'Data' / 'kws' / 'tfrecords'))
    parser.add_argument(
        '-ds', '--ds_type', type=str,
        help='Choose "log-mel", "cpc-enc" or "samples".',
        default='log-mel')
    parser.add_argument(
        '--optimizer', type=str,
        help='Choose optimizer: "adam" or "sgd". (default=adam)',
        default='adam')
    parser.add_argument(
        '-bn', '--batch_norm', type=int,
        help='Enable batch normalization. 1 to enable, 0 to disable. (default = 1).',
        default=1)  # azure hyperparam doesn't like flags and only displays numbers
    parser.add_argument(
        '-tk', '--tall_kernel', type=int,
        help='Convolve with fully connected filters covering T=3 input steps. 1 to enable.',
        default=0)  # azure hyperparam doesn't like flags and only displays numbers
    parser.add_argument(
        '-df', '--dense_final', type=int,
        help='Use dense layer as final output instead of Global Average Pool. 1 to enable.',
        default=0)  # azure hyperparam doesn't like flags and only displays numbers
    parser.add_argument(
        '--load_genc', type=int,
        help='Load trained encoder (typically from CPC training).',
        default=0)  # azure hyperparam doesn't like flags and only displays numbers
    parser.add_argument(
        '--train_genc', type=int,
        help='Enable training gradients to encoder. For ds_type = "samples".',
        default=0)  # azure hyperparam doesn't like flags and only displays numbers
    parser.add_argument(
        '--azure_ml', action='store_true',
        help='Enable Azure ML logging.')
    parser.set_defaults(azure_ml=False)
    parser.add_argument(
        '-v', '--verbose', type=int,
        help='Verbose level for Keras Model.fit(). (default=2)',
        default=2
    )

    main(vars(parser.parse_args()))
