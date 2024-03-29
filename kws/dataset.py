"""
Construct tf.data.Dataset from processed Kaggle KWS data.

https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html

Use tfrecords.py to construct TFRecords.
"""
import tensorflow as tf
from pathlib import Path


def decode(example, ds_type):
    """Parses example from `serialized_example`."""
    epsilon = 1e-5
    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'y': tf.io.FixedLenFeature([], tf.int64, default_value=0)
    }
    example = tf.io.parse_single_example(example, feature_description)
    x = tf.io.parse_tensor(example['x'], out_type=tf.float32)
    if ds_type == 'samples':
        x = tf.reshape(x, (16000,))
        x = x - tf.reduce_mean(x)
        max_x = tf.reduce_max(x)
        min_x = tf.reduce_min(x)
        scale = tf.maximum(-min_x, max_x)
        x = x / (scale + epsilon)
    elif ds_type in ['mfcc', 'log-mel']:
        x = tf.reshape(x, (61, 40, 1))
        # magic numbers to roughly scale inputs to [-1, 1]
        x = tf.clip_by_value(x, -10.0, 5.0)
        x = (x + 2.5) / 7.5
    elif ds_type == 'cpc-enc':
        # magic number to roughly scale inputs to [-1, 1]
        x = tf.reshape(x, (63, 40, 1)) / 4
    else:
        return None
    y = tf.cast(example['y'], tf.int32)

    return x, y


def build_filelist(config, mode):
    path = Path(config['data_dir']) / config['ds_type']
    filenames = path.rglob('{}_*.tfr'.format(mode))

    return list(map(lambda fn: str(fn), filenames))


def build_dataset(config, mode):
    filenames = build_filelist(config, mode)
    filenames.sort()  # deterministic ordering
    ds_filenames = tf.data.Dataset.from_tensor_slices(tf.constant(filenames))

    if mode == 'train':
        ds_filenames = ds_filenames.shuffle(len(filenames),
                                            reshuffle_each_iteration=True)
    ds = tf.data.TFRecordDataset(ds_filenames, num_parallel_reads=8)
    ds = ds.map(lambda example: decode(example, config['ds_type']),
                num_parallel_calls=8)
    if mode == 'train':
        ds = ds.shuffle(1024)
    ds = ds.batch(config['batch_sz'])
    ds = ds.prefetch(4)

    return ds
