"""
Use trained Constrastive Predictive Coding model to encode audio samples
in learned latent space.

by: Patrick Coady
"""
import tensorflow as tf
from pathlib import Path


def decode(example):
    """Parses an image and depth map from the given `serialized_example`."""
    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'y': tf.io.FixedLenFeature([], tf.int64, default_value=0)
    }
    example = tf.io.parse_single_example(example, feature_description)
    x = tf.io.parse_tensor(example['x'], out_type=tf.float32)
    x = tf.reshape(x, (16000,))
    x = x - tf.reduce_mean(x)
    max_x = tf.reduce_max(x)
    min_x = tf.reduce_min(x)
    scale = tf.maximum(-min_x, max_x)
    x = x / scale
    y = tf.cast(example['y'], tf.int32)

    return x, y


def build_filelist(mode):
    path = Path.home() / 'Data' / 'kws' / 'TFRecords' / 'samples'
    filenames = path.rglob('{}_*.tfr'.format(mode))

    return list(map(lambda fn: str(fn), filenames))


def build_dataset(mode):
    filenames = build_filelist(mode)
    filenames.sort()  # deterministic ordering
    ds_filenames = tf.data.Dataset.from_tensor_slices(tf.constant(filenames))

    if mode == 'train':
        ds_filenames = ds_filenames.shuffle(len(filenames),
                                            reshuffle_each_iteration=True)
    ds = tf.data.TFRecordDataset(ds_filenames, num_parallel_reads=8)
    ds = ds.map(decode, num_parallel_calls=8)
    if mode == 'train':
        ds = ds.shuffle(1024)
    ds = ds.batch(64)
    ds = ds.prefetch(4)

    return ds
