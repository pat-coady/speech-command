"""
Construct tf.data.Dataset from Kaggle KWS Dataset:
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge

by: Patrick Coady
"""
import tensorflow as tf
import glob


def decode(example):
    """Parses an image and depth map from the given `serialized_example`."""
    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'y': tf.io.FixedLenFeature([], tf.int64, default_value=0)
    }
    example = tf.io.parse_single_example(example, feature_description)
    x = tf.io.parse_tensor(example['x'], out_type=tf.float32)
    x = tf.reshape(x, (61, 40, 1))
    x = tf.clip_by_value(x, -10.0, 5.0)
    x = (x + 2.5) / 7.5
    y = tf.cast(example['y'], tf.int32)

    return x, y


def build_filelist(ds_type, mode):
    return glob.glob('../data/TFRecords/{}/{}_*.tfr'.format(ds_type, mode))


def build_dataset(ds_type, mode):
    filenames = build_filelist(ds_type, mode)
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
