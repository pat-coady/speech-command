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
    kw_samples = tf.io.parse_tensor(example['x'], out_type=tf.float32)
    kw_samples = tf.reshape(kw_samples, (61, 40, 1))
    kw_class = tf.cast(example['y'], tf.int32)

    return kw_samples, kw_class


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
    ds = ds.batch(128)
    ds = ds.prefetch(4)

    return ds
