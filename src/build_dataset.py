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
        'kw_samples': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'kw_class': tf.io.FixedLenFeature([], tf.int64, default_value=0)
    }
    example = tf.io.parse_single_example(example, feature_description)
    kw_samples = tf.io.parse_tensor(example['kw_samples'], out_type=tf.float32)
    kw_samples = tf.reshape(kw_samples, (16000,))
    kw_class = example['kw_class']

    return kw_samples, kw_class


def transform_fn(kw_samples, kw_class, M):
    """Get up and running with log-mel-spectrogram."""
    stft = tf.signal.stft(kw_samples, frame_length=512, frame_step=256)
    spectrogram = tf.abs(stft)
    x = tf.math.log(spectrogram @ M + 1e-6)
    x = tf.expand_dims(x, axis=2)
    y = tf.cast(kw_class, dtype=tf.uint16)

    return x, y


def build_filelist(mode):
    return glob.glob('../data/TFRecords/{}_*.tfr'.format(mode))


def build_dataset(mode):
    filenames = build_filelist(mode)
    filenames.sort()  # deterministic ordering
    ds_filenames = tf.data.Dataset.from_tensor_slices(tf.constant(filenames))

    if mode == 'train':
        ds_filenames = ds_filenames.shuffle(len(filenames),
                                            reshuffle_each_iteration=True)
    M = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=30,
        num_spectrogram_bins=257,
        sample_rate=16000,
        lower_edge_hertz=40,
        upper_edge_hertz=8000)
    ds = tf.data.TFRecordDataset(ds_filenames, num_parallel_reads=6)
    ds = ds.map(decode, num_parallel_calls=4)
    ds = ds.map(lambda x, y: transform_fn(x, y, M))
    if mode == 'train':
        ds = ds.shuffle(1024)
    ds = ds.batch(64)
    ds = ds.prefetch(2)

    return ds
