#! /usr/bin/env python
"""
Build training and validation sets in TFRecord format.

30 classes (keywords), no unknown or background_noise class for now.

by: Patrick Coady
"""
import tensorflow as tf
import hashlib
import os
import re
import glob
import random


# from: https://www.tensorflow.org/alpha/tutorials/load_data/tf_records
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# from: README.md, Speech Commands Data Set v0.01
# (Fixed a few minor items --pco)
def which_set(filename, validation_percentage=10, testing_percentage=10):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
      filename: File path of the data sample.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      String, one of 'training', 'validation', or 'testing'.
    """
    max_num_waves_per_class = 2 ** 27 - 1  # ~134M
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name).encode('utf-8')
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (max_num_waves_per_class + 1)) *
                       (100.0 / max_num_waves_per_class))
    if percentage_hash < validation_percentage:
        result = 'val'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'test'
    else:
        result = 'train'
    return result


# Start of my code
def get_all_filenames():
    """Return list of all training files except '_background_noise_"""
    filenames = glob.glob('../data/train/audio/*/*.wav')

    return list(filter(lambda x: 'background' not in x, filenames))


def build_class_dict():
    """Key = class (str), value = integer (0 to num_classes-1)."""
    folders = glob.glob('../data/train/audio/*')
    folders = map(lambda x: x.split('/')[-1], folders)
    folders = list(filter(lambda x: 'background' not in x, folders))
    folders.sort()
    class_dict = {}
    for i, folder in enumerate(folders):
        class_dict[folder] = i

    return class_dict


def get_class_int(filename, class_dict):
    class_str = filename.split('/')[-2]

    return class_dict[class_str]


class MyTFRWriter(object):
    def __init__(self, basename, shard_size=256):
        self.basename = basename
        self.shard_size = shard_size
        self.path = '../data/TFRecords/'
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.count = 0
        self.shard_num = 0
        self.writer = tf.io.TFRecordWriter(self._tfr_name())

    def write(self, example):
        self.writer.write(example)
        self.count += 1
        if self.count % self.shard_size == 0:
            self.writer.close()
            self.shard_num += 1
            self.writer = tf.io.TFRecordWriter(self._tfr_name())

    def close(self):
        self.writer.close()

    def _tfr_name(self):
        return self.path + self.basename + '_{:04d}.tfr'.format(self.shard_num)


def serialized_example(kw_samples, kw_class):
    feature = {
        'kw_samples': _bytes_feature(tf.io.serialize_tensor(kw_samples)),
        'kw_class': _int64_feature(kw_class)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()


def main():
    class_dict = build_class_dict()
    filenames = get_all_filenames()
    random.seed(0)
    random.shuffle(filenames)
    writers = {x: MyTFRWriter(x) for x in ['train', 'val', 'test']}
    for filename in filenames:
        rawdata = tf.io.read_file(filename)
        kw_samples = tf.audio.decode_wav(rawdata).audio
        if kw_samples.shape[0] != 16000:
            continue  # only load examples with exactly 16ksamples (i.e. 1second)
        kw_class = get_class_int(filename, class_dict)
        kw_split = which_set(filename)
        example = serialized_example(kw_samples, kw_class)
        writers[kw_split].write(example)
    for writer in writers.values():
        writer.close()


if __name__ == "__main__":
    main()
