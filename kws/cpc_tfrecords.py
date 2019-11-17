#! /usr/bin/env python
"""
Encode audio sample data using learned Contrastive Predictive Coding.

See this repo: https://github.com/pat-coady/contrast-pred-code

Input: `TFRecords/samples/*.tfr`, Output `TFRecords/cpc-enc/*.tfr`
"""
import tensorflow as tf
import dataset
from pathlib import Path
import numpy as np

from genc import genc_model


# from: https://www.tensorflow.org/alpha/tutorials/load_data/tf_records
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class MyTFRWriter(object):
    def __init__(self, mode, ds_type, shard_size=256):
        self.mode = mode
        self.shard_size = shard_size
        path = Path.home() / 'Data' / 'kws' / 'tfrecords' / ds_type
        path.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
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
        tfr_name = Path(self.path) / '{}_{:04d}.tfr'.format(self.mode, self.shard_num)
        return str(tfr_name)


def serialized_example(x, y):
    feature = {
        'x': _bytes_feature(tf.io.serialize_tensor(x)),
        'y': _int64_feature(y)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()


def main():
    dim_z = 40  # dimension of latent embedding space
    genc = genc_model(dim_z=dim_z)
    zeros = np.zeros((64, 16000), dtype=np.float32)
    genc(zeros)  # send dummy batch through model to force Model.build()
    genc.load_weights(str(Path.cwd() / 'genc.h5'))
    config = {'batch_sz': 64,
              'ds_type': 'samples',
              }
    for mode in ['train', 'val', 'test']:
        writer = MyTFRWriter(mode, 'cpc-enc')
        ds = dataset.build_dataset(config, mode=mode)
        for batch in ds:
            x, y = batch
            x = genc(x)  # shape = (batch_sz, 63, dim_z)
            for idx in range(x.shape[0]):
                example = serialized_example(x[idx, ...], y[idx])
                writer.write(example)
        writer.close()


if __name__ == "__main__":
    main()
