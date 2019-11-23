# Speech Command Classification

### Overview

This code implements a TensorFlow training pipeline for the [Google Speech Command Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) and establishes the baseline performance using a "vanilla" CNN. The motivation is to have an MNIST-like dataset, but for an audio classification task. The speech command dataset contains ~2,000 x 1 second recordings for each of 30 speech commands.

**NOTE:** I just noticed a [pull request](https://github.com/tensorflow/datasets/pull/992) in the [tensorflow datasets GitHub repo](https://github.com/tensorflow/datasets) to add the speech command dataset. When that is merged, that will be a clean way to use this dataset in TensorFlow. 

#### Data Preparation (`tfrecords.py`)

I've gotten into the habit of parsing my data into TFRecord files. It is a little extra work at the front-end, but it generally pays off in training speed and traing pipeline simplicity.

Download the `tar.gz` from the above link. By default, `tfrecords.py` expects the file to be extracted to: `~/Data/speech_commands_v0.01`. The default location for the processed TFRecord files is under`~/Data/speech_commands/tfrecord`.

The script has an option to process the data in 3 ways (selected by `--ds_type` option):

- `samples` : no processing, audio samples as found in the `*.wav` files
- `log-mel` : log-mel spectrogram with 40 bins, 512 sample frame length, 256 sample overlap
  - each example has shape = (61, 40, 1)
- `mfcc` : mel-frequency cepstral coefficients, same settings at log-mel

**Note:** The original speech command challenge used only 10 of the words from the dataset plus "silence" and "unknown" classes. Even though this is a smaller number of words, it is actually a more challenging task with a catch-all unknown class. 

### Default Model (`model.py`)

| layer                | kernel        | output shape |
| -------------------- | ------------- | ------------ |
| input                |               | 61x40x1      |
| conv-batch norm-relu | 3x3x16        | 61x40x16     |
| conv-batch norm-relu | 3x3x16        | 61x40x16     |
| maxpool              | 2x2, stride=2 | 30x20x16     |
| conv-batch norm-relu | 3x3x32        | 30x20x32     |
| conv-batch norm-relu | 3x3x32        | 30x20x32     |
| maxpool              | 2x2, stride=2 | 15x10x32     |
| conv-batch norm-relu | 3x3x64        | 15x10x64     |
| conv-batch norm      | 3x3x30        | 15x10x30     |
| global average pool  |               | 30           |

#### Command Line Options (`train.py`)

Use `-h` to see current options. Note, some binary settings use 1 or 0 to keep the Azure ML hyperparameter search tool happy (instead of a more intuitive command line flag).

Options include: learning rate, momentum, epochs, batch size, data directory path, data set type (see below), optimizer (adam or plain SGD), batch norm, and global average pool vs. dense final layer.

##### `--ds_type`

There are 3 options here:

`log-mel` : log-mel spectrogram 61x40

`cpc-enc` : data encoded using layer trained with [contrastive predictive coding](https://github.com/pat-coady/contrast-pred-code), 63x40

`samples` : raw audio samples, 16ksamples

### Results

Here are the results using default settings. Learning rate decay in active by default, with learning decreasing by 3x at 10, 15, and 20 epochs. The model is trained for a total of 25 epochs.

| Representation      | Train Accuracy | Validation Accuracy |
| ------------------- | -------------- | ------------------- |
| log-mel spectrogram | 99.0%          | 95.2%               |

Requirements

See `setup.py`.

Note: `azureml` is optional, only needed if you train using Azure ML Workspaces.

### References

1. [Google Speech Command Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) (Pete Warden)