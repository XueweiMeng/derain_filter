import argparse
import os

import tensorflow as tf
from tqdm import tqdm

import dataset

BATCH_SIZE = 64
CHANNEL_NUM = 24
NET_DEPTH = 7
DATASET_PATH = './datasets/test_Rain100H/dataset.tfrecords'
BUFFER_SIZE = 3000
OPTIMIZER='adam'
LEARNING_RATE = 5e-3
NUM_STEPS = 20000
SAVE_NUM = 4
STEPS_PER_LOG = 100
STEPS_PER_SAVE = 1000
LOGDIR = './logdir/train'


def main():
    """Main entry for training process."""
    train_dt = dataset.Dataset('train', DATASET_PATH, BATCH_SIZE,
                               shuffle=True, repeat=True)
    data_iter = train_dt.get_data_iterator()

    with tf.Session() as sess:
        O, B = data_iter.get_next()
        sess.run(data_iter.initializer)
        O, B = sess.run([O, B])
        print(O.shape)
        print(B.shape)


if __name__ == '__main__':
    main()
