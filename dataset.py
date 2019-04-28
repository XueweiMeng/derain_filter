import os

import cv2
import numpy as np
import tensorflow as tf


class Dataset(object):
    """Represent input dataset for Derain model."""

    def __init__(self, split_name, dataset_dir, batch_size, 
                 shuffle=False, repeat=False):
        """Initializes the dataset.
        
        Args:
            split_name: A train/val split name.
            dataset_dir: The directory of the dataset sources.
            batch_size: Batch size.
            shuffle: Boolean, if should shuffle the input data.
            repeat: Boolean, if should repeat the input data.
        """
        self.split_name = split_name
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.repeat = repeat

    def get_data_iterator(self):
        """Get an iterator that iterates across the dataset once.
        
        Returns:
            An iterator of type tf.data.Iterator
        """
        dataset = tf.data.TFRecordDataset(self.dataset_dir).map(
            self._parse_function)

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=100)

        if self.repeat:
            dataset = dataset.repeat()
        else:
            dataset = dataset.repeat(1)

        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_initializable_iterator()
        return iterator

    def _parse_function(self, proto):
        """Function to parse the proto.
        
        Args:
            proto: Proto in the format of tf.Example.

        Returns:
            Sample with paired O and B.
        """
        features = {
            'O': tf.FixedLenFeature((),  tf.string, default_value=''),
            'B': tf.FixedLenFeature((), tf.string, default_value=''),
            'height': tf.FixedLenFeature((), tf.int64, default_value=0),
            'width': tf.FixedLenFeature((), tf.int64, default_value=0),
        }

        parsed_features = tf.parse_single_example(proto, features)

        shape = tf.stack((
            parsed_features['height'], parsed_features['width'], 3))
        O = tf.decode_raw(parsed_features['O'], tf.uint8)
        B = tf.decode_raw(parsed_features['B'], tf.uint8)

        sample = [
            tf.reshape(tf.cast(O, tf.float32), shape) / 255.0,
            tf.reshape(tf.cast(B, tf.float32), shape) / 255.0,
        ]
        return sample
