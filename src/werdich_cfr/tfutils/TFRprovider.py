#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 23:24:38 2018

Some helper functions for processing noMNIST data sets

@author: andy
"""
#%% Imports
import os
import sys
import numpy as np
import tensorflow as tf
from pdb import set_trace
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocV3

#%% Functions and classes

class Dset:

    def __init__(self, data_root):
        self.data_root = data_root

    def create_tfr(self, filename, image_data, cfr_data, record_data):
        ''' Build a TFRecoreds data set from numpy arrays'''

        file = os.path.join(self.data_root, filename)

        with tf.io.TFRecordWriter(file) as writer:

            print('Converting:', filename)
            n_images = len(image_data)

            for i in range(n_images):

                # Print the percentage-progress.
                self._print_progress(count = i, total = n_images-1)

                im_bytes = image_data[i].astype(np.uint16).tobytes()
                im_shape_bytes = np.array(image_data[i].shape).astype(np.uint16).tobytes()
                cfr = cfr_data[i]
                idx = record_data[i]

                # Build example
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': self._wrap_bytes(im_bytes),
                    'shape': self._wrap_bytes(im_shape_bytes),
                    'cfr': self._wrap_float(cfr),
                    'record': self._wrap_int64(idx)}))

                # Serialize example
                serialized = example.SerializeToString()

                # Write example to disk
                writer.write(serialized)

    def list_tfr(self, data_path, tfrext = '.tfrecords'):
        ''' Return a list of TFRecords data files in path'''

        file_list = os.listdir(data_path)
        tfr_list = [os.path.join(data_path, f) for f in file_list if os.path.splitext(f)[-1] == tfrext]

        return tfr_list

    # Integer data (labels)
    def _wrap_int64(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # Byte data (images, arrays)
    def _wrap_bytes(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _wrap_float(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    # Progress update
    def _print_progress(self, count, total):
        pct_complete = float(count) / total

        # Status-message.
        # Note the \r which means the line should overwrite itself.
        msg = "\r- Progress: {0:.1%}".format(pct_complete)

        # Print it.
        sys.stdout.write(msg)
        sys.stdout.flush()

class DatasetProvider:
    ''' Creates a dataset from a list of .tfrecords files.'''

    def __init__(self,
                 tfr_file_list,
                 repeat_count=None,
                 n_frames=30,
                 cfr_boundaries=(1.232, 1.556, 2.05),
                 output_height=299,
                 output_width=299,
                 record_output=False):

        self.tfr_file_list = tfr_file_list
        self.repeat_count = repeat_count
        self.n_frames = n_frames
        self.cfr_boundaries = cfr_boundaries
        self.output_height = output_height
        self.output_width = output_width
        self.record_output = record_output

    @tf.function
    def _cfr_label(self, cfr_value):
        ''' classification label for cfr value '''
        percentile_list = self.cfr_boundaries
        label = 0
        if cfr_value < percentile_list[0]:
            label = 0
        elif cfr_value >= percentile_list[-1]:
            label = len(percentile_list)
        for p in range(1, len(percentile_list)):
            if (cfr_value >= percentile_list[p - 1]) & (cfr_value < percentile_list[p]):
                label = p
        return tf.one_hot(label, depth = len(percentile_list)+1)

    def _process_image(self, image):

        # Original video shape in TFR is [sample, height, width, frames]
        # First, we resize the images. Channels should stay unaffected

        #image = tf.image.resize_with_pad(image,
        #                                 target_height = self.output_height,
        #                                 target_width = self.output_width,
        #                                 antialias = True)

        # We can crop and then resize the images, but that would potentially cut off some information
        # Also, this seems to cause problems with distribution strategy
        image = tf.image.resize_with_crop_or_pad(image, target_height = 500, target_width = 500)
        # Then, resize the whole thing
        image = tf.image.resize(image, size = (self.output_height, self.output_width))

        # Now we need to reshape the image batch as [frames, height, width, channels]
        # video = layers.Input(shape=(30, 299, 299, 3), name='video input vector')
        # Set the shape and transpose so that time steps are first
        image = tf.reshape(image, shape=(self.output_height, self.output_width, self.n_frames))
        image = tf.transpose(image, perm = [2, 0, 1])

        # Add a fourth dimension for RGB values
        image = tf.expand_dims(image, axis = -1)

        # Convert to float
        image = tf.cast(image, tf.float64)

        # Linearly scale images to have zero mean and unit std
        image = tf.image.adjust_contrast(image, contrast_factor=2)
        output_image = tf.image.per_image_standardization(image)

        return output_image

    def _parse(self, serialized):

        example = {'image': tf.io.FixedLenFeature([], tf.string),
                   'shape': tf.io.FixedLenFeature([], tf.string),
                   'cfr': tf.io.FixedLenFeature([], tf.float32),
                   'record': tf.io.FixedLenFeature([], tf.int64)}

        # Extract example from the data record
        example = tf.io.parse_single_example(serialized, example)

        # Convert image to tensor and shape it
        image_raw = tf.io.decode_raw(example['image'], tf.uint16)
        shape = tf.io.decode_raw(example['shape'], tf.uint16)
        shape = tf.cast(shape, tf.int32) # tf.reshape requires int16 or int32 types
        image = tf.reshape(image_raw, shape)

        # Here, we have recovered the original shape of the images.
        # Now we need to process them.

        cfr = example['cfr']
        record = example['record']

        # categorical and regression outputs (tuple of dicts)
        if self.record_output:
            # Add record output for testing only (additional output gives an error during training)
            outputs = ({'video': self._process_image(image)},
                       {'class_output': self._cfr_label(cfr),
                        'score_output': cfr},
                       {'record': record})
        else:
            # For training, use only model input/outputs
            outputs = ({'video': self._process_image(image)},
                       {'class_output': self._cfr_label(cfr),
                        'score_output': cfr})
        return outputs

    def make_batch(self, batch_size, shuffle):

        # Shuffle data
        if shuffle:

            files = tf.data.Dataset.list_files(self.tfr_file_list, shuffle = True)

            dataset = files.interleave(tf.data.TFRecordDataset,
                                       cycle_length = len(self.tfr_file_list),
                                       num_parallel_calls = tf.data.experimental.AUTOTUNE)

            dataset = dataset.shuffle(buffer_size = 100 * batch_size,
                                      reshuffle_each_iteration = True)

            n_parallel_calls = tf.data.experimental.AUTOTUNE

        else:
            dataset = tf.data.TFRecordDataset(self.tfr_file_list)
            n_parallel_calls = 1

        # Parse records
        dataset = dataset.map(map_func = self._parse, num_parallel_calls = n_parallel_calls)

        # Batch it up
        dataset = dataset.batch(batch_size)

        # Prefetch
        dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

        return dataset.repeat(count = self.repeat_count)
