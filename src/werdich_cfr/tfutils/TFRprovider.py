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

    def create_tfr(self, filename, image_data, cfr_data,
                   rest_mbf_data, stress_mbf_data, record_data):
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
                rest_mbf = rest_mbf_data[i]
                stress_mbf = stress_mbf_data[i]
                idx = record_data[i]

                # Build example
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': self._wrap_bytes(im_bytes),
                    'shape': self._wrap_bytes(im_shape_bytes),
                    'cfr': self._wrap_float(cfr),
                    'rest_mbf': self._wrap_float(rest_mbf),
                    'stress_mbf': self._wrap_float(stress_mbf),
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
                 cfr_boundaries=(1.232, 1.556, 2.05),
                 output_height=299,
                 output_width=299,
                 im_scale_factor=None,
                 model_output='cfr'):

        self.cfr_boundaries = cfr_boundaries
        self.output_height = output_height
        self.output_width = output_width
        self.im_scale_factor = im_scale_factor
        self.model_output = model_output

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

    def _process_image(self, image, shape):

        # original shape is [height, width, frames] -> [frames, height, width, 1]
        # If there is no scale-factor, the images will be resized to fit
        image = tf.reshape(image, shape=shape)
        image = tf.transpose(image, perm=[2, 0, 1])
        image = tf.expand_dims(image, axis=-1)

        if self.im_scale_factor is None:
            image = tf.image.resize_with_pad(image,
                                             target_height=self.output_height,
                                             target_width=self.output_width)
        else:
            # Re-size the image with a single scale factor, then pad or crop to output_size
            im_size = tf.cast(tf.slice(shape, [0], [2]), dtype=tf.float32)
            new_im_size = tf.cast(tf.math.ceil(tf.math.scalar_mul(self.im_scale_factor, im_size)), tf.int32)
            image = tf.image.resize(image, size=new_im_size, antialias=True)
            # Crop or pad to the desired output size
            image = tf.image.resize_with_crop_or_pad(image,
                                                     target_height=self.output_height,
                                                     target_width=self.output_width)

        # Scale image to have mean 0 and variance 1
        image = tf.cast(image, tf.float32)
        image = tf.image.adjust_contrast(image, contrast_factor=5)
        output_image = tf.image.per_image_standardization(image)

        return output_image

    def _parse(self, serialized):

        example = {'image': tf.io.FixedLenFeature([], tf.string),
                   'shape': tf.io.FixedLenFeature([], tf.string),
                   'cfr': tf.io.FixedLenFeature([], tf.float32),
                   'rest_mbf': tf.io.FixedLenFeature([], tf.float32),
                   'stress_mbf': tf.io.FixedLenFeature([], tf.float32),
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
        rest_mbf = example['rest_mbf']
        stress_mbf = example['stress_mbf']
        record = example['record']

        # Create output tuple

        video_output = {'video': self._process_image(image, shape)}

        if self.model_output == 'cfr':
            score_output = {'score_output': cfr}
        elif self.model_output == 'rest_mbf':
            score_output = {'score_output': rest_mbf}
        elif self.model_output == 'stress_mbf':
            score_output = {'score_output': stress_mbf}
        else:
            # Enable all outputs for testing.
            score_output = {'class_output': self._cfr_label(cfr),
                            'cfr_output': cfr,
                            'mbf_output': rest_mbf,
                            'record': record}

        return (video_output, score_output)

    def make_batch(self, tfr_file_list, batch_size, shuffle,
                   buffer_n_batches=100, repeat_count=1, drop_remainder=False):

        # Shuffle data
        if shuffle:

            #n_parallel_calls = tf.data.experimental.AUTOTUNE

            files = tf.data.Dataset.list_files(tfr_file_list, shuffle = True)

            dataset = files.interleave(tf.data.TFRecordDataset,
                                       cycle_length = len(tfr_file_list),
                                       num_parallel_calls = None)

            dataset = dataset.shuffle(buffer_size = buffer_n_batches * batch_size,
                                      reshuffle_each_iteration = True)

        else:
            dataset = tf.data.TFRecordDataset(tfr_file_list)
            #n_parallel_calls = 1

        # Parse records
        dataset = dataset.map(map_func=self._parse, num_parallel_calls = None)

        # Batch it up
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

        # Prefetch
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset.repeat(count=repeat_count)
