import os
import gc
import glob
import pickle
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

# Custom imports
from werdich_cfr.tfutils.TFRprovider import DatasetProvider
from werdich_cfr.models.Inc2 import Inc2model
from werdich_cfr.tfutils.tfutils import use_gpu_devices

#%% Custom callbacks for information about training

class Gcallback(tf.keras.callbacks.Callback):
    """ Cleans memory after every epoch """
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

#%% Video trainer

class VideoTrainer:

    def __init__(self, log_dir, model_dict, train_dict):
        self.log_dir = log_dir
        self.model_dict = model_dict
        self.train_dict = train_dict
        self.Gcallback = Gcallback

    def create_dataset_provider(self):
        dataset_provider = DatasetProvider(output_height=self.model_dict['im_size'][0],
                                           output_width=self.model_dict['im_size'][1],
                                           im_scale_factor=self.model_dict['im_scale_factor'],
                                           model_output=self.model_dict['model_output'])
        return dataset_provider

    def count_steps_per_epoch(self, tfr_file_list, batch_size):
        """ Calculate the number of batches required to run one epoch
            We use the .parquet files to do this
        """
        # We assume, that the .parquet file with all training samples has the same name (except the extension)
        parquet_file_list = [file.replace('.tfrecords', '.parquet') for file in tfr_file_list]
        df = pd.concat([pd.read_parquet(file) for file in parquet_file_list], axis=0, ignore_index=True)
        n_records = len(df.filename.unique())
        steps_per_epoch = int(np.floor(n_records / batch_size)) + 1

        return steps_per_epoch

    def compile_inc2model(self):
        """ Set up the model with loss function, metrics, etc. """

        # Loss and accuracy metrics for each output
        loss = {'score_output': tf.keras.losses.MeanSquaredError()}

        #loss_weights = {'cfr_output': self.train_dict['loss_weight_cfr'],
        #               'mbf_output': self.train_dict['loss_weight_mbf']}

        metrics = {'score_output': tf.keras.metrics.MeanAbsolutePercentageError()}

        # Optimizer
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.train_dict['learning_rate'])

        # Build the model
        # mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=self.train_dict['train_device_list'])
        with mirrored_strategy.scope():
            model = Inc2model(model_dict=self.model_dict).video_encoder()
            model.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=metrics)
        return model

    def create_callbacks(self):
        """ Callbacks for model checkpoints and tensorboard visualizations """
        checkpoint_name = self.model_dict['name']+'_chkpt_{epoch:02d}'+'.h5'
        checkpoint_file = os.path.join(self.log_dir, checkpoint_name)

        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_file,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=False,
                                              save_freq='epoch')

        tensorboard_callback = TensorBoard(log_dir=self.log_dir,
                                           histogram_freq=1,
                                           write_graph=True,
                                           update_freq=10,
                                           profile_batch=0,
                                           embeddings_freq=0)

        callback_list = [checkpoint_callback, tensorboard_callback]

        return callback_list

    def train(self, model):
        """ Set up the training loop using model.fit """

        # Create datasets
        train_steps_per_epoch = self.count_steps_per_epoch(tfr_file_list=self.train_dict['train_file_list'],
                                                           batch_size=self.train_dict['train_batch_size'])

        dataset_provider = self.create_dataset_provider()
        train_set = dataset_provider.make_batch(tfr_file_list=self.train_dict['train_file_list'],
                                                batch_size=self.train_dict['train_batch_size'],
                                                shuffle=True,
                                                buffer_n_batches=self.train_dict['buffer_n_batches_train'],
                                                repeat_count=None,
                                                drop_remainder=True)

        eval_set = dataset_provider.make_batch(tfr_file_list=self.train_dict['eval_file_list'],
                                               batch_size=self.train_dict['eval_batch_size'],
                                               shuffle=False,
                                               buffer_n_batches=None,
                                               repeat_count=1,
                                               drop_remainder=True)

        hist = model.fit(x=train_set,
                         epochs=self.train_dict['n_epochs'],
                         verbose=self.train_dict['verbose'],
                         validation_data=eval_set,
                         initial_epoch=0,
                         steps_per_epoch=train_steps_per_epoch,
                         validation_steps=self.train_dict['validation_batches'],
                         validation_freq=self.train_dict['validation_freq'],
                         callbacks=self.create_callbacks())

        # After fit, save the model and weights
        model.save(os.path.join(self.log_dir, self.model_dict['name'] + '.h5'))

        return hist
