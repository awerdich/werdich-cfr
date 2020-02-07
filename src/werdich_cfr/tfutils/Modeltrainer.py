import os
import glob
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

print('TensorFlow Version:', tf.__version__)

# Custom imports
from werdich_cfr.tfutils.TFRprovider import DatasetProvider
from werdich_cfr.models.Cnn3D import Convmodel

#%% Video trainer

class VideoTrainer:

    def __init__(self, log_dir, model_dict, train_dict):
        self.log_dir = log_dir
        self.model_dict = model_dict # Model parameter
        self.train_dict = train_dict # Training parameter

    def build_dataset(self, tfr_files, batch_size, buffer_n_batches, repeat_count = None, shuffle = False):
        """ Create TFR dataset object as input to the network
        """
        dset_provider = DatasetProvider(tfr_files,
                                        repeat_count=repeat_count,
                                        n_frames=self.model_dict['n_frames'],
                                        cfr_boundaries=self.model_dict['cfr_boundaries'],
                                        output_height=self.model_dict['im_size'][0],
                                        output_width=self.model_dict['im_size'][1],
                                        im_resize_crop = self.model_dict['im_resize_crop'])

        dataset = dset_provider.make_batch(batch_size=batch_size,
                                           shuffle=shuffle,
                                           buffer_n_batches=buffer_n_batches)

        # We need steps_per_epoch: number of samples in tfr_files. We can use the .parquet files
        parquet_files = [file.split('.')[0]+'.parquet' for file in tfr_files]
        df = pd.concat([pd.read_parquet(file) for file in parquet_files], axis=0, ignore_index=True)
        n_records = len(df.filename.unique())

        return n_records, dataset

    def compile_convmodel(self):
        """ Set up the model with loss functions, metrics, etc
        arg: Cnn3D.Convmodel
        """

        # Define loss, metrics and optimizer

        loss = {'class_output': tf.keras.losses.CategoricalCrossentropy(),
                'score_output': tf.keras.losses.MeanSquaredError()}

        loss_weights = {'class_output': self.train_dict['loss_weights_class_ouput'],
                        'score_output': self.train_dict['loss_weights_score_output']}

        metrics = {'class_output': tf.keras.metrics.CategoricalAccuracy(),
                   'score_output': tf.keras.metrics.MeanAbsolutePercentageError()}

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.train_dict['learning_rate'])

        # Build the model
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():

            model = Convmodel(model_dict = self.model_dict).video_encoder()
            model.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=metrics,
                          loss_weights=loss_weights)

        return model

    def create_callbacks(self):
        """ Callbacks for model checkpoints and tensorboard visualizations """
        checkpoint_name = self.model_dict['name']+'_chkpt_{epoch:02d}'+'.hdf5'
        checkpoint_file = os.path.join(self.log_dir, checkpoint_name)

        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_file,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=False,
                                              save_freq='epoch')

        tensorboard_callback = TensorBoard(log_dir=self.log_dir,
                                           histogram_freq=1,
                                           write_graph=True,
                                           update_freq=100,
                                           profile_batch=0,
                                           embeddings_freq=0)

        callback_list = [checkpoint_callback, tensorboard_callback]

        return callback_list

    def train(self, model, train_tfr_files, eval_tfr_files):

        n_train, train_set = self.build_dataset(train_tfr_files,
                                                batch_size=self.train_dict['train_batch_size'],
                                                buffer_n_batches=self.train_dict['buffer_n_batches_train'],
                                                repeat_count=None,
                                                shuffle=True)

        steps_per_epoch_train = int(np.floor(n_train/self.train_dict['train_batch_size']))

        n_eval, eval_set = self.build_dataset(eval_tfr_files,
                                              batch_size=self.train_dict['eval_batch_size'],
                                              buffer_n_batches=self.train_dict['buffer_n_batches_train'],
                                              repeat_count = None,
                                              shuffle = True)

        hist = model.fit(x=train_set,
                         epochs=self.train_dict['epochs'],
                         verbose=self.train_dict['verbose'],
                         validation_data=eval_set,
                         initial_epoch=0,
                         steps_per_epoch=steps_per_epoch_train,
                         validation_steps=self.train_dict['validation_batches'],
                         validation_freq=self.train_dict['validation_freq'],
                         callbacks=self.create_callbacks())

        # After fit, save the model and weights
        model.save(os.path.join(self.log_dir, self.model_dict['name']+'.h5'))

        return hist
