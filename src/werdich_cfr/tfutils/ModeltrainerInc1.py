import os
import glob
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

# Custom imports
from werdich_cfr.tfutils.TFRprovider import DatasetProvider
from werdich_cfr.models.Inc1 import Inc1model

#%% Video trainer

class VideoTrainer:

    def __init__(self, log_dir, model_dict, train_dict):
        self.log_dir = log_dir
        self.model_dict = model_dict # Model parameter
        self.train_dict = train_dict # Training parameter

    def create_dataset_provider(self):
        dataset_provider = DatasetProvider(output_height=model_dict['im_size'][0],
                                           output_width=model_dict['im_size'][1],
                                           im_scale_factor=model_dict['im_scale_factor'],
                                           model_outputs=True)
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

    def compile_inc1model(self):
        """ Set up the model with loss function, metrics, etc. """

        # Loss and accuracy metrics for each output
        loss={'score_output': tf.keras.losses.MeanSquaredError()}
        metrics={'score_output': tf.keras.metrics.MeanAbsolutePercentageError()}

        # Optimizer
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=train_dict['learning_rate'])

        # Build the model
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = Inc1model(model_dict=self.model_dict).video_encoder()
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
                                           update_freq=100,
                                           profile_batch=0,
                                           embeddings_freq=0)

        callback_list = [checkpoint_callback, tensorboard_callback]

        return callback_list

    def train(self, model):
        """ Set up the training loop using model.fit """

        # Create datasets
        train_steps_per_epoch = self.count_steps_per_epoch(tfr_file_list=self.train_dict['train_file_list'],
                                                           batch_size=self.train_dict['train_batch_size'])

        dataset_provider=self.create_dataset_provider()
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
                                                drop_remainder=False)

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
