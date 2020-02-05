import os
import glob
import pandas as pd

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

import tensorflow as tf
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

    def build_dataset(self, tfr_files, batch_size, repeat_count = None, shuffle = False):
        """ Create TFR dataset object as input to the network
        """
        dset_provider = DatasetProvider(tfr_files,
                                        repeat_count = None,
                                        n_frames = self.model_dict['n_frames'],
                                        cfr_boundaries = self.model_dict['cfr_boundaries'],
                                        output_height = self.model_dict['im_size'][0],
                                        output_width = self.model_dict['im_size'][1])

        dataset = dset_provider.make_batch(batch_size = batch_size, shuffle = shuffle)

        # We need steps_per_epoch: number of samples in tfr_files. We can use the .parquet files
        parquet_files = [file.split('.')[0]+'.parquet' for file in tfr_files]
        df = pd.concat([pd.read_parquet(file) for file in parquet_files], axis=0, ignore_index=True)
        n_records = len(df.filename.unique())

        return n_records, dataset

    def compile_convmodel(self):
        """ Set up the model with loss functions, metrics, etc
        arg: Cnn3D.Convmodel
        """
        model = Convmodel(model_dict = self.model_dict).video_encoder()
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.train_dict['learning_rate'])

        loss = {'class_output': tf.keras.losses.CategoricalCrossentropy(),
                'score_output': tf.keras.losses.MeanSquaredError()}

        loss_weights = {'class_output': train_dict['loss_weights_class_ouput'],
                        'score_output': train_dict['loss_weights_score_output']}

        metrics = {'class_output': tf.keras.metrics.CategoricalAccuracy(),
                   'score_output': tf.keras.metrics.MeanAbsolutePercentageError()}

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics,
                      loss_weights=loss_weights)

        return model

    def train(self, model, train_tfr_files, eval_tfr_files):

        pass

#%% Test parameters
tfr_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr/tfr_200202')
log_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr/log')
p_list = [1.247, 1.583, 2.075]
im_size = (299, 299, 1)
n_frames = 30

# Model parameters
model_dict = {'name': 'testmodel_a4c',
              'im_size': (299, 299, 1),
              'n_frames': 30,
              'cfr_boundaries': p_list,
              'cl_outputs': len(p_list)+1,
              'filters': 32,
              'pool_nodes': 256,
              'fc_nodes': 256,
              'fullnet': True}

train_dict = {'learning_rate': 0.0001,
              'loss_weights_class_ouput': 1.0,
              'loss_weights_score_output': 9.0}

trainer = VideoTrainer(log_dir=log_dir,
                       model_dict=model_dict,
                       train_dict=train_dict)

train_files = sorted(glob.glob(os.path.join(tfr_dir, 'CFR_200202_view_a4c_train_*.tfrecords')))
n_train, train_set = trainer.build_dataset(train_files, 8, shuffle = False)
model = trainer.compile_convmodel()
