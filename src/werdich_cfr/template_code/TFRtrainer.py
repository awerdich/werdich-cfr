# imports
import os
import glob
import pickle
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
from pdb import set_trace
import tensorflow as tf
#tf.enable_eager_execution()
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

print('TensorFlow Version:', tf.__version__)

from tensorflow_cfr.tfutils.TFRprovider import DatasetProvider
from tensorflow_cfr.models.Cnn3D import VideoNet as Cnn3Dnet

# https://www.tensorflow.org/alpha/guide/keras/training_and_evaluation

#%% Video Model Trainer Class

class VideoTrainer:

    def __init__(self,
                 data_root,
                 log_dir,
                 view,
                 model_dict,
                 train_dict,
                 cfr_boundaries):

        self.data_root = data_root
        self.log_dir = log_dir
        self.view = view
        self.cfr_boundaries = cfr_boundaries

        # MODEL PARAMETERS
        self.im_size = model_dict['im_size']
        self.cat_reg_outputs = model_dict['cat_reg_outputs']
        self.model_dict = model_dict

        # TRAINING PARAMTERS
        self.learning_rate = train_dict['learning_rate']
        #loss_weights = {'class_output': 2.0, 'score_output': 1.0}
        self.loss_weights = {'class_output': train_dict['loss_weights_class_output'], 
                             'score_output': train_dict['loss_weights_score_output']}

    def tfr2csv_file_list(self, tfr_files):
        ''' Converts .tfrecords file paths into .csv file paths'''

        # get the path and construct the file name
        fdir = lambda d: os.path.dirname(d)
        cname = lambda x: os.path.splitext(os.path.basename(x))[0] + '.csv'
        csv_files = [os.path.join(fdir(f), cname(f)) for f in tfr_files]

        # Make sure, those files actually exist
        file_list = [file for file in csv_files if os.path.exists(file)]

        # Warning if there are fewer .csv files
        if len(file_list) < len(tfr_files):
            print('Warning. Found fewer .csv files:', len(file_list))
            print('steps_per_epoch might be incorrect.')

        return file_list

    def load_csv_info(self, csv_files):
        ''' csv_files: list of .csv files matched with .tfrecords names '''
        df = pd.DataFrame()
        for file in csv_files:
            df_file = pd.read_csv(file)
            df = pd.concat([df, df_file], ignore_index = True)
        return df

    def build_datset(self, tfr_files, batch_size, repeat_count = None, shuffle = False):
        '''tfr_files : list of tfr file paths (complete paths) '''

        # Number of samples in each file list and number of timesteps
        df_csv = self.load_csv_info(self.tfr2csv_file_list(tfr_files))
        n_records = df_csv.shape[0]
        n_frames = df_csv.iloc[0].frames

        # Infer rgb from im_size
        rgb = False
        if self.im_size[2] == 3: rgb = True

        dset_provider = DatasetProvider(tfr_files,
                                        repeat_count = repeat_count,
                                        n_frames = n_frames,
                                        cfr_boundaries = self.cfr_boundaries,
                                        output_height = self.im_size[0],
                                        output_width = self.im_size[1],
                                        cat_reg_outputs = self.cat_reg_outputs,
                                        rgb = rgb)

        dataset = dset_provider.make_batch(batch_size = batch_size, shuffle = shuffle)

        return n_records, dataset

    def build_Cnn3Dnet(self):
        ''' Build network. Parmameters are in parms_dict
        parms_dict['filters'] Number of filter kernels for Conv3D layers
        parms_dict['fc_nodes'] Number of nodes in fc layers '''

        model_init = Cnn3Dnet(model_dict = self.model_dict)
        model = model_init.video_encoder()
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        if self.cat_reg_outputs:

            loss = {'class_output': tf.keras.losses.CategoricalCrossentropy(),
                    'score_output': tf.keras.losses.MeanSquaredError()}
            loss_weights = self.loss_weights
            metrics = {'class_output': tf.keras.metrics.CategoricalAccuracy(),
                       'score_output': tf.keras.metrics.MeanAbsolutePercentageError()}
            model.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=metrics,
                          loss_weights = loss_weights)
        else:
            loss = tf.keras.CategoricalCrossentropy()
            metrics = tf.keras.metrics.CategoricalAccuracy()
            model.compile(loss = loss,
                          optimizer = optimizer,
                          metrics = metrics)

        return model

    def train(self, model, n_epochs, train_dataset, eval_dataset, steps_per_epoch, model_name = 'Cnn3D'):

        # We need to provide steps_per_epoch because the system does not know
        # how many samples are in each .tfrecords file

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")

        # All log files for this model go in here:
        model_log_dir = os.path.join(self.log_dir, model_name + '_' + timestamp)

        checkpoint_name = model_name + '_chkpt_{epoch:02d}' + '.hdf5'
        checkpoint_file = os.path.join(model_log_dir, checkpoint_name)

        # Callbacks
        checkpoint_callback = ModelCheckpoint(filepath = checkpoint_file,
                                              monitor = 'val_loss',
                                              verbose = 1,
                                              save_best_only = False,
                                              mode = 'min',
                                              period = 2)

        tensorboard_callback = TensorBoard(log_dir = model_log_dir,
                                           histogram_freq = 5,
                                           write_graph = True,
                                           update_freq = 100)

        hist = model.fit(train_dataset,
                         epochs = n_epochs,
                         steps_per_epoch = steps_per_epoch,
                         validation_data = eval_dataset,
                         validation_steps = None,
                         validation_freq = 1,
                         verbose = True,
                         callbacks=[checkpoint_callback,
                                    tensorboard_callback])
        return model_log_dir, hist

    def save_model(self, model_log_dir, model_name, model):
        print('Saving model architecture and weights.')
        json_string = model.to_json()
        open(os.path.join(model_log_dir, model_name+'.json'), 'w').write(json_string)
        yaml_string = model.to_yaml()
        open(os.path.join(model_log_dir, model_name+'.yaml'), 'w').write(yaml_string)
        model.save_weights(os.path.join(model_log_dir, model_name+'_weights.hdf5'))
