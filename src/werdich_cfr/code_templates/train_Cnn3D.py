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
from tensorflow_cfr.models.Cnn3Ddo import VideoNet as Cnn3Dnet

# https://www.tensorflow.org/alpha/guide/keras/training_and_evaluation

#%% Video Model Trainer Class

class VideoTrainer:

    def __init__(self,
                 data_root,
                 log_dir,
                 model_log_dir,
                 view,
                 model_dict,
                 train_dict,
                 cfr_boundaries):

        self.data_root = data_root
        self.log_dir = log_dir
        self.model_log_dir = model_log_dir
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

            # Save the model

        return model

    def train(self, model, n_epochs, train_dataset, eval_dataset, steps_per_epoch, model_name = 'Cnn3D'):

        # We need to provide steps_per_epoch because the system does not know
        # how many samples are in each .tfrecords file

        checkpoint_name = model_name + '_chkpt_{epoch:02d}' + '.hdf5'
        checkpoint_file = os.path.join(self.model_log_dir, checkpoint_name)

        # Callbacks
        checkpoint_callback = ModelCheckpoint(filepath = checkpoint_file,
                                              monitor = 'val_loss',
                                              verbose = 1,
                                              save_best_only = False,
                                              period = 2)

        tensorboard_callback = TensorBoard(log_dir = self.model_log_dir,
                                           histogram_freq = 5,
                                           write_graph = True,
                                           update_freq = 100)

        callback_list = [checkpoint_callback, tensorboard_callback]


        hist = model.fit(train_dataset,
                         epochs = n_epochs,
                         steps_per_epoch = steps_per_epoch,
                         validation_data = eval_dataset,
                         validation_steps = 50,
                         validation_freq = 1,
                         verbose = True,
                         callbacks = callback_list)

        # After fit, save the final model and weights
        model.save(os.path.join(self.model_log_dir, model_name + '.h5'))

        return model_log_dir, hist


#%% Run the training
data_root = os.path.normpath('/tf/data')
image_dir = os.path.normpath('/tf/imagedata')
log_dir = os.path.join(data_root, 'log')
cfr_boundaries = (1.355, 1.874)
train_batch_size = 10
eval_batch_size = 12
test_batch_size = 12
n_epochs = 70
view = 'a4c'
model_name = '3catw1cfrw9do_sp2_' + '_' + view
# All log files for this model go in here:
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
# We can resume training if we set the model_log_dir:
model_log_dir = os.path.join(log_dir, model_name + '_' + timestamp)

# NETWORK AND TRAINING PARAMETERS AS DICTIONARIES
model_dict = dict() # Model parameters (e.g. filters)
model_dict['name'] = model_name
model_dict['view'] = view
model_dict['im_size'] = (200, 200, 1)
model_dict['cl_outputs'] = len(cfr_boundaries)+1
model_dict['filters'] = 32
model_dict['pool_nodes'] = 256
model_dict['fc_nodes'] = 256
model_dict['fullnet'] = True
model_dict['cat_reg_outputs'] = True

# SMALL NETWORK FOR DEVELOPMENT DO NOT USE FOR REAL TRAINING!!!
#model_dict = dict() # Model parameters (e.g. filters)
#model_dict['name'] = model_name
#model_dict['view'] = view
#model_dict['im_size'] = (5, 5, 1)
#model_dict['cl_outputs'] = len(cfr_boundaries)+1
#model_dict['filters'] = 2
#model_dict['pool_nodes'] = 32
#model_dict['fc_nodes'] = 16
#model_dict['fullnet'] = False
#model_dict['cat_reg_outputs'] = True

train_dict = dict() # Training parameters (e.g. optimizer, learning rate)
train_dict['learning_rate'] = 0.0001
train_dict['loss_weights_class_output'] = 1.0 
train_dict['loss_weights_score_output'] = 9.0

# Compute global batch size using number of replicas.
#mirrored_strategy = tf.distribute.MirroredStrategy()
#replicas = mirrored_strategy.num_replicas_in_sync
#BATCH_SIZE_PER_REPLICA = train_batch_size
#global_batch_size = (BATCH_SIZE_PER_REPLICA * replicas)

# Adjust learning rate to multi-GPU trainingdock
#LEARNING_RATES_BY_GPU = {1: 0.0001, 8: 0.001}
#learning_rate = LEARNING_RATES_BY_GPU[replicas]

trainer = VideoTrainer(data_root = data_root,
                       log_dir = log_dir,
                       model_log_dir = model_log_dir,
                       view = view,
                       model_dict = model_dict,
                       train_dict = train_dict,
                       cfr_boundaries = cfr_boundaries)

# Set up the network
model = trainer.build_Cnn3Dnet()
model.summary()

# Load weights from checkpoint file
#chkpt = os.path.join(log_dir, '3catw1cfrw9do_sp2_a4c_201906010423', '3catw1cfrw9do_sp2_a4c_chkpt_30.hdf5')
#model.load_weights(chkpt)
#print('Continue training from checkpoint:', chkpt)

# Data sets
train_files = glob.glob(os.path.join(image_dir, 'train_' + view + '*' + '.tfrecords'))
eval_files = glob.glob(os.path.join(image_dir, 'eval_' + view + '*' + '.tfrecords'))
test_files = glob.glob(os.path.join(image_dir, 'test_' + view + '*' + '.tfrecords'))


n_train, train_set = trainer.build_datset(tfr_files = train_files,
                                          batch_size = train_batch_size,
                                          shuffle = True)

n_eval, eval_set = trainer.build_datset(tfr_files = eval_files,
                                        repeat_count = 1,
                                        batch_size = eval_batch_size,
                                        shuffle = True)

n_test, test_set = trainer.build_datset(tfr_files = test_files,
                                        repeat_count = 1,
                                        batch_size = test_batch_size,
                                        shuffle = False)

print('Train files:\n',*train_files, sep = '\n')
print('Eval files:\n', *eval_files, sep = '\n')
print('Test files:\n', *test_files, sep = '\n')

#%% Run the training

# Steps per epoch is the number of batches in all training record files
steps_per_epoch = int(np.ceil(n_train/train_batch_size))

print('TRAINING MODEL:', model_name)

model_log_dir, hist = trainer.train(model = model,
                                    n_epochs = n_epochs,
                                    train_dataset = train_set,
                                    eval_dataset = eval_set,
                                    steps_per_epoch = steps_per_epoch,
                                    model_name = model_name)

#%% Predict on the test set
# Load weights from latest checkpoint (lowest loss)
#model_log_dir = os.path.join(data_root, 'log', '3cat_cfr_a4c_201905270400')

print('Saving predictions from the latest checkpoint.')

# Load the latest checkpoint
checkpoint_files = sorted(glob.glob(os.path.join(model_log_dir, model_name + '_chkpt_*' + '.hdf5')))
chkpt_file = checkpoint_files[-1]

# Load the weights
model.load_weights(chkpt_file)

# Get the predictions from the model
class_model, score_model = model.predict(test_set)
class_preds = list(np.argmax(class_model, axis = 1))
score_preds = list(score_model.flatten())

# Get the true labels from the data set
n_test_batches = int(np.ceil(n_test/test_batch_size))
class_labels = []
score_labels = []
for _, labels_tf in test_set.take(n_test_batches+1):
    class_batch = list(np.argmax(labels_tf['class_output'].numpy(), axis = 1))
    score_batch = list(labels_tf['score_output'].numpy())
    class_labels.extend(class_batch)
    score_labels.extend(score_batch)

# Load also the labels from the csv files
test_csv_files = trainer.tfr2csv_file_list(test_files)
df = trainer.load_csv_info(test_csv_files)

labels_df = pd.DataFrame({'cfrtfr_labs': score_labels,
                          'class_labs': class_labels,
                          'score_preds': score_preds,
                          'class_preds': class_preds})

df_pred = pd.concat([df, labels_df], axis = 1)

# Save everything when done
# Model parameters
parameters = [model_dict, train_dict]
with open(os.path.join(model_log_dir, model_name+'.pickle'), 'wb') as f:
    pickle.dump(parameters, f, protocol = pickle.HIGHEST_PROTOCOL)
# Fit history
with open(os.path.join(model_log_dir, model_name+'_hist.pickle'), 'wb') as f:
    pickle.dump(hist.history, f, protocol = pickle.HIGHEST_PROTOCOL)
# Predictions on test set
df_pred.to_csv(os.path.join(model_log_dir, model_name + '_pred.csv'), index = False)