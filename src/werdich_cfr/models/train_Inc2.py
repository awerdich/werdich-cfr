import os
import glob
import pickle

import tensorflow as tf

# Custom imports
from werdich_cfr.tfutils.ModeltrainerInc2 import VideoTrainer
from werdich_cfr.tfutils.tfutils import use_gpu_devices

#%% GPU CONFIGURATION

physical_devices, device_list = use_gpu_devices(gpu_device_string='0,1')

#%% Some support functions

def write_model_dict(model_dict, file):
    with open(file, 'wb') as f:
        pickle.dump(model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved {}.'.format(file))
#%% Host name
import socket
print(socket.gethostname())

#%% Directories and data sets: For all models
cfr_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr')

run_dict_1 = {'model_name': }


# Model name
cfr_meta_date = '200425'
model_name = 'rest_global_'+'0503dgx1'
#model_name = 'test_inc2'

log_dir = os.path.join(cfr_dir, 'log', model_name)
tfr_data_dir = os.path.join(cfr_dir, 'tfr_'+cfr_meta_date, 'global')
train_files = sorted(glob.glob(os.path.join(tfr_data_dir, 'cfr_global_a4c_train_'+cfr_meta_date+'_*.tfrecords')))
eval_files = sorted(glob.glob(os.path.join(tfr_data_dir, 'cfr_global_a4c_eval_'+cfr_meta_date+'_*.tfrecords')))

# feature_dict
feature_dict_file = os.path.join(tfr_data_dir, 'global_pet_echo_dataset_200425.pkl')
with open(feature_dict_file, 'rb') as fl:
    feature_dict = pickle.load(fl)

#  ----- TESTING
#train_files = [eval_files[0]]
#eval_files = [eval_files[1]]

print('TRAIN:')
print(*train_files, sep='\n')
print('EVAL:')
print(*eval_files, sep='\n')

# Model parameters
model_dict = {'name': model_name,
              'im_size': (299, 299, 1),
              'im_scale_factor': 1.177,
              'min_rate': 21,
              'n_frames': 40,
              'filters': 64,
              'fc_nodes': 1,
              'model_output': 'rest_global_mbf',
              'kernel_init': tf.keras.initializers.GlorotNormal(),
              'bias_init': tf.keras.initializers.Zeros()}

print('model_output: {}'.format(model_dict['model_output']))

# Training parameters
train_dict = {'train_device_list': device_list,
              'learning_rate': 0.0001,
              'augment': False,
              'train_batch_size': 64,
              'eval_batch_size': 64,
              'validation_batches': None,
              'validation_freq': 1,
              'n_epochs': 150,
              'verbose': 1,
              'train_file_list': train_files,
              'eval_file_list': eval_files}

#%% Save model parameters and run the model

# Save model parameters before training
# Create the log dir, if it does not exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
model_dict_file = os.path.join(log_dir, model_name+'_model_dict.pkl')
train_dict_file = os.path.join(log_dir, model_name+'_train_dict.pkl')
write_model_dict(model_dict, model_dict_file)
write_model_dict(train_dict, train_dict_file)

# Compile the model
VT = VideoTrainer(log_dir=log_dir, model_dict=model_dict, train_dict=train_dict, feature_dict=feature_dict)
model=VT.compile_inc2model()
model.summary()

# Run the training and save the history data
hist=VT.train(model)
hist_file = os.path.join(log_dir, model_name+'_hist_dict.pkl')
write_model_dict(hist.history, hist_file)
