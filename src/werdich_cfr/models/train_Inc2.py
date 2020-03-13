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

#%% Directories and data sets

# Model name
cfr_meta_date = '200304'
model_name = 'meta'+cfr_meta_date+'_restmbf_aug_'+'0313gpu2'
#model_name = 'meta'+cfr_meta_date+'_testmodel'
cfr_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr')
log_dir = os.path.join(cfr_dir, 'log', model_name)
tfr_data_dir = os.path.join(cfr_dir, 'tfr_'+cfr_meta_date)
train_files = sorted(glob.glob(os.path.join(tfr_data_dir, 'cfr_resized75_a4c_train_'+cfr_meta_date+'_*.tfrecords')))
eval_files = sorted(glob.glob(os.path.join(tfr_data_dir, 'cfr_resized75_a4c_eval_'+cfr_meta_date+'_*.tfrecords')))

print('TRAIN:')
print(*train_files, sep='\n')
print(*eval_files, sep='\n')

# Model parameters
model_dict = {'name': model_name,
              'im_size': (299, 299, 1),
              'im_scale_factor': 1.177,
              'n_frames': 40,
              'filters': 64,
              'fc_nodes': 1,
              'model_output': 'rest_mbf',
              'kernel_init': tf.keras.initializers.GlorotNormal(),
              'bias_init': tf.keras.initializers.Zeros()}

print('model_output: {}'.format(model_dict['model_output']))

# Training parameters
train_dict = {'train_device_list': device_list,
              'learning_rate': 0.0001,
              'augment': True,
              'train_batch_size': 18,
              'eval_batch_size': 18,
              'validation_batches': None,
              'validation_freq': 1,
              'n_epochs': 500,
              'verbose': 1,
              'buffer_n_batches_train': 4,
              'train_file_list': train_files,
              'eval_file_list': eval_files}

# Save model parameters before training
# Create the log dir, if it does not exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
model_dict_file = os.path.join(log_dir, model_name+'_model_dict.pkl')
train_dict_file = os.path.join(log_dir, model_name+'_train_dict.pkl')
write_model_dict(model_dict, model_dict_file)
write_model_dict(train_dict, train_dict_file)

# Compile the model
VT = VideoTrainer(log_dir=log_dir, model_dict=model_dict, train_dict=train_dict)
model=VT.compile_inc2model()
model.summary()

# Run the training and save the history data
hist=VT.train(model)
hist_file = os.path.join(log_dir, model_name+'_hist_dict.pkl')
write_model_dict(hist.history, hist_file)
