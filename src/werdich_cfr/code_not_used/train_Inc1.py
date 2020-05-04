import os
import glob
import pickle

import tensorflow as tf

# Custom imports
from werdich_cfr.tfutils.ModeltrainerInc1 import VideoTrainer

#%% GPU CONFIGURATION
# GPU
use_device_string = '0,1'
use_device_idx = list(range(len(use_device_string.split(','))))

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = use_device_string

physical_devices = tf.config.list_physical_devices('GPU')
device_list = [physical_devices[idx].name.replace('physical_device:', '') for idx in use_device_idx]

print('AVAILABLE GPUs:')
print(*physical_devices, sep='\n')
print('TRAIN DEVICE LIST:')
print(*device_list, sep='\n')

try:
  for dev in physical_devices:
    tf.config.experimental.set_memory_growth(dev, True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


#%% Some support functions

def write_model_dict(model_dict, file):
    with open(file, 'wb') as f:
        pickle.dump(model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved {}.'.format(file))

#%% Directories and data sets

# Model name
cfr_meta_date = '200304'
model_name = 'cfr'+cfr_meta_date+'gpu4'
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
              'fc_nodes': 128,
              'kernel_init': tf.keras.initializers.GlorotNormal(),
              'bias_init': tf.keras.initializers.Zeros()}

# Training parameters
train_dict = {'train_device_list': device_list,
              'learning_rate': 0.0001,
              'train_batch_size': 20,
              'eval_batch_size': 20,
              'validation_batches': None,
              'validation_freq': 1,
              'n_epochs': 100,
              'verbose': 1,
              'buffer_n_batches_train': 8,
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
model=VT.compile_inc1model()
model.summary()

# Run the training and save the history data
hist=VT.train(model)
hist_file = os.path.join(log_dir, model_name+'_hist_dict.pkl')
write_model_dict(hist.history, hist_file)
