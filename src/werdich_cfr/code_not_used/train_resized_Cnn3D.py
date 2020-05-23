import os
import glob
import pickle

# Custom imports
from werdich_cfr.tfutils.Modeltrainer import VideoTrainer

#%% Some support functions

def write_model_dict(model_dict, file):
    with open(file, 'wb') as f:
        pickle.dump(model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved {}.'.format(file))

#%% Directories and data sets
tfr_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr/tfr_200208')
log_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr/log')
p_list = [1.259, 1.591, 2.066]

# Training and evaluation files
train_files = glob.glob(os.path.join(tfr_dir, 'cfr_resized_a4c_train_200208_*.tfrecords'))
eval_files = glob.glob(os.path.join(tfr_dir, 'cfr_resized_a4c_eval_200208_*.tfrecords'))

# GPUs
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

# Model name
model_name = '200208a4cresizedpad'
# Model directory
log_dir_model = os.path.join(log_dir, model_name)

# Model parameters
model_dict = {'name': model_name,
              'im_size': (299, 299, 1),
              'im_scale_factor': 0.5642,
              'n_frames': 40,
              'cfr_boundaries': p_list,
              'cl_outputs': len(p_list)+1,
              'filters': 32,
              'pool_nodes': 256,
              'fc_nodes': 256,
              'fullnet': True}

train_dict = {'learning_rate': 0.0001,
              'loss_weights_class_ouput': 1.0,
              'loss_weights_score_output': 9.0,
              'train_batch_size': 24,
              'eval_batch_size': 8,
              'validation_batches': None,
              'validation_freq': 1,
              'epochs': 100,
              'verbose': 1,
              'buffer_n_batches_train': 16,
              'train_file_list': train_files,
              'eval_file_list': eval_files}

# Save model parameters before training
model_dict_file = os.path.join(log_dir_model, model_name+'_model_dict.pkl')
train_dict_file = os.path.join(log_dir_model, model_name+'_train_dict.pkl')
write_model_dict(model_dict, model_dict_file)
write_model_dict(train_dict, train_dict_file)

# Run training
trainer = VideoTrainer(log_dir=log_dir_model,
                       model_dict=model_dict,
                       train_dict=train_dict)

convmodel = trainer.compile_convmodel()
convmodel.summary()

# Run the training and save the history data
hist = trainer.train(convmodel, train_files, eval_files)
hist_file = os.path.join(log_dir_model, model_name+'_hist_dict.pkl')
write_model_dict(hist.history, hist_file)
