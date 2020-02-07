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
tfr_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr/tfr_200202')
log_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr/log')
p_list = [1.247, 1.583, 2.075]

# Training and evaluation files
train_files = glob.glob(os.path.join(tfr_dir, 'CFR_200202_view_a4c_train_*.tfrecords'))
eval_files = glob.glob(os.path.join(tfr_dir, 'CFR_200202_view_a4c_eval_*.tfrecords'))

# Model name
base_name = '20020a4c'
model_name = base_name + '_pad'

# Model parameters
model_dict = {'name': model_name,
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
              'loss_weights_score_output': 9.0,
              'train_batch_size': 20,
              'eval_batch_size': 20,
              'test_batch_size': 20,
              'validation_batches': 50,
              'validation_freq': 1,
              'epochs': 100,
              'verbose': 1,
              'buffer_n_batches_train': 20,
              'buffer_n_batches_eval': 5,
              'train_file_list': train_files,
              'eval_file_list': eval_files}

# Model directory
log_dir_model = os.path.join(log_dir, model_name)

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
#hist = trainer.train(convmodel, train_files, eval_files)
#hist_file = os.path.join(log_dir_model, model_name+'_hist_dict.pkl')
#save_model_dict(hist.history, hist_file)

