import os
import glob
import pickle

# Custom imports
from werdich_cfr.tfutils.Modeltrainer import VideoTrainer

#%% Directories and data sets
tfr_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr/tfr_200202')
log_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr/log')
p_list = [1.247, 1.583, 2.075]

# Training, evaluation and test sets
train_files = glob.glob(os.path.join(tfr_dir, 'CFR_200202_view_a4c_train_*.tfrecords'))
eval_files = glob.glob(os.path.join(tfr_dir, 'CFR_200202_view_a4c_eval_*.tfrecords'))
test_files = glob.glob(os.path.join(tfr_dir, 'CFR_200202_view_a4c_test_*.tfrecords'))

# Model name
base_name = '20020a4c'

# Model parameters
model_dict = {'name': base_name,
              'im_size': (299, 299, 1),
              'n_frames': 30,
              'cfr_boundaries': p_list,
              'cl_outputs': len(p_list)+1,
              'filters': 32,
              'pool_nodes': 256,
              'fc_nodes': 256,
              'fullnet': True,
              'im_resize_crop': True}

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
              'buffer_n_batches_eval': 5}

# We can run different models by changing some parameters
im_resize_crop = False

if im_resize_crop:
    model_name = base_name+'_crop'
    log_dir_model = os.path.join(log_dir, model_name)
else:
    model_name = base_name + '_pad'
    log_dir_model = os.path.join(log_dir, model_name)

model_dict['im_resize_crop'] = im_resize_crop
model_dict['name'] = model_name

trainer = VideoTrainer(log_dir=log_dir_model,
                       model_dict=model_dict,
                       train_dict=train_dict)

convmodel = trainer.compile_convmodel()
convmodel.summary()

# Run the training
hist = trainer.train(convmodel, train_files, eval_files)

# Save fit history, model and training parameters
with open(os.path.join(log_dir, model_name+'_hist.pickle'), 'wb') as f:
    pickle.dump(hist.history, f, protocol = pickle.HIGHEST_PROTOCOL)

with open(os.path.join(log_dir_model, model_name+'_model_dict.pickle'), 'wb') as f:
    pickle.dump(model_dict, f, protocol = pickle.HIGHEST_PROTOCOL)

with open(os.path.join(log_dir_model, model_name+'_train_dict.pickle'), 'wb') as f:
    pickle.dump(train_dict, f, protocol = pickle.HIGHEST_PROTOCOL)
