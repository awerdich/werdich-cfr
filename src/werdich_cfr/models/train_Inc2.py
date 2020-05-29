import os
import glob
import pickle
import socket
import pandas as pd
from scipy.stats import spearmanr

import tensorflow as tf
from tensorflow.keras.models import load_model
print(f'Tensorflow version: {tf.__version__}')

# Custom imports
from werdich_cfr.models.Modeltrainer_Inc2 import VideoTrainer
from werdich_cfr.tfutils.tfutils import use_gpu_devices

#%% GPU CONFIGURATION

physical_devices, device_list = use_gpu_devices(gpu_device_string='0,1')

#%% Some helper functions

def write_model_dict(model_dict, file):
    with open(file, 'wb') as f:
        pickle.dump(model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved {}.'.format(file))

def get_file_list(tfr_data_dir, meta_date, dset, view, mode):
    file_pattern = os.path.join(tfr_data_dir, 'cfr_'+dset+'_'+view+'_'+mode+'_'+meta_date+'_*.tfrecords')
    file_list = sorted(glob.glob(file_pattern))
    print(mode)
    print(*file_list, sep='\n')
    return file_list

#%% Directories and parameters
cfr_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr')
hostname = socket.gethostname()
meta_date = '200519'
view = 'a4c'

# MODEL DICTIONARIES
run_model_dict_list = []
# Dataset: global
dset = 'global'
tfr_data_dir = os.path.join(cfr_dir, 'tfr_' + meta_date, dset)
response_variables_list = ['rest_global_mbf', 'stress_global_mbf', 'global_cfr_calc']
features_dict_file = os.path.join(tfr_data_dir, dset+'_pet_echo_dataset_'+meta_date+'.pkl')
model_name = dset+'_fc128_'+view+'_'+hostname.strip('obi-')
run_model_dict_1 = {'model_name': model_name,
                    'dset': dset,
                    'response_variables_list': response_variables_list,
                    'tfr_data_dir': tfr_data_dir,
                    'features_dict_file': features_dict_file}
run_model_dict_list.append(run_model_dict_1)

# Dataset: nondefect
dset = 'nondefect'
tfr_data_dir = os.path.join(cfr_dir, 'tfr_' + meta_date, dset)
response_variables_list = ['rest_mbf_unaff', 'stress_mbf_unaff', 'unaffected_cfr']
features_dict_file = os.path.join(tfr_data_dir, dset+'_pet_echo_dataset_'+meta_date+'.pkl')
model_name = dset+'_fc128_'+view+'_'+hostname.strip('obi-')
run_model_dict_2 = {'model_name': model_name,
                    'dset': dset,
                    'response_variables_list': response_variables_list,
                    'tfr_data_dir': tfr_data_dir,
                    'features_dict_file': features_dict_file}
run_model_dict_list.append(run_model_dict_2)

#%% Model training
# We could loop over the run_model dicts
# for run_model_dict in run_model_dict_list
run_model_dict = run_model_dict_list[0]

# feature_dict
with open(run_model_dict['features_dict_file'], 'rb') as fl:
    feature_dict = pickle.load(fl)

# TFR files
tfr_data_dir = run_model_dict['tfr_data_dir']
train_file_list = get_file_list(tfr_data_dir, meta_date=meta_date, dset=run_model_dict['dset'], view=view, mode='train')
eval_file_list = get_file_list(tfr_data_dir, meta_date=meta_date, dset=run_model_dict['dset'], view=view, mode='eval')
test_file_list = get_file_list(tfr_data_dir, meta_date=meta_date, dset=run_model_dict['dset'], view=view, mode='test')

# TESTING
#train_file_list = [eval_file_list[0]]
#eval_file_list = [eval_file_list[1]]
#print('WARNING: *** TESTING: USING WRONG TFR FILE LISTS ***')

#%% Training the models defined by the response_variables_list

response_variables_list = run_model_dict['response_variables_list']
for m, model_output in enumerate(response_variables_list):

    model_name = run_model_dict['model_name']+'_'+model_output
    log_dir = os.path.join(cfr_dir, 'log', model_name)

    # Model parameters
    model_dict = {'name': model_name,
                  'im_size': (299, 299, 1),
                  'im_scale_factor': 1.177,
                  'min_rate': 21,
                  'n_frames': 40,
                  'filters': 64,
                  'fc_nodes': 128,
                  'model_output': model_output,
                  'kernel_init': tf.keras.initializers.GlorotNormal(),
                  'bias_init': tf.keras.initializers.Zeros()}

    # Training parameters
    train_dict = {'train_device_list': device_list,
                  'learning_rate': 0.0001,
                  'augment': False,
                  'train_batch_size': 18,
                  'eval_batch_size': 18,
                  'validation_batches': None,
                  'validation_freq': 1,
                  'n_epochs': 150,
                  'verbose': 1,
                  'train_file_list': train_file_list,
                  'eval_file_list': eval_file_list}

    # Save model dictionaries before starting to train
    # Create the log dir, if it does not exist already
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    model_dict_file = os.path.join(log_dir, model_name+'_model_dict.pkl')
    train_dict_file = os.path.join(log_dir, model_name+'_train_dict.pkl')
    write_model_dict(model_dict, model_dict_file)
    write_model_dict(train_dict, train_dict_file)

    # Compile the model
    VT = VideoTrainer(log_dir=log_dir, model_dict=model_dict, train_dict=train_dict, feature_dict=feature_dict)
    model = VT.compile_inc2model()
    model.summary()

    # Run the training and save the history data
    print(f'Training model {m+1}/{len(response_variables_list)}: {model_name}')
    hist = VT.train(model)
    hist_file = os.path.join(log_dir, model_name+'_hist_dict.pkl')
    write_model_dict(hist.history, hist_file)

    # Forward pass on test set
    chkpt_list = [50, 100, 150]
    chkpt_name_list = [model_dict['name'] + '_chkpt_' + str(chkid).zfill(3) + '.h5' for chkid in chkpt_list]
    chkpt_file_list = [os.path.join(log_dir, chkpoint_name) for chkpoint_name in chkpt_name_list]

    # Get the predictions from the checkpoint files
    pred_df_list = []
    for c, checkpoint_file in enumerate(chkpt_file_list):
        print(f'Checkpoint {c + 1}/{len(chkpt_file_list)}: {checkpoint_file}')
        pred = VT.predict_on_test(test_file_list, checkpoint_file)
        pred_df_list.append(pred)

    # Concatenate all predictions
    pred_df = pd.concat(pred_df_list, axis=1)
    pred_df = pred_df.loc[:,~pred_df.columns.duplicated()]

    # Load the meta data from parquet files and add predictions
    parquet_file_list = [file.replace('.tfrecords', '.parquet') for file in test_file_list]
    df_test = pd.concat([pd.read_parquet(file) for file in parquet_file_list]).reset_index(drop=True)
    df_test = pd.concat([df_test, pred_df], axis=1).reset_index(drop=True)

    # Save predictions
    testset_predicted_file = os.path.join(log_dir, model_dict['name']+'_pred.parquet')
    pred_df.to_parquet(testset_predicted_file)
