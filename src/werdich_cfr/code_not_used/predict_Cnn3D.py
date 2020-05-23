import os
import glob
import pickle
import pandas as pd

from tensorflow.keras.models import load_model

# Custom imports
from werdich_cfr.tfutils.Modeltrainer import VideoTrainer

#%% Support functions

def read_model_dict(file):
    with open(file, 'rb') as f:
        model_dict = pickle.load(f)
    return model_dict

#%% Directories and data sets
cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
log_dir = os.path.join(cfr_data_root, 'log', '200208a4c')
test_data_dir = os.path.join(cfr_data_root, 'tfr_200208')
model_dict_file = glob.glob(os.path.join(log_dir, '*_model_dict.pkl'))[0]
train_dict_file = glob.glob(os.path.join(log_dir, '*_train_dict.pkl'))[0]
model_dict = read_model_dict(model_dict_file)
train_dict = read_model_dict(train_dict_file)

# GPUs
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

# Re-create the model from checkpoint
checkpoint_file_list = sorted(glob.glob(os.path.join(log_dir, model_dict['name']+'*_chkpt_*.hdf5')))
checkpoint_file_list_predict = [checkpoint_file_list[-1]]

test_tfr_files = sorted(glob.glob(os.path.join(test_data_dir, '*_test_*.tfrecords')))
test_parquet_files = [file.replace('.tfrecords', '.parquet') for file in test_tfr_files]

if len(test_parquet_files)>1:
    test_df = pd.concat([pd.read_parquet(file) for file in test_parquet_files], axis = 0)
else:
    test_df = pd.read_parquet(test_parquet_files[0])

#%% estimator initialization and weights

#Instantiate the prediction
estimator = VideoTrainer(log_dir=log_dir,
                         model_dict=model_dict,
                         train_dict=None)

#%% get the labels from the TFR data
# We do not need that, just for consistency check.

n_steps_test, test_set = estimator.build_dataset(test_tfr_files,
                                                 batch_size=64,
                                                 buffer_n_batches=None,
                                                 repeat_count=1,
                                                 shuffle=False,
                                                 drop_remainder=False)
tfr_cfr_list = []
for step, output_batch in enumerate(test_set.take(n_steps_test)):
    print('Generating labels from tfrecords: {} of {}.'.format(step+1, n_steps_test))
    tfr_cfr_list.extend(list(output_batch[1]['score_output'].numpy()))

#%% Create the model from checkpoint

model = load_model(checkpoint_file_list_predict[-1])

#%% Run forward pass
print('Generating {} predictions steps for this testset.'.format(n_steps_test))
for c, cfile in enumerate(checkpoint_file_list_predict):
    print('Running predictinos from checkpoint {} of {}: {}'.format(c+1, len(checkpoint_file_list_predict),
                                                                    os.path.basename(cfile)))
    model.load_weights(cfile)
    class_output, score_output = estimator.predict(model, test_tfr_files, steps=None)
    test_df_predict = test_df.assign(cfr_predicted=score_output,
                                     cfr_tfr=tfr_cfr_list)
    predict_file = os.path.basename(cfile).split('.')[0]+'_predicted.parquet'
    test_df_predict.to_parquet(os.path.join(log_dir, predict_file))
