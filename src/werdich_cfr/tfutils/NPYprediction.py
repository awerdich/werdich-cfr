"""
Model predictions on .npy.lzy video files
"""

import os
import glob
import pickle
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

import tensorflow as tf
from scipy.stats import spearmanr
from tensorflow.keras.models import load_model

from werdich_cfr.tfutils.TFRprovider import DatasetProvider
from werdich_cfr.tfutils.tfutils import use_gpu_devices

#%% Select GPU device
physical_devices, device_list = use_gpu_devices(gpu_device_string='0,1,2,3')

# Directories and files
meta_date = '200304'
cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
meta_dir = os.path.join(cfr_data_root, 'metadata_'+meta_date)
log_dir = os.path.join(cfr_data_root, 'log', 'meta200304_restmbf_0311gpu2')
model_name = 'meta200304_restmbf_0311gpu2'
checkpoint_file = os.path.join(log_dir, 'meta200304_restmbf_0311gpu2_chkpt_150.h5')

# We need the model_dict for the correct image transformations
model_dict_file = os.path.join(log_dir, model_name+'_model_dict.pkl')
with open(model_dict_file, 'rb') as fl:
    model_dict = pickle.load(fl)


#%% Datasets
# Our test set
testset_predicted_file = os.path.join(log_dir, 'cfr_resized75_a4c_test_200304.parquet')
tdf_original = pd.read_parquet(testset_predicted_file)
tdf_label = tdf_original[['study', 'filename', 'dir', 'rest_mbf_unaff', 'label', 'pred']]

# Filenames for prediction
file_list = list(tdf_label.filename)
file_df = pd.DataFrame({'filename': file_list})

# Get BWH metadata
meta_filename = 'echo_BWH_meta_cfr_'+meta_date+'.parquet'
meta_file = os.path.join(meta_dir, meta_filename)
meta_df = pd.read_parquet(meta_file)
print(f'Loaded meta data with {meta_df.shape[0]} rows.')
file_df = file_df.merge(right=meta_df, how='left', on='filename')

# Preprocessing step 1: PRE-TFR



#%%



tfr_file_list = sorted(glob.glob(os.path.join(tfr_dir, 'cfr_resized75_a4c_test_200304_*.tfrecords')))
parquet_file_list = [file.replace('.tfrecords', '.parquet') for file in tfr_file_list]
dataset_basename = os.path.basename(tfr_file_list[0]).split('.')[0].rsplit('_', maxsplit=1)[0]

# Create a test data set
testset_provider = DatasetProvider(output_height=model_dict['im_size'][0],
                                   output_width=model_dict['im_size'][1],
                                   im_scale_factor=model_dict['im_scale_factor'],
                                   model_output=model_dict['model_output'])

testset = testset_provider.make_batch(tfr_file_list=tfr_file_list,
                                      batch_size=12,
                                      shuffle=False,
                                      buffer_n_steps=None,
                                      repeat_count=1,
                                      drop_remainder=False)

#%% Get true labels labels from tfr
score_list = []
print('Extracting true labels from data set: {}'.format(os.path.basename(tfr_dir)))
for n_steps, batch in enumerate(testset):
    score_list.extend(batch[1]['score_output'].numpy())
print('Samples: {}, steps: {}'.format(len(score_list), n_steps+1))

# We will add the predictions to the original df
df_list = []
print('Loading metadata.')
for f, parquet_file in enumerate(parquet_file_list):
    df_list.append(pd.read_parquet(parquet_file))
df = pd.concat(df_list).reset_index(drop=True)

#%% Predictions
print('Loading model from checkpoint {}.'.format(os.path.basename(checkpoint_file)))
model = load_model(checkpoint_file)
predictions = model.predict(testset, verbose=1, steps=n_steps+1)

#%% Add predictions and labels to df
pred_df = df.assign(label=score_list, pred=predictions)
testset_predicted_file = os.path.join(log_dir, dataset_basename+'.parquet')
pred_df.to_parquet(testset_predicted_file)

# Lets report the correlation
cor = spearmanr(pred_df.label, pred_df.pred)
print('Correlation {:.3f}'.format(cor[0]))





