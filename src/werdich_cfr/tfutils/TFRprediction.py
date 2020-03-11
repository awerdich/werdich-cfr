"""
Model predictions from checkpoint file
"""

import os
import glob
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from werdich_cfr.tfutils.TFRprovider import DatasetProvider
from werdich_cfr.tfutils.tfutils import use_gpu_devices

#%% Select GPU device
physical_devices, device_list = use_gpu_devices(gpu_device_string='0,1')

# Directories and files
tfr_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr/tfr_200304')
log_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr/log/meta200304_cfr_0308gpu2')
model_name = os.path.basename(log_dir)
checkpoint_file = os.path.join(log_dir, 'meta200304_cfr_0308gpu2_chkpt_149.h5')

# We need the model_dict for the correct image transformations
model_dict_file = os.path.join(log_dir, model_name+'_model_dict.pkl')
with open(model_dict_file, 'rb') as fl:
    model_dict = pickle.load(fl)

tfr_file_list = sorted(glob.glob(os.path.join(tfr_dir, 'cfr_resized75_a4c_test_200304_*.tfrecords')))
parquet_file_list = [file.replace('.tfrecords', '.parquet') for file in tfr_file_list]
dataset_basename = os.path.basename(tfr_file_list[0]).split('.')[0].rsplit('_', maxsplit=1)[0]

# Create a test data set
testset_provider = DatasetProvider(output_height=model_dict['im_size'][0],
                                   output_width=model_dict['im_size'][1],
                                   im_scale_factor=model_dict['im_scale_factor'],
                                   model_output='cfr')

testset = testset_provider.make_batch(tfr_file_list=tfr_file_list,
                                      batch_size=16,
                                      shuffle=False,
                                      buffer_n_batches=None,
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




