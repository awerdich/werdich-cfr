"""
Model predictions from checkpoint file
"""

import os
import glob
import pickle
import pandas as pd
import tensorflow as tf

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

# Create a test data set
testset_provider = DatasetProvider(output_height=model_dict['im_size'][0],
                                   output_width=model_dict['im_size'][1],
                                   im_scale_factor=model_dict['im_scale_factor'],
                                   model_output='cfr')

testset = testset_provider.make_batch()


dset = dataset_provider.make_batch(tfr_file_list=tfr_file_list,
                                   batch_size=16,
                                   shuffle=False,
                                   buffer_n_batches=None,
                                   repeat_count=1,
                                   drop_remainder=False)

steps_per_epoch = predictor.count_steps_per_epoch(tfr_file_list=tfr_file_list,
                                                  batch_size=16)

#%% Get true labels labels from tfr
score_list = []
for i, batch in enumerate(dset):
    print('Batch {}'.format(i))
    score_list.extend(batch[1]['score_output'].numpy())

#%% Predictions
from tensorflow.keras.models import load_model
model = load_model(checkpoint_file)
predictions = model.predict(dset, verbose=1)

# We will add the predictions to the original df
df_list = []
for f, parquet_file in enumerate(parquet_file_list):
    df_list.append(pd.read_parquet(parquet_file))
df = pd.concat(df_list).reset_index(drop=True)
