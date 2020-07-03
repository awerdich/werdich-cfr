"""
Model predictions from checkpoint file
Compile all data as .npy array and expand into memory
"""

import os
import numpy as np
import pickle
import glob
import pandas as pd
import time

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

import tensorflow as tf
from tensorflow.keras.models import load_model

from werdich_cfr.tfutils.TFRprovider import DatasetProvider
from werdich_cfr.utils.processing import Videoconverter
from werdich_cfr.tfutils.tfutils import use_gpu_devices

#%% Directories and parameters

physical_devices, device_list = use_gpu_devices(gpu_device_string='0,1')

cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
predict_dir = os.path.join(cfr_data_root, 'predictions_echodata','FirstEchoEvents2')

# This should give us ~70% useful files
max_frame_time_ms = 33.34 # Maximum frame_time acceptable in ms
min_rate = 1/max_frame_time_ms*1e3
min_frames = 40 # Minimum number of frames at min_rate (2 s)
min_length = max_frame_time_ms*min_frames*1e-3
batch_size = 14

# Model info
# This meta_date should correspond to the meta data used for trainin (dictionaries)
meta_date = '200617'
best_models = pd.read_parquet(os.path.join(predict_dir, 'cfr_correlations_bestmodels_30FPS.parquet')).reset_index(drop=True)

model_list = list(best_models.model_name.unique())
meta_dir = os.path.join(cfr_data_root, 'metadata_'+meta_date)

#%% Some helper functions

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_im_generator(im_array_list):
    """ Yield successive images from list """
    def im_generator():
        for element in im_array_list:
            yield (element[0], element[1])
    return im_generator

def predict_from_array_list(model, array_list, batch_size):
    im_generator = get_im_generator(array_list)

    dset = tf.data.Dataset.from_generator(generator=im_generator,
                                          output_types=(tf.int32, tf.int32),
                                          output_shapes=(tf.TensorShape([None, None, model_dict['n_frames']]),
                                                         tf.TensorShape([3])))
    dset = dset.map(dsp._process_image)
    dset = dset.map(lambda x: ({'video': x}, {'score_output': 0}))
    dset = dset.batch(batch_size=batch_size, drop_remainder=False).repeat(count=1)

    predict_list = list(np.ndarray.flatten(model.predict(dset, verbose=1, steps=n_steps)))

    return predict_list

#%% Copy image data into memory

# File list for test data depends on the model
#dset = 'nondefect'
#model_list = [model for model in model_list if model.split('_')[0]==dset]
#echo_dir = os.path.join(cfr_data_root, 'tfr_'+meta_date, dset)
#test_parquet_list = sorted(glob.glob(os.path.join(echo_dir, '*_test_'+meta_date+'_?.parquet')))
#echo_df_list = [pd.read_parquet(parquet_file) for parquet_file in test_parquet_list]
#echo_df_file = test_parquet_list[0].replace('_0', '')
#echo_df = pd.concat(echo_df_list)

# File list with .npy.lz4 files
# NPY file list
echo_df_file = os.path.join(predict_dir, 'BWH_2015-06-01_2015-11-30_FirstEcho_a4c.parquet')
echo_df = pd.read_parquet(echo_df_file)
file_list = list(echo_df.filename.unique())

print(f'Running inference on: {os.path.basename(echo_df_file)}.')

# Image processing class
#max_frame_time_ms, min_frames, meta_df)
vc = Videoconverter(max_frame_time_ms=max_frame_time_ms, min_frames=min_frames, meta_df=echo_df)

image_array_file_list = []
image_array_list = []
meta_disqualified_list = []
start_time = time.perf_counter()
for f, filename in enumerate(file_list):

    if (f+1) % 10 == 0:
        time_passed = (time.perf_counter()-start_time)/60
        print(f'Loading file {f+1} of {len(file_list)}: {filename}. Time: {time_passed:.2f}')

    error, im = vc.process_video(filename)

    if np.any(im):
        image_array_list.append((im, np.asarray(im.shape, np.int32)))
        image_array_file_list.append(filename)
    else:
        echo_df_fl = echo_df[echo_df.filename==filename].assign(err=[error])
        meta_disqualified_list.append(echo_df_fl)
        print('Skipping this one.')

if len(meta_disqualified_list)>0:
    echo_df_disqualified = pd.concat(meta_disqualified_list, ignore_index=True).reset_index(drop=True)
    # Save disqualified metadata
    print(f'Found {echo_df_disqualified.shape[0]} of {len(file_list)} disqualified videos.')
    disqualified_filename = os.path.basename(echo_df_file).split('.')[0] + '_disqualified.parquet'
    echo_df_disqualified.to_parquet(os.path.join(predict_dir, disqualified_filename))

print(f'Loaded {len(image_array_list)} of {len(file_list)} videos into memory.')


#%% Run predictions for all models

# Loop over the models
for m, model_name in enumerate(model_list):

    print(f'Loading model {m+1}/{len(model_list)}: {model_name}')

    model_s = best_models[best_models.model_name==model_name].iloc[0]
    dset = model_s.dset
    model_name = model_s.model_name
    tfr_dir = os.path.join(cfr_data_root, 'tfr_'+meta_date, dset)
    log_dir = os.path.join(cfr_data_root, 'log', model_name)
    checkpoint_file = model_s.checkpoint_file

    print('Loading model from checkpoint {}.'.format(os.path.basename(checkpoint_file)))
    model = load_model(checkpoint_file)

    # We need the model_dict for the correct image transformations
    model_dict_file = os.path.join(log_dir, model_name+'_model_dict.pkl')
    with open(model_dict_file, 'rb') as fl:
        model_dict = pickle.load(fl)
    model_output = model_dict['model_output']

    # feature_dict
    feature_dict_file = glob.glob(os.path.join(tfr_dir, '*.pkl'))[0]
    with open(feature_dict_file, 'rb') as fl:
        feature_dict = pickle.load(fl)

    dsp = DatasetProvider(augment=False,
                          im_scale_factor=model_dict['im_scale_factor'],
                          feature_dict=feature_dict)

    # Predict from image_array_list

    n_steps = int(np.ceil(len(image_array_list) / batch_size))
    predictions = predict_from_array_list(model=model,
                                          array_list=image_array_list,
                                          batch_size=batch_size)

    pred_df_c = pd.DataFrame({'filename': image_array_file_list,
                              'model_output': model_output,
                              'dataset': [dset]*len(predictions),
                              'predictions': predictions,
                              'model_name': [model_name]*len(predictions),
                              'checkpoint': [os.path.basename(checkpoint_file)]*len(predictions)})

    # Merge output predictions and save data
    pred_df = pred_df_c.merge(right=echo_df, on='filename', how='left').reset_index(drop=True)

    video_list_file_name = model_name+'.parquet'
    video_list_file = os.path.join(predict_dir, 'cfr_models_30fps', video_list_file_name)

    pred_df.to_parquet(video_list_file)
