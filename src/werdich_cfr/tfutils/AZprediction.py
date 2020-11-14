"""
Model predictions from checkpoint file
Compile all data as .npy array and expand into memory
"""

#%% Imports

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

#%% Some small helper functions

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

#%% Parameters and files
# Video selection criteria
max_frame_time_ms = 33.34 # Maximum frame_time acceptable in ms
min_rate = 1/max_frame_time_ms*1e3
min_frames = 40 # Minimum number of frames at min_rate (2 s)
min_length = max_frame_time_ms*min_frames*1e-3

# GPU parameters
physical_devices, device_list = use_gpu_devices(gpu_device_string='0,1')
batch_size = 8

# Directory for metadata, predictions
cfr_project_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr_AZ')
predict_dir = os.path.join(cfr_project_dir, 'predictions')

# Model names and checkpoint files
model_dir = os.path.join(cfr_project_dir, 'models')
checkpoint_list = 'cfr_correlations_bestmodels_30FPS.parquet'
checkpoint_df = pd.read_parquet(os.path.join(cfr_project_dir, 'models', checkpoint_list))
model_list = sorted(list(checkpoint_df.model_name.unique()))

# Video list
video_meta_file_name = 'bwh_metadata.parquet'
video_meta_file = os.path.join(cfr_project_dir, video_meta_file_name)
video_meta_df = pd.read_parquet(os.path.join(video_meta_file))
file_list = list(video_meta_df.filename.unique())

#meta_date = '200617'
#meta_dir = os.path.join(cfr_data_root, 'metadata_'+meta_date)

#%% Load video data into memory and start preprocessing
print(f'Loading {len(file_list)} echos into memory from file: {os.path.basename(video_meta_file_name)}.')

# Image processing class
VC = Videoconverter(max_frame_time_ms=max_frame_time_ms, min_frames=min_frames, meta_df=video_meta_df)

image_array_file_list = []
image_array_list = []
meta_disqualified_list = []
start_time = time.perf_counter()
for f, filename in enumerate(file_list):

    if (f+1) % 10 == 0:
        time_passed = (time.perf_counter()-start_time)/60
        print(f'Loading file {f+1} of {len(file_list)}: {filename}. Time: {time_passed:.2f}')

    error, im = VC.process_video(filename)

    if np.any(im):
        image_array_list.append((im, np.asarray(im.shape, np.int32)))
        image_array_file_list.append(filename)
    else:
        echo_df_fl = video_meta_df[video_meta_df.filename == filename].assign(err=[error])
        meta_disqualified_list.append(echo_df_fl)
        print('Skipping this one.')

if len(meta_disqualified_list) > 0:
    echo_df_disqualified = pd.concat(meta_disqualified_list, ignore_index=True)
    # Save disqualified metadata
    print(f'Found {echo_df_disqualified.shape[0]} of {len(file_list)} disqualified videos.')
    disqualified_filename = os.path.basename(video_meta_df).split('.')[0] + '_disqualified.parquet'
    echo_df_disqualified.to_parquet(os.path.join(predict_dir, disqualified_filename))

print(f'Loaded {len(image_array_list)} of {len(file_list)} videos into memory.')


#%% Run predictions for all models

# Loop over the models
for m, model_name in enumerate(model_list):

    print(f'Loading model {m+1}/{len(model_list)}: {model_name}')

    model_s = checkpoint_df[checkpoint_df.model_name == model_name].iloc[0]
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
