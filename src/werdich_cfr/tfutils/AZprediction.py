"""
Model predictions from checkpoint file
Compile all data as .npy array and expand into memory
"""

#%% Imports

import os
import numpy as np
import pickle
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

def predict_from_array_list(model, model_dict, feature_dict, array_list, batch_size):
    '''
    Predict from list of echo videos
    model: compiled (or loaded from checkpoint) keras model
    model_dict: dict with model hyperparameters
    feature_dict: names of input and output tensors
    array_list: list of tuples with video and shape (array, array.shape)
    '''

    im_generator = get_im_generator(array_list)

    dsp = DatasetProvider(augment=False,
                          im_scale_factor=model_dict['im_scale_factor'],
                          feature_dict=feature_dict)

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
physical_devices, device_list = use_gpu_devices(gpu_device_string='0,1,2,3,4,5,6,7')
batch_size = 32

# Directory for metadata, predictions
cfr_project_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr_AZ')
predict_dir = os.path.join(cfr_project_dir, 'predictions')
meta_date = '200617'

# Video list
az_dir = os.path.normpath('/mnt/obi0/sgoto/AZ_Project')
az_echo_dir = os.path.join(az_dir, 'npyFiles')
video_meta_file_name = 'metadata.tsv'
video_meta_file = os.path.join(az_dir, video_meta_file_name)
video_meta_df = pd.read_csv(video_meta_file, sep='\t')
# Some small adjustments to this data frame
video_meta_df = video_meta_df.dropna(subset=['frametime']).\
    rename(columns={'frametime': 'frame_time'}).\
    reset_index(drop=True)
video_meta_df['filename'] = video_meta_df['filename']+'.npy.lz4'
video_meta_df['dir'] = az_echo_dir
file_list = list(video_meta_df.filename.unique())

#video_meta_file_name = 'bwh_metadata.parquet'
#video_meta_file = os.path.join(cfr_project_dir, video_meta_file_name)
#video_meta_df = pd.read_parquet(os.path.join(video_meta_file))
#file_list = list(video_meta_df.filename.unique())

# Model names and checkpoint files
model_dir = os.path.join(cfr_project_dir, 'models')
checkpoint_list = 'cfr_correlations_bestmodels_30FPS.parquet'
checkpoint_df = pd.read_parquet(os.path.join(cfr_project_dir, 'models', checkpoint_list))
model_list = sorted(list(checkpoint_df.model_name.unique()))

# Features for the model
feature_dict_name = 'feature_dict_' + 'tfr_' + meta_date+'.pkl'
feature_dict_file = os.path.join(model_dir, feature_dict_name)
with open(feature_dict_file, 'rb') as fl:
    feature_dict = pickle.load(fl)

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
        meta_disqualified_list.append(video_meta_df[video_meta_df.filename == filename].\
                                      assign(err=[error]))
        print('Skipping this one.')

if len(meta_disqualified_list) > 0:
    echo_df_disqualified = pd.concat(meta_disqualified_list, ignore_index=True)
    # Save disqualified metadata
    print(f'Found {echo_df_disqualified.shape[0]} of {len(file_list)} disqualified videos.')
    disqualified_filename = os.path.basename(video_meta_file_name).split('.')[0] + '_disqualified.parquet'
    echo_df_disqualified.to_parquet(os.path.join(cfr_project_dir, disqualified_filename))

print(f'Loaded {len(image_array_list)} of {len(file_list)} videos into memory.')

#%% Run predictions for all models

# Loop over the models
for m, model_name in enumerate(model_list):

    # Model information from this series
    model_s = checkpoint_df[checkpoint_df.model_name == model_name].iloc[0]
    dset = model_s.dset

    # Load model from checkpoint
    checkpoint_file_name = os.path.basename(model_s.checkpoint_file)
    checkpoint_file = os.path.join(model_dir, checkpoint_file_name)
    print('Loading model from checkpoint: {}.'.format(os.path.basename(checkpoint_file)))
    model = load_model(checkpoint_file)
    model.summary()

    # Model hyperparameters and output name
    model_dict_file_name = model_name + '_model_dict.pkl'
    model_dict_file = os.path.join(model_dir, model_dict_file_name)
    with open(model_dict_file, 'rb') as fl:
        model_dict = pickle.load(fl)
    model_output = model_dict['model_output']

    # Predict from image_array_list
    n_steps = int(np.ceil(len(image_array_list) / batch_size))
    predictions = predict_from_array_list(model=model,
                                          model_dict=model_dict,
                                          feature_dict=feature_dict,
                                          array_list=image_array_list,
                                          batch_size=batch_size)

    pred_df_c = pd.DataFrame({'filename': image_array_file_list,
                              'model_output': model_output,
                              'dataset': [dset]*len(predictions),
                              'predictions': predictions,
                              'model_name': [model_name]*len(predictions),
                              'checkpoint': [os.path.basename(checkpoint_file)]*len(predictions)})

    # Merge output predictions and save data
    pred_df = pred_df_c.merge(right=video_meta_df, on='filename', how='left').reset_index(drop=True)

    video_list_file_name = video_meta_file_name.split('.')[0]+'_'+dset+'_'+model_output+'.parquet'
    video_list_file = os.path.join(predict_dir, video_list_file_name)

    pred_df.to_parquet(video_list_file)
