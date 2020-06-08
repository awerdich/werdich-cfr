"""
Model predictions from checkpoint file
"""

import os
import numpy as np
import pickle
import glob
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

import tensorflow as tf
from tensorflow.keras.models import load_model

from werdich_cfr.tfutils.TFRprovider import DatasetProvider
from werdich_cfr.utils.processing import Videoconverter
from werdich_cfr.tfutils.tfutils import use_gpu_devices

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

#%% Select GPU device
physical_devices, device_list = use_gpu_devices(gpu_device_string='0,1,2,3')
batch_size = 32
# Maximum number of files to load into memory at a time
n_files_in_mem = 1000

# Model info
cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
meta_date = '200519'
best_models = pd.read_parquet(os.path.join(cfr_data_root, 'best_models_200607.parquet')).reset_index(drop=True)
model_list = list(best_models.model_name.unique())
meta_dir = os.path.join(cfr_data_root, 'metadata_'+meta_date)

# LOOP OVER THE MODELS
for m, model_name in enumerate(model_list):

    print(f'Loading model {m+1}/{len(model_list)}: {model_name}')

    model_s = best_models[best_models.model_name==model_name].iloc[0]
    dset = model_s.dset
    model_name = model_s.model_name
    tfr_dir = os.path.join(cfr_data_root, 'tfr_'+meta_date, dset)
    log_dir = os.path.join(cfr_data_root, 'log', model_name)
    checkpoint_file = model_s.chechkpoint_file

    print('Loading model from checkpoint {}.'.format(os.path.basename(checkpoint_file)))
    model = load_model(checkpoint_file)

    # We need the model_dict for the correct image transformations
    model_dict_file = os.path.join(log_dir, model_name+'_model_dict.pkl')
    with open(model_dict_file, 'rb') as fl:
        model_dict = pickle.load(fl)
    model_dict['min_rate'] = 21
    model_output = model_dict['model_output']

    # feature_dict
    feature_dict_file = glob.glob(os.path.join(tfr_dir, '*.pkl'))[0]
    with open(feature_dict_file, 'rb') as fl:
        feature_dict = pickle.load(fl)

    # File list with .npy.lz4 files
    # NPY file list
    echo_df_file = os.path.join(cfr_data_root, 'metadata_200606', 'echo_BWH_meta_pred_200606_a4c.parquet')
    echo_df = pd.read_parquet(echo_df_file)
    file_list = list(echo_df.filename.unique())

    # Split files into chunks that fit into memory
    pred_df_list = []
    file_chunks = list(chunks(file_list, n_files_in_mem))
    for c, file_list_c in enumerate(file_chunks):
        print(f'File list {c+1} of {len(file_chunks)}.')
        image_array_file_list = []
        image_array_list = []
        # Image processing and dataset
        vc = Videoconverter(min_rate=model_dict['min_rate'], min_frames=model_dict['n_frames'], meta_df=echo_df)
        dsp = DatasetProvider(augment=False,
                              im_scale_factor=model_dict['im_scale_factor'],
                              feature_dict=feature_dict)

        for f, filename in enumerate(file_list_c):
            print(f'Loading file {f+1} of {len(file_list_c)}: {filename}.')
            im = vc.process_video(filename)
            if np.any(im):
                image_array_list.append((im, np.asarray(im.shape, np.int32)))
                image_array_file_list.append(filename)
            else:
                print('Skipping this one.')
        # Predict from image_array_list
        if len(image_array_file_list)>0:
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
            #true_list = [df[df.filename == file].pred.values[0] for file in file_list_c]
            #pred_df_c = pred_df_c.assign(label=true_list)
            pred_df_list.append(pred_df_c)

    # Concat output predictions and save data
    pred_df = pd.concat(pred_df_list, axis=0, ignore_index=True)
    pred_df = pred_df.merge(right=echo_df, on='filename', how='left').reset_index(drop=True)

    video_list_file_name = 'BWH_2015-05-01_2015-10-31_FirstEcho_'+model_output+'.parquet'
    video_list_file = os.path.join(cfr_data_root, 'predictions_echodata', video_list_file_name)

    pred_df.to_parquet(video_list_file)
