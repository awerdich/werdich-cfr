"""
Model predictions from checkpoint file
"""

import os
import numpy as np
import pickle
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

def predict_from_array_list(array_list):
    im_generator = get_im_generator(array_list)

    dset = tf.data.Dataset.from_generator(generator=im_generator,
                                          output_types=(tf.int32, tf.int32),
                                          output_shapes=(tf.TensorShape([None, None, model_dict['n_frames']]),
                                                         tf.TensorShape([3])))
    dset = dset.map(dsp._process_image)
    dset = dset.map(lambda x: ({'video': x}, {'score_output': 0}))
    dset = dset.batch(batch_size, drop_remainder=False).repeat(count=1)

    predict_list = list(np.ndarray.flatten(model.predict(dset, verbose=1, steps=n_steps)))

    return predict_list

#%% Select GPU device
physical_devices, device_list = use_gpu_devices(gpu_device_string='0,1')
batch_size = 16
# Maximum number of files to load into memory at a time
n_files_in_mem = 1000

# Model
cfr_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr')
log_dir = os.path.join(cfr_dir, 'log', 'meta200304_restmbf_0311gpu2')
model_name = 'meta200304_restmbf_0311gpu2'
checkpoint_file = os.path.join(log_dir, model_name+'_chkpt_150.h5')
print('Loading model from checkpoint {}.'.format(os.path.basename(checkpoint_file)))
model = load_model(checkpoint_file)

# Metadata
meta_date = '200320'
meta_dir = os.path.join(cfr_dir, 'metadata_'+meta_date)
meta_df = pd.read_parquet(os.path.join(meta_dir, 'echo_BWH_meta_200320.parquet'))

# We need the model_dict for the correct image transformations
model_dict_file = os.path.join(log_dir, model_name+'_model_dict.pkl')
with open(model_dict_file, 'rb') as fl:
    model_dict = pickle.load(fl)
model_dict['min_rate'] = 21
model_output = model_dict['model_output']

#%% File list with .npy.lz4 files

# NPY file list
video_list_filename = 'a4cname.txt'
video_list_file = os.path.join(cfr_dir, video_list_filename)
df = pd.read_csv(video_list_file)
df = df.rename(columns={'names': 'filename'})
#df = df.sample(n=50)
file_list = list(df.filename.unique())

#df = pd.read_parquet(os.path.join(log_dir, 'cfr_resized75_a4c_test_200304.parquet'))
#df_cols = ['filename', 'rest_mbf_unaff', 'label', 'pred']
#df = df.sample(frac=1)[df_cols]
#file_list = list(df.filename.unique()[:50])
#pred_list = [df[df.filename == fn].pred.values[0] for fn in file_list]

#%% Split files into chunks that fit into memory
pred_df_list = []
file_chunks = list(chunks(file_list, n_files_in_mem))
for c, file_list_c in enumerate(file_chunks):
    print(f'File list {c+1} of {len(file_chunks)}.')
    image_array_file_list = []
    image_array_list = []
    # Image processing and dataset
    vc = Videoconverter(min_rate=model_dict['min_rate'], min_frames=model_dict['n_frames'], meta_df=meta_df)
    dsp = DatasetProvider(augment=False, im_scale_factor=model_dict['im_scale_factor'])
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
        predictions = predict_from_array_list(image_array_list)
        pred_df_c = pd.DataFrame({'filename': image_array_file_list,
                                  model_output+'_pred': predictions,
                                  'model_name': [model_name]*len(predictions),
                                  'checkpoint': [os.path.basename(checkpoint_file)]*len(predictions)})
        #true_list = [df[df.filename == file].pred.values[0] for file in file_list_c]
        #pred_df_c = pred_df_c.assign(label=true_list)
        pred_df_list.append(pred_df_c)

#%% Concat output predictions and save data
save_name = os.path.basename(video_list_file).strip('.')+'_pred_'+model_output+'.parquet'
save_dir = os.path.dirname(video_list_file)
pred_df = pd.concat(pred_df_list, axis=0, ignore_index=True).reset_index(drop=True)
pred_df.to_parquet(os.path.join(save_dir, save_name))
# Load result
#save_name = 'test.txt_pred_rest_mbf.parquet'
#save_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr')
#pred_df = pd.read_parquet(os.path.join(save_dir, save_name))
