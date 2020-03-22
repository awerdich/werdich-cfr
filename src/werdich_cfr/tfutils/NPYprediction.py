"""
Model predictions from checkpoint file
"""

import os
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr
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
physical_devices, device_list = use_gpu_devices(gpu_device_string='0,1,2,3')
batch_size = 8
# Maximum number of files to load into memory at a time
n_files_in_mem = 49

# Directories and files: model and metadata
meta_date='200320'
cfr_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr')
log_dir = os.path.join(cfr_dir, 'log', 'meta200304_restmbf_0311gpu2')
meta_dir = os.path.join(cfr_dir, 'metadata_'+meta_date)
model_name = 'meta200304_restmbf_0311gpu2'
checkpoint_file = os.path.join(log_dir, 'meta200304_restmbf_0311gpu2_chkpt_150.h5')
print('Loading model from checkpoint {}.'.format(os.path.basename(checkpoint_file)))
model = load_model(checkpoint_file)
meta_df = pd.read_parquet(os.path.join(meta_dir, 'echo_BWH_meta_200320.parquet'))

# We need the model_dict for the correct image transformations
model_dict_file = os.path.join(log_dir, model_name+'_model_dict.pkl')
with open(model_dict_file, 'rb') as fl:
    model_dict = pickle.load(fl)

# The rate should be in model_dict:
model_dict['min_rate'] = 21

#%% NPY file list: testset for development

# Take 100 random file from test set
df = pd.read_parquet(os.path.join(log_dir, 'cfr_resized75_a4c_test_200304.parquet'))
df_cols = ['filename', 'rest_mbf_unaff', 'label', 'pred']
df = df.sample(frac=1)[df_cols]
file_list = list(df.filename.unique()[:50])
pred_list = [df[df.filename == fn].pred.values[0] for fn in file_list]

#%% Split files into chunks that fit into memory
npy_pred_list = []
file_chunks = list(chunks(file_list, n_files_in_mem))
for c, file_list_c in enumerate(file_chunks):
    print(f'File list {c+1} of {len(file_chunks)}.')
    image_array_list = []
    # Image processing and dataset
    vc = Videoconverter(min_rate=model_dict['min_rate'], min_frames=model_dict['n_frames'], meta_df=meta_df)
    dsp = DatasetProvider(augment=False, im_scale_factor=model_dict['im_scale_factor'])
    for f, filename in enumerate(file_list_c):
        print(f'Loading file {f+1} of {len(file_list_c)}: {filename}.')
        im = vc.process_video(filename)
        if np.any(im):
            image_array_list.append((im, np.asarray(im.shape, np.int32)))
        else:
            print('Skipping this one.')
    # Predict from image_array_list
    n_steps = int(np.ceil(len(image_array_list)/batch_size))
    predictions = predict_from_array_list(image_array_list)
    npy_pred_list.extend(predictions)

#%% Compare NPY predictions with TFR predictions
df_pred = pd.DataFrame({'pred': pred_list,
                        'np_pred': npy_pred_list})

d = [idx for idx in range(len(pred_list)) if pred_list[idx]!=npy_pred_list[idx]]
