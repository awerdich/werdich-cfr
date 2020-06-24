import os
import glob
import numpy as np
import pandas as pd
import lz4.frame

from werdich_cfr.utils.processing import Videoconverter


#%% files and directories and parameters for all data sets
cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
meta_date = '200617'
dset='cfr'

# Additional information for filename
meta_dir = os.path.join(cfr_data_root, 'metadata_'+meta_date)
tfr_dir = os.path.join(cfr_data_root, 'tfr_'+meta_date, dset)

# This should give us ~70% useful files
max_frame_time_ms = 33.34 # Maximum frame_time acceptable in ms
min_rate = 1/max_frame_time_ms*1e3
min_frames = 40 # Minimum number of frames at min_rate (2 s)
min_length = max_frame_time_ms*min_frames*1e-3

n_tfr_files = 8 # We should have at least one TFR file per GPU

# TFR .parquet data files
train_files = sorted(glob.glob(os.path.join(tfr_dir, dset+'_a4c_train_'+meta_date+'_*.parquet')))
eval_files = sorted(glob.glob(os.path.join(tfr_dir, dset+'_a4c_eval_'+meta_date+'_*.parquet')))
test_files = sorted(glob.glob(os.path.join(tfr_dir, dset+'_a4c_test_'+meta_date+'_*.parquet')))

# List of files that failed TFR conversion
train_failed_files = [file.replace('.parquet', '.failed') for file in train_files]
eval_failed_files = [file.replace('.parquet', '.failed') for file in eval_files]
test_failed_files = [file.replace('.parquet', '.failed') for file in test_files]

print(train_failed_files)

#%% Data set files
train_df = pd.concat([pd.read_parquet(file) for file in train_files])
eval_df = pd.concat([pd.read_parquet(file) for file in eval_files])
test_df = pd.concat([pd.read_parquet(file) for file in test_files])
df = pd.concat([train_df, eval_df, test_df], ignore_index=True).reset_index(drop=True)

train_failed_df = pd.concat([pd.read_parquet(file) for file in train_failed_files])
eval_failed_df = pd.concat([pd.read_parquet(file) for file in eval_failed_files])
test_failed_df = pd.concat([pd.read_parquet(file) for file in test_failed_files])

tf_data = pd.concat([train_df, eval_df, test_df], ignore_index=True).reset_index(drop=True)
tf_failed_data = pd.concat([train_failed_df, eval_failed_df, test_failed_df], ignore_index=True).reset_index(drop=True)

tf_failed_data = tf_failed_data.assign(dur = tf_failed_data.frame_time*1e-3*tf_failed_data.number_of_frames)
tf_data = tf_data.assign(dur = tf_data.frame_time*1e-3*tf_data.number_of_frames)

n_videos_success = len(tf_data.filename.unique())
n_videos_failed = len(tf_failed_data.filename.unique())
n_videos = n_videos_success + n_videos_failed
n_videos_success_frac = np.around(n_videos_success/n_videos, decimals=2)
n_videos_failed_frac = np.around(n_videos_failed/n_videos, decimals=2)
print(f'Successful conversions: {n_videos_success} of {n_videos}, {n_videos_success_frac}')
print(f'Failed conversions:     {n_videos_failed}  of {n_videos}, {n_videos_failed_frac}')

#%% Lets get the true size of the arrays

vc = Videoconverter(max_frame_time_ms=max_frame_time_ms, min_frames=min_frames, meta_df=tf_failed_data)
file_list = list(tf_failed_data.filename.unique())

im_array_ser_list = []  # list of pd.Series object for the files in im_array_list

for f, filename in enumerate(file_list):

    if (f + 1) % 50 == 0:
        print('Loaded video {} of {} into memory.'.format(f + 1, len(file_list)))

    ser_df = tf_failed_data.loc[tf_failed_data.filename == filename, :]
    file = os.path.join(ser_df.dir.values[0], filename)
    frame_time = ser_df.frame_time.values[0] * 1e-3
    rate = 1 / frame_time
    ser = ser_df.iloc[0]

    try:
        with lz4.frame.open(file, 'rb') as fp:
            data = np.load(fp)
    except IOError as err:
        print('Cannot open npy file.')
        print(err)
        error = 'load'
    else:
        video_len = data.shape[0] / rate
        ser_df2 = ser_df.assign(data_n_frames=data.shape[0],
                                data_rows=data.shape[1],
                                data_cols=data.shape[2],
                                data_video_len=video_len)
        im_array_ser_list.append(ser_df2)

# When this is done, save the parquet file
im_array_df = pd.concat(im_array_ser_list)
failed_file_name = 'global_pet_echo_dataset_200617_shape.failed'
im_array_df.to_parquet(os.path.join(meta_dir, failed_file_name))
