import os
import numpy as np
import lz4.frame
import cv2
import pandas as pd

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

# Custom import
from werdich_cfr.tfutils.TFRprovider import Dset
from werdich_cfr.utils.processing import Videoconverter

#%% files and directories
cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
meta_date = '200425'
# Additional information for filename
meta_dir = os.path.join(cfr_data_root, 'metadata_'+meta_date)
cfr_meta_file = 'global_pet_echo_dataset_'+meta_date+'.parquet'
tfr_dir = os.path.join(cfr_data_root, 'tfr_'+meta_date, 'global')
meta_df = pd.read_parquet(os.path.join(meta_dir, cfr_meta_file))

# Variable names
var_list = ['rest_global_mbf', 'stress_global_mbf', 'global_cfr_calc']

#max_samples_per_file = 2000
n_tfr_files = 8 # We should have one TFR file per GPU

# This should give us ~70% useful files
min_rate = 21 # Minimum acceptable frame rate [fps]
min_frames = 40 # Minimum number of frames at min_rate (2 s)
min_length = min_frames/min_rate
max_frame_time = 1/min_rate*1e3 # Maximum frame time [ms]

#%% Support functions

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

#%% Select one view and process files

view = 'a4c'
tfr_info = 'nondefectp'

for mode in meta_df['mode'].unique():

    # Filter view, mode and rates. Shuffle.
    df = meta_df[(meta_df.max_view == view) & (meta_df['mode'] == mode)].sample(frac=1)
    print('View:{}, mode:{}, min_rate:{}, min_length: {}, n_videos:{}'.format(view,
                                                                              mode,
                                                                              min_rate,
                                                                              min_length,
                                                                              len(df.filename.unique())))

    file_list_complete = list(df.filename.unique())
    # Split filename_list into multiple parts
    # n_samples_per_file = max_samples_per_file
    n_samples_per_file = int(np.ceil(len(file_list_complete)/n_tfr_files))
    file_list_parts = list(chunks(file_list_complete, n_samples_per_file))
    mag = int(np.floor(np.log10(len(file_list_parts)))) + 1

    vc = Videoconverter(min_rate=min_rate, min_frames=min_frames, meta_df=meta_df)
    # Each part will have its own TFR filename
    for part, file_list in enumerate(file_list_parts):

        # TFR filename
        tfr_basename = 'cfr_'+tfr_info+'_'+view+'_'+mode+'_'+meta_date+'_'+str(part).zfill(mag)
        tfr_filename = tfr_basename+'.tfrecords'
        parquet_filename = tfr_basename+'.parquet'

        print()
        print('Processing {} part {} of {}'.format(tfr_filename, part + 1, len(file_list_parts)))

        im_array_list = [] # list of image arrays [row, col, frame]
        im_array_ser_list = [] # list of pd.Series object for the files in im_array_list
        im_failed_ser_list = [] # list of pd.Series objects for failed videos
        cfr_list = []
        rest_mbf_list = []
        stress_mbf_list = []
        record_list = []

        for f, filename in enumerate(file_list):

            if (f+1) % 200 == 0:
                print('Loading video {} of {} into memory.'.format(f+1, len(file_list)))

            ser_df = df.loc[df.filename == filename, :]
            ser = ser_df.iloc[0]

            im_array = vc.process_video(filename)

            if np.any(im_array):
                im_array_list.append(im_array)
                cfr_list.append(ser.unaffected_cfr)
                rest_mbf_list.append(ser.rest_mbf_unaff)
                stress_mbf_list.append(ser.stress_mbf_unaff)
                record_list.append(ser.name)
                ser_df2 = ser_df.assign(im_array_shape=[list(im_array.shape)])
                im_array_ser_list.append(ser_df2)
            else:
                print('Could not open this file: {}. Skipping'.format(filename))
                im_failed_ser_list.append(ser_df)

        # Write TFR file
        if len(im_array_list) > 0:
            TFR_saver = Dset(data_root=tfr_dir)

            TFR_saver.create_tfr(filename=tfr_filename,
                                 image_data=im_array_list,
                                 cfr_data=cfr_list,
                                 rest_mbf_data=rest_mbf_list,
                                 stress_mbf_data=stress_mbf_list,
                                 record_data=record_list)

            # When this is done, save the parquet file
            im_array_df = pd.concat(im_array_ser_list)
            im_array_df.to_parquet(os.path.join(tfr_dir, parquet_filename))
