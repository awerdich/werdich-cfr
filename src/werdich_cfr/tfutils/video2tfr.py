import os
import numpy as np
import pickle
import lz4.frame
import cv2
import pandas as pd

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

# Custom import
from werdich_cfr.tfutils.TFRprovider import Dset
from werdich_cfr.utils.processing import Videoconverter
from werdich_cfr.tfutils.tfutils import use_gpu_devices

#%% Select GPUs

physical_devices, device_list = use_gpu_devices(gpu_device_string='0,1,2,3')

#%% files and directories and parameters for all data sets
cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
meta_date = '200617'

# Additional information for filename
meta_dir = os.path.join(cfr_data_root, 'metadata_'+meta_date)

# This should give us ~70% useful files
max_frame_time_ms = 33.34 # Maximum frame_time acceptable in ms
min_rate = 1/max_frame_time_ms*1e3
min_frames = 40 # Minimum number of frames at min_rate (2 s)
min_length = max_frame_time_ms*min_frames*1e-3

n_tfr_files = 8 # We should have at least one TFR file per GPU

#%% Support functions

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

#%% Data set files

dset_list = ['cfr', 'mbf_ammonia', 'mbf_rubidium']
tracer_list = ['ammonia', 'rubidium']

# THIS COULD BE A LOOP
for dset in dset_list:
#dset = dset_list[1]

    cfr_meta_file = 'global_pet_echo_dataset_'+meta_date+'.parquet'
    tfr_dir = os.path.join(cfr_data_root, 'tfr_'+meta_date, dset)
    float_label_list = ['rest_global_mbf', 'stress_global_mbf', 'global_cfr_calc']
    meta_df = pd.read_parquet(os.path.join(meta_dir, cfr_meta_file))

    # Filter the data set for mbf models
    tracer=dset.split('_')[-1]
    if tracer in tracer_list:
        meta_df = meta_df[meta_df.tracer_obi==tracer]

    #%% Select one view and process files
    # We cannot insert NAs into the label lists.
    # Drop rows with NAs in the label columns
    meda_df = meta_df.dropna(subset=float_label_list, how='any', axis=0)
    print(f'Copying meta data {cfr_meta_file} into TFR format.')
    print(f'Processing data set {dset} with tracer filter {list(meta_df.tracer_obi.unique())}')
    print(f'Saving data to {tfr_dir}.')

    view = 'a4c'
    tfr_info = dset

    for mode in meta_df.dset_mode.unique():

        # Filter view, mode and rates. Shuffle.
        df = meta_df[(meta_df.max_view == view) & (meta_df.dset_mode == mode)].sample(frac=1)

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

        vc = Videoconverter(max_frame_time_ms=max_frame_time_ms, min_frames=min_frames, meta_df=meta_df)
        # Each part will have its own TFR filename
        for part, file_list in enumerate(file_list_parts):

            # TFR filename
            tfr_basename = tfr_info+'_'+view+'_'+mode+'_'+meta_date+'_'+str(part).zfill(mag)
            tfr_filename = tfr_basename+'.tfrecords'
            parquet_filename = tfr_basename+'.parquet'

            print()
            print('Processing {} part {} of {}'.format(tfr_filename, part + 1, len(file_list_parts)))

            # Data dictionaries
            array_data_dict = {'image': []}
            float_data_dict = {name: [] for name in float_label_list}
            int_data_dict = {'record': []}

            im_array_ser_list = [] # list of pd.Series object for the files in im_array_list
            im_failed_ser_list = [] # list of pd.Series objects for failed videos

            for f, filename in enumerate(file_list):

                if (f+1) % 200 == 0:
                    print('Loading video {} of {} into memory.'.format(f+1, len(file_list)))

                ser_df = df.loc[df.filename == filename, :]
                # Exclude post-2018 data if there is more than one row for this file
                if ser_df.shape[0] > 1:
                    ser_df = ser_df[ser_df['post-2018'] == 0]
                ser = ser_df.iloc[0]

                im_array = vc.process_video(filename)

                if np.any(im_array):
                    # Data dictionaries
                    array_data_dict['image'].append(im_array)
                    for label in float_label_list:
                        float_data_dict[label].append(ser[label])
                    int_data_dict['record'].append(ser.name)

                    ser_df2 = ser_df.assign(im_array_shape=[list(im_array.shape)])
                    im_array_ser_list.append(ser_df2)
                else:
                    im_failed_ser_list.append(ser_df)

            # Write TFR file
            if len(array_data_dict['image']) > 0:
                TFR_saver = Dset(data_root=tfr_dir)

                TFR_saver.create_tfr(filename=tfr_filename,
                                     array_data_dict=array_data_dict,
                                     float_data_dict=float_data_dict,
                                     int_data_dict=int_data_dict)

                # Save feature names (needed for parsing the tfrechords files)
                array_list = list(array_data_dict.keys())
                array_list.append('shape')
                feature_dict = {'array': array_list,
                                'float': list(float_data_dict.keys()),
                                'int': list(int_data_dict.keys()),
                                'features': list(TFR_saver.feature_dict.keys())}
                feature_dict_file_name = os.path.splitext(cfr_meta_file)[0]+'.pkl'
                feature_dict_file = os.path.join(tfr_dir, feature_dict_file_name)

                # Save the feature. We need them to decode the data.
                if not os.path.exists(feature_dict_file):
                    with open(feature_dict_file, 'wb') as fs:
                        pickle.dump(feature_dict, fs)

                # When this is done, save the parquet file
                im_array_df = pd.concat(im_array_ser_list)
                im_array_df.to_parquet(os.path.join(tfr_dir, parquet_filename))
