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

#%% files and directories
cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
meta_date = '200202'
meta_dir = os.path.join(cfr_data_root, 'metadata_'+meta_date)
cfr_meta_file = 'tfr_files_dset_BWH_'+meta_date+'.parquet'
meta_df = pd.read_parquet(os.path.join(meta_dir, cfr_meta_file))
max_samples_per_file = 2000

# This should give us ~75% qualified files
min_rate = 20 # Minimum acceptable frame rage [fps]
min_frames = 30 # Minimum number of frames at min_rate (1.5 s)
max_frame_time = 1/min_rate*1e3 # Maximum frame time [ms]

#%% Support functions

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def im_scale(im):
    """ convert single images to uint8 and contrast en"""
    # We can do other things here: e.g. background subtraction or contrast enhancement
    im_scaled = np.uint8((im - np.amin(im))/(np.amax(im) - np.amin(im))*256)
    #im_scaled_eq = cv2.equalizeHist(im_scaled)
    return im_scaled

def data2imarray(im_data):
    """
    apply imscale function to np.array
    arg: im_array (frame, height, width)
    returns: im_array (height, width, frame)
    """
    im_data = np.squeeze(im_data)
    im_list = [im_scale(im_data[im]) for im in range(im_data.shape[0])]
    im_array = np.array(im_list, dtype=np.uint16)
    im_array = np.moveaxis(im_array, 0, -1)
    return im_array

def subsample_time_index_list(frame_time, default_rate, min_frames):
    """
    rate: data frame rate,
    default_rate: desired frame rate,
    n_frames: number frames in the default rate (30)
    """
    default_times = np.arange(0, min_frames, 1) / default_rate
    times = np.arange(0, default_times[-1] + frame_time, frame_time)
    time_index_list = [np.argmin(np.abs(times - t)) for t in default_times]

    return time_index_list

def subsample_video(image_array, frame_time, min_rate, min_frames):
    """
    Select frames that are closest to a constant frame rate
    arg: image_array: np.array() [rows, columns, frame]
    """
    convert_video = True
    rate = 1 / frame_time
    # Check if the video is long enough
    min_video_len = min_frames / min_rate
    video_len = image_array.shape[-1] / rate
    if (min_video_len <= video_len) & (min_rate < rate):
        # print('Video is long enough and the rate is good.')
        # Get the frame index list
        time_index_list = subsample_time_index_list(frame_time=frame_time,
                                                    default_rate=min_rate,
                                                    min_frames=min_frames)
        # Select the frames from the video
        image_array = image_array[:, :, time_index_list]
    else:
        print('Frame rate: {:.1f} fps, length: {:.1f} s. Skipping.'.format(rate, video_len))
        convert_video = False

    return convert_video, image_array

#%% Select one view and process files

view = 'view_a4c'
#for view in meta_df.max_view.unique():

for mode in meta_df.dset.unique():

    # Filter view, mode and rates. Shuffle.
    df = meta_df[(meta_df.max_view == view) & (meta_df.dset == mode) & (meta_df.frame_time < max_frame_time)].\
                                                                                                sample(frac=1)
    print('View:{}, mode:{}, min_rate:{}, n_videos:{}'.format(view, mode, min_rate, len(df.filename.unique())))

    file_list_complete = list(df.filename.unique())
    # Split filename_list into multiple parts
    file_list_parts = list(chunks(file_list_complete, max_samples_per_file))
    mag = int(np.floor(np.log10(len(file_list_parts)))) + 1

    # Each part will have its own TFR filename
    for part, file_list in enumerate(file_list_parts):
        print()
        print('Processing TFR part {} of {}'.format(part+1, len(file_list_parts)))

        # TFR filename
        tfr_basename = 'CFR_'+meta_date+'_'+view+'_'+mode+'_'+str(part).zfill(mag)
        tfr_filename = tfr_basename+'.tfrecords'
        parquet_filename = tfr_basename+'.parquet'

        im_array_list = [] # list of image arrays [row, col, frame]
        im_array_ser_list = [] # list of pd.Series object for the files in im_array_list
        im_failed_ser_list = [] # list of pd.Series objects for failed videos
        cfr_list = []
        record_list = []

        for f, filename in enumerate(file_list):

            if (f+1)%200==0:
                print('Loading video {} of {} into memory.'.format(f+1, len(file_list)))

            ser_df = df.loc[df.filename == filename, :]
            ser = ser_df.iloc[0]
            file = os.path.join(ser.dir, filename)

            try:
                with lz4.frame.open(file, 'rb') as fp:
                    data = np.load(fp)

            except IOError as err:
                print('Could not open this file: {}\n {}'.format(file, err))
                im_failed_ser_list.append(ser_df)
            else:
                im_array_original = data2imarray(data)
                frame_time = ser.frame_time * 1e-3
                convert_video, im_array = subsample_video(image_array=im_array_original,
                                                          frame_time=frame_time,
                                                          min_rate=min_rate,
                                                          min_frames=min_frames)
                if convert_video:
                    # SUCCESS: save this video in list
                    im_array_list.append(im_array)
                    cfr_list.append(ser.cfr)
                    record_list.append(ser.name)
                    im_array_ser_list.append(ser_df)
                else:
                    im_failed_ser_list.append(ser_df)

        # Write TFR file
        TFR_saver = Dset(data_root=os.path.join(cfr_data_root,'tfrdata'))
        print()
        TFR_saver.create_tfr(filename=tfr_filename,
                             image_data=im_array_list,
                             cfr_data=cfr_list,
                             record_data=record_list)

        # When this is done, save the parquet file
        im_array_df = pd.concat(im_array_ser_list)
        im_array_df.to_parquet(os.path.join(cfr_data_root, 'tfrdata', parquet_filename))
