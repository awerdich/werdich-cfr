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
meta_date = '200208'
# Additional information for filename
tfr_info = 'scaled'
tfr_dir = os.path.join(cfr_data_root, 'tfr_'+meta_date)
meta_dir = os.path.join(cfr_data_root, 'metadata_'+meta_date)
cfr_meta_file = 'tfr_files_dset_BWH_'+meta_date+'.parquet'
meta_df_original = pd.read_parquet(os.path.join(meta_dir, cfr_meta_file))

# Get rid of files with invalid scale factors

print('Original number of files {}'.format(len(meta_df_original.filename.unique())))
meta_df = meta_df_original[(0 < meta_df_original.deltaX) & (meta_df_original.deltaX < 1) &
                           (0 < meta_df_original.deltaY) & (meta_df_original.deltaY < 1)]
print('After removing invalid scale factors {}'.format(len(meta_df.filename.unique())))

max_samples_per_file = 2000

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

def im_scale(im, dx, dy):
    """ convert single images to uint8 and resize by scale factors """
    # We can do other things here: e.g. background subtraction or contrast enhancement
    im_scaled = np.uint8((im - np.amin(im))/(np.amax(im) - np.amin(im))*256)
    #im_scaled_eq = cv2.equalizeHist(im_scaled) # histogram equalization (not needed)
    if (dx is not None) & (dy is not None):
        width = int(np.round(im_scaled.shape[1]*10*dx))
        height = int(np.round(im_scaled.shape[0]*10*dy))
        im_resized = cv2.resize(im_scaled, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        im_resized = im_scaled
    return im_resized

def data2imarray(im_data, dx=None, dy=None):
    """
    apply imscale function to np.array
    arg: im_array (frame, height, width)
    returns: im_array (height, width, frame)
    """
    im_data = np.squeeze(im_data)
    im_list = [im_scale(im_data[im], dx, dy) for im in range(im_data.shape[0])]
    im_array = np.array(im_list, dtype=np.uint16)
    im_array = np.moveaxis(im_array, 0, -1)
    return im_array

def subsample_time_index_list(frame_time, default_rate, n_frames):
    """
    frame_time: time interval between frames [s]
    default_rate: matching frame rate [fps],
    n_frames: number of frames in the output
    """
    default_times = np.arange(0, n_frames, 1) / default_rate
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
                                                    n_frames=min_frames)
        # Select the frames from the video
        image_array = image_array[:, :, time_index_list]
    else:
        convert_video = False

    return convert_video, image_array

#%% Select one view and process files

view = 'a4c'

#for view in meta_df.max_view.unique():

for mode in meta_df['mode'].unique():

    # Filter view, mode and rates. Shuffle.
    df = meta_df[(meta_df.max_view == view) & (meta_df['mode'] == mode) & (meta_df.frame_time < max_frame_time)].\
                                                                                                sample(frac=1)
    print('View:{}, mode:{}, min_rate:{}, min_length: {}, n_videos:{}'.format(view,
                                                                              mode,
                                                                              min_rate,
                                                                              min_length,
                                                                              len(df.filename.unique())))

    file_list_complete = list(df.filename.unique())
    # Split filename_list into multiple parts
    file_list_parts = list(chunks(file_list_complete, max_samples_per_file))
    mag = int(np.floor(np.log10(len(file_list_parts)))) + 1

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
                im_array_original = data2imarray(data, dx=ser.deltaX, dy=ser.deltaY)
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
                    ser_df2 = ser_df.assign(im_array_shape=[im_array.shape])
                    im_array_ser_list.append(ser_df)
                else:
                    print('{} w/ rate: {} fps and duration: {:.2f} s. Skipping'.\
                          format(filename,
                                 ser.rate,
                                 data.shape[0]*ser.frame_time*1e-3))

                    im_failed_ser_list.append(ser_df)

        # Write TFR file
        TFR_saver = Dset(data_root=tfr_dir)
        print()
        TFR_saver.create_tfr(filename=tfr_filename,
                             image_data=im_array_list,
                             cfr_data=cfr_list,
                             record_data=record_list)

        # When this is done, save the parquet file
        im_array_df = pd.concat(im_array_ser_list)
        im_array_df.to_parquet(os.path.join(tfr_dir, parquet_filename))
