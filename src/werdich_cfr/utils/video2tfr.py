import os
import numpy as np
import lz4.frame
import cv2
import pandas as pd

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

#%% files and directories
cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
meta_date = '200131'
meta_dir = os.path.join(cfr_data_root, 'metadata_'+meta_date)
cfr_meta_file = '210_getStressTest_files_dset_BWH_'+meta_date+'.parquet'
meta_df = pd.read_parquet(os.path.join(meta_dir, cfr_meta_file))
max_samples_per_file = 15
min_rate = 25 # Minimum acceptable frame rage
min_frames = 40 # Minimum number of frames at min_rate (1.6 s)

#%% Support functions

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def imscale(im):
    """ convert single images to uint8 and contrast en"""
    # We can do other things here: e.g. background subtraction or contrast enhancement
    im_scaled = np.uint8((im - np.amin(im))/(np.amax(im) - np.amin(im))*256)
    im_scaled_eq = im_scaled
    #im_scaled_eq = cv2.equalizeHist(im_scaled)
    return im_scaled_eq

def im_array_scale(im_data):
    """
    apply imscale function to np.array
    arg: im_array (frame, height, width)
    returns: im_array (height, width, frame)
    """
    im_list = [imscale(data[im]) for im in range(im_data.shape[0])]
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
        print('Frame rate: {:.1f}fps or length: {:.1f}s are note suitable. Skipping.'.format(rate, video_len))
        convert_video = False

    return convert_video, image_array

#%% Select one view and process files
# There should be no empty rows. But sometimes this could happen (when new data is added to the drives)

# Remove empty rows (where we are missing view classification)
meta_df = meta_df.loc[~meta_df.max_view.isnull()]

# Filter low rates and short videos

view_list = sorted(list(meta_df.max_view.unique()))
mode_list = sorted(list(meta_df.dset.unique()))

# LOOP 1: VIEWS
view = view_list[2]

# LOOP 2: MODE
mode = mode_list[2]

# Filter view and mode. Shuffle.
df = meta_df[(meta_df.max_view == view) & (meta_df.dset == mode)].sample(frac = 1)

# LOOP 3: FILES loop over all file names
file_list_complete = list(df.filename.unique())[0:30]
# Split filename_list into multiple parts
file_list_parts = list(chunks(file_list_complete, max_samples_per_file))
mag = int(np.floor(np.log10(len(file_list_parts)))) + 1

# Each part will have its own TFR filename
part = 0

# TFR filename
tfr_basename = 'CFR_'+view+'_'+mode+'_'+str(part).zfill(mag)
tfr_filename = tfr_basename+'.tfrecords'
parquet_filename = tfr_basename+'.tfrecords'
file_list = file_list_parts[part]

im_array_list = [] # list of image arrays [row, col, frame]
im_array_ser_list = [] # list of pd.Series object for the files in im_array_list
im_failed_ser_list = [] # list of pd.Series objects for failed videos
cfr_list = []
record_list = []

for f, filename in enumerate(file_list):

    if (f+1)%5==0:
        print('Loading video {} of {} into memory.'.format(f+1, len(file_list)))

    ser = df.loc[df.filename == filename, :].iloc[0]
    file = os.path.join(ser.dir, filename)

    try:
        with lz4.frame.open(file, 'rb') as fp:
            data = np.load(fp)

    except IOError as err:
        print('Could not open this file: {}\n {}'.format(file, err))
        im_failed_ser_list.append(ser)
    else:
        data = np.squeeze(data)
        im_array_original = im_array_scale(data)
        frame_time = ser.frame_time * 1e-3
        convert_video, im_array = subsample_video(image_array = im_array_original,
                                                  frame_time = frame_time,
                                                  min_rate = min_rate)
        if convert_video:
            # SUCCESS: save this video in list
            im_array_list.append(im_array)
            cfr_list.append(ser.cfr)
            record_list.append(ser.name)
            im_array_ser_list.append(ser)
        else:
            im_failed_ser_list.append(ser)
