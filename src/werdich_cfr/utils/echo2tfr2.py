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
cfr_meta_dir = os.path.join(cfr_data_root, 'metadata_200131')
cfr_meta_file = '210_getStressTest_files_dset_BWH_200131.parquet'
meta_df = pd.read_parquet(os.path.join(cfr_meta_dir, cfr_meta_file))
max_samples_per_file = 15

#%% Support functions

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def imscale(im):
    """ convert single images to uint8 and contrast en"""
    # We can do other things here: e.g. background subtraction or contrast enhancement
    im_scaled = np.uint8((im - np.amin(im))/(np.amax(im) - np.amin(im))*256)
    #im_scaled_eq = im_scaled
    im_scaled_eq = cv2.equalizeHist(im_scaled)
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

def subsample_video(image_array, frame_time, default_rate=25, min_frames=40):
    convert_video = True
    rate = 1 / frame_time
    # Check if the video is long enough
    min_video_len = min_frames / default_rate
    video_len = image_array.shape[-1] / rate
    if (min_video_len <= video_len) & (default_rate < rate):
        # print('Video is long enough and the rate is good.')
        # Get the frame index list
        time_index_list = subsample_time_index_list(frame_time=frame_time,
                                                    default_rate=default_rate,
                                                    min_frames=min_frames)
        # Select the frames from the video
        image_array = image_array[:, :, time_index_list]
    else:
        print('Video is too short or the rate is too low.')
        convert_video = False

    return convert_video, image_array

#%% Select one view and process files
# There should be no empty rows. But sometimes this could happen (when new data is added to the drives)
# Remove empty rows (where we are missing view classification)
meta_df = meta_df.loc[~meta_df.max_view.isnull()]
view_list = sorted(list(meta_df.max_view.unique()))
mode_list = sorted(list(meta_df.dset.unique()))

# LOOP 1: VIEWS
view = view_list[2]

# LOOP 2: MODE
mode = mode_list[2]

# Filter view and mode. Shuffle.
df = meta_df[(meta_df.max_view == view) & (meta_df.dset == mode)].sample(frac = 1)

# LOOP 3: FILES loop over all file names
filename_list = list(df.filename.unique())[0:30]
# Split filename_list into multiple parts
file_name_list_parts = list(chunks(filename_list, max_samples_per_file))

filename = df.filename.iloc[101]
ser = df.loc[df.filename == filename, :]
dir = ser.dir.values[0]
file = os.path.join(dir, filename)
frame_time = ser.frame_time.values[0]*1e-3

try:
    with lz4.frame.open(file, 'rb') as fp:
        data = np.load(fp)

    im_array_original = im_array_scale(data)
    convert_video, im_array = subsample_video(im_array_original, frame_time)

except IOError as err:
    print('Could not open this file: {}\n {}'.format(file, err))
