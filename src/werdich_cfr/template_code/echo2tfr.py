import os
import glob
import pandas as pd
import numpy as np
import gzip
import pickle
from scipy import stats
from random import shuffle
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

import matplotlib
#import string

matplotlib.use('TKAgg')
from matplotlib import pyplot as plt

import tensorflow as tf

print('Tensorflow version:', tf.__version__)

# Custom import
from tensorflow_cfr.tfutils.TFRprovider import Dset

#%% Help functions

def gather( df, key, value, cols ):
    id_vars = [ col for col in df.columns if col not in cols ]
    id_values = cols
    var_name = key
    value_name = value
    return pd.melt( df, id_vars, id_values, var_name, value_name )

def na_rows(df, column_name):
    ''' Identify rows with NA in column = column_name '''
    nan_list = list(df[column_name].isnull())
    return [idx for idx in range(len(nan_list)) if nan_list[idx] == True]

#%% paths and file names
data_root = os.path.normpath('/tf/data')
image_root = os.path.normpath('/tf/imagedata')
view = 'a4c_laocc'
image_dir = os.path.join(image_root, 'a4c_laocc')
patient_split_file = os.path.join(image_root, 'patient_split_0530a123.pkl')
filext = '_1'
maximum_samples_per_file = 1500 # This determines the size of the TFRecords data set file

# Load the cfr table
df_cfr = pd.read_csv(os.path.join(image_root, 'Identifier_CFR_study_round1-2-3_a2c_ft_hashed_mrn.txt'), sep = '\t')

#%% Calculate frame rates and add to cfr_table
# Add frame rates to df

def frame_time2frame_rate(frame_time):
    time_object = eval(frame_time)
    # check if this value is a list
    if type(time_object) == list:
        dt = stats.mode([float(n) for n in time_object])[0][0]
    else:
        dt = time_object

    frame_rate = np.round(1 / dt * 1e3, decimals=3)
    if np.isnan(frame_rate):
        print('Invalid frame rate.')
        raise ValueError
    return frame_rate

# Apply this function to the data frame
df_cfr = df_cfr.assign(frame_rate=df_cfr['ft(ms)'])
df_cfr.frame_rate = df_cfr['ft(ms)'].apply(frame_time2frame_rate)

# Video requirements
default_rate = 22.0
min_video_len = 1.5
min_frames = int(np.ceil(min_video_len*default_rate))

#%% Patient splits

def patientsplit(patient_list):

    train_test_split = 0.86
    train_eval_split = 0.90

    # Split train/test sets
    patient_list_train = np.random.choice(patient_list,
                                          size = int(np.floor(train_test_split*len(patient_list))),
                                          replace = False)
    patient_list_test = list(set(patient_list).difference(patient_list_train))
    train_test_intersection = set(patient_list_train).intersection(set(patient_list_test)) # This should be empty
    print('Intersection of patient_list_train and patient_list_test:', train_test_intersection)

    # Further separate some patients for evaluation
    patient_list_eval = np.random.choice(patient_list_train,
                                         size = int(np.ceil((1-train_eval_split)*len(patient_list_train))),
                                         replace = False)

    patient_list_train = set(patient_list_train).difference(patient_list_eval)
    train_eval_intersection = set(patient_list_train).intersection(set(patient_list_eval))
    print('Intersection of patient_list_train and patient_list_eval:', train_eval_intersection)

    # Show the numbers
    print('total patients:', len(patient_list))
    print('patients in set:', np.sum([len(patient_list_train),
                                     len(patient_list_eval),
                                     len(patient_list_test)]))
    print('patients in train:', len(patient_list_train))
    print('patients in eval:', len(patient_list_eval))
    print('patients in test:', len(patient_list_test))

    return patient_list_train, patient_list_eval, patient_list_test

# Check if the split file exists and load it
try:
    # Load it
    print('Attenpting to load patient split:', patient_split_file)
    with open(patient_split_file, mode = 'rb') as f:
        patient_split = pickle.load(f)

except IOError:
    print('File not found. Unable to continue.')


# Use this to generate new patient list.
#patient_list = list(df_cfr.hashed_mrn.unique())
# Run the split
#patient_list_train, patient_list_eval, patient_list_test = patientsplit(patient_list)
#patient_split = {'train': patient_list_train,
#                 'eval': patient_list_eval,
#                 'test': patient_list_test}
# Save patient lists
#with open(patient_split_file, mode = 'wb') as f:
#    pickle.dump(patient_split, f, protocol = pickle.HIGHEST_PROTOCOL)
#print('New patient split saved:', patient_split_file)

# Check for contamination
print('contamination train-test:', set(patient_split['train']).intersection(set(patient_split['test'])))
print('contamination train-eval:', set(patient_split['train']).intersection(set(patient_split['eval'])))
print('contamination eval-test:', set(patient_split['eval']).intersection(set(patient_split['test'])))

print('Patient IDs in train:', len(patient_split['train']))
print('Patient IDs in eval:', len(patient_split['eval']))
print('Patient IDs in test:', len(patient_split['test']))

#%% WE DO NOT NEED TO SPLIT THE VIDEO COLUMN FOR THIS DF

# Drop the empty rows
print('Missing data:', df_cfr.isnull().values.any())
df_cfr = df_cfr.dropna(axis = 0)

# Filter df_cfr by those video files that are in the image_dir
disk_files = set(os.listdir(image_dir))
df_cfr_ondisk = df_cfr[df_cfr.study.isin(disk_files)]
print('For video directory {vd}'.format(vd = image_dir))
print('Number of video files in image_dir: {}'.format(len(list(disk_files))))
print('Video files found in df_cfr:', df_cfr_ondisk.shape[0])
print('Filtered by view {}: {}'.format(view, df_cfr_ondisk[df_cfr_ondisk.view == view].shape[0]))

print()
for dset in sorted(patient_split.keys()):
    df_split = df_cfr_ondisk[df_cfr_ondisk.hashed_mrn.isin(patient_split[dset])]
    print('Patients in {d}: {p} videos:{v}'.format(d = dset, p=len(df_split.hashed_mrn.unique()), v = df_split.shape[0]))

#%% Collect the data

def subsample_time_index_list(rate, default_rate=30, n_frames=30):
    '''rate: data frame rate,
       default_rate: desired frame rate,
       n_frames: number frames in the default rate (30)'''

    default_times = np.arange(0, n_frames, 1) / default_rate
    dt = 1 / rate
    times = np.arange(0, default_times[-1] + dt, dt)
    time_index_list = [np.argmin(np.abs(times - t)) for t in default_times]

    if rate < default_rate:
        print('frame rate <', default_rate, '! Undersampling.')

    return time_index_list


def subsample_video(image_array, file, default_rate=30, min_frames=30):

    convert_video = True

    # Check if there is a frame rate
    rate = df_cfr[df_cfr.study == file].frame_rate.values
    if rate.size > 0:
        rate = rate[0]
        #print('rate:', rate)
        # Check if the video is long enough
        video_len = image_array.shape[-1] / rate
        if (min_video_len <= video_len) & (default_rate < rate):
            #print('Video is long enough and the rate is good.')
            # Get the frame index list
            time_index_list = subsample_time_index_list(rate=rate,
                                                        default_rate=default_rate,
                                                        n_frames=min_frames)
            # Select the frames from the video
            image_array = image_array[:, :, time_index_list]

        else:
            print('Video is too short or the rate is too low.')
            convert_video = False

    else:
        print('No frame rate available.')
        # At least, we should have 30 frames
        # Check the number of frames, and if there are enough, we can return 30 frames
        if (image_array.shape[-1] > min_frames):
            print('But we have enough frames. Good to go.')
            image_array = image_array[:, :, :min_frames]
        else:
            print('Only', image_array.shape[-1], 'frames in file:\n', filepath, '\nNot enough! -- skipping.')
            convert_video = False

    return convert_video, image_array

def imdict2array(imdict):
    ''' Create an uint16 image array from a dictionary
    movies are scaled to 8 bit. We need uint16 for tfrecords generation
    output shape is [height, width, frames]'''

    image_list = [imdict[key] for key in sorted(imdict.keys())]
    image_list_scaled = [np.uint8((im - np.amin(im))/(np.amax(im) - np.amin(im))*256) for im in image_list]
    image_array = np.array(image_list_scaled, dtype = np.uint16)
    image_array = np.moveaxis(image_array, 0, -1)
    return image_array

def files2tfrecords(df, image_root, mode, view, default_rate, min_frames):

    ''' convert image files to TFRecord data

    df_files: pandas data frame with all file names for one view
    patient_list: list patients selected from df_files
    image_root: folder where the images are stored
    mode: 'train', 'eval' or 'test'
    view: 'a4c' or 'plax'
    min_frames: frames cutoff.

    Images are stored as TFRecords with shape [height, width, min_frames]
    '''


    # Video list of video filenames an shuffle it
    vlist = list(set(df.study))
    shuffle(vlist)
    videos = np.array(vlist)
    print('Processing {} videos for {} patients'.format(len(videos), len(df.hashed_mrn.unique())))

    # Split it into equal chunks
    n_parts = int(np.floor(len(videos)/maximum_samples_per_file))

    if n_parts > 0:
        video_parts = np.array_split(videos, n_parts)
        print('Dividing video list into', n_parts, 'parts.')
    else:
        video_parts = []
        video_parts.append(videos)
        n_parts+=1

    csv_file_list = []
    # This will be a loop over the video_parts
    for part in range(n_parts):

        # TFR file name
        tfr_file = mode + '_' + view + filext + '_p' + str(part) + '.tfrecords'
        csv_file = os.path.splitext(tfr_file)[0] + '.csv'

        # Video file list
        video_list = list(video_parts[part])
        # Data frame should have been filterd py patients and view
        df_part = df[df.study.isin(video_list)]

        print()
        print('File {} of {}.'.format(part+1, n_parts))
        print('Loading {} videos from {} patients into memory. Please wait'.format(len(video_list),
                                                                                   len(df_part.hashed_mrn.unique())))

        # Load the video file into a list of numpy arrays with the frames in the correct sequence
        df_tfr = pd.DataFrame()
        image_list = []
        study_list = []
        id_list = []
        cfr_list = []
        record_list = []

        for idx, file in enumerate(video_list):

            if ((idx+1) % 10) == 0:
                print('Video:', idx+1, 'of', len(video_list))

            filepath = os.path.join(image_root, file)
            try:
                with gzip.open(filepath, mode='rb') as f:
                    imdict = pickle.load(f, encoding='bytes')

                image_array_original = imdict2array(imdict)

                convert_video, image_array = subsample_video(image_array_original,
                                                             file = file,
                                                             default_rate = default_rate,
                                                             min_frames = min_frames)

                # Check the outcome of the subsample_video function
                # Making sure that we have a minum number of frames at the default rate
                if convert_video:

                    # Collect the data
                    image_list.append(image_array)

                    # Collect other data based on the video_list
                    study = df[df.study == file].study.values[0]
                    id = df[df.study == file].hashed_mrn.values[0]
                    cfr = float(df[df.study == file].cfr.values[0])
                    study_list.append(study)
                    id_list.append(id)
                    cfr_list.append(cfr)
                    record_list.append(idx)

                    tfr_dict = {'patient_ID': [id],
                                'record_ID': [idx],
                                'study': [study],
                                'cfr': [cfr],
                                'view': [view],
                                'mode': [mode],
                                'frames': [image_array.shape[2]],
                                'rows': [image_array.shape[0]],
                                'cols': [image_array.shape[1]],
                                'video_file': [file],
                                'tfr_file': [tfr_file]}

                    df_tfr = pd.concat([df_tfr, pd.DataFrame(tfr_dict)], ignore_index=True)

                else:
                    print('Video does not meet minimum requirements -- skipping.\n', filepath)

            except IOError as e:
                print('Unable to open file:', filepath, '-- skipping.')

        # Save the TFR file
        TFR_saver = Dset(data_root = data_root)
        TFR_saver.create_tfr(tfr_file, image_list, cfr_list, record_list)

        # Save the df as .csv
        df_tfr.to_csv(os.path.join(data_root, csv_file), index = False)
        csv_file_list.append(csv_file)

    return csv_file_list

#%% Run the conversion

def csv_summary(csv_file):
    df = pd.read_csv(os.path.join(data_root, csv_file))
    print()
    print('File:', csv_file)
    print('tfr_file:', df.iloc[0].tfr_file)
    print('View:', df.iloc[0]['view'])
    print('Mode:', df.iloc[0]['mode'])
    print('Unique patients:', len(df.patient_ID.unique()))
    print('Samples:', len(df.record_ID.unique()))

# Convert video data

for mode in patient_split.keys():

    filtered_df_cfr_ondisk = df_cfr_ondisk[(df_cfr_ondisk.view == view) & (df_cfr_ondisk.hashed_mrn.isin(patient_split[mode]))]
    print('There are {v} videos for {p} patients in split {d}'.format(d = mode,
                                                                      p=len(filtered_df_cfr_ondisk.hashed_mrn.unique()),
                                                                      v = filtered_df_cfr_ondisk.shape[0]))

    #args(df, image_root, mode, view, default_rate, min_frames):
    csv_file_list = files2tfrecords(df = filtered_df_cfr_ondisk,
                                    image_root = image_dir,
                                    mode = mode,
                                    view = view,
                                    default_rate = default_rate,
                                    min_frames = 30)
    for csv_file in csv_file_list:
        csv_summary(csv_file)
