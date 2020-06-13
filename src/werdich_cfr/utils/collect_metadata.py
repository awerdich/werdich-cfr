"""
Create a metadata lookup table for ALL existing npy files
Or just the MRNs for the project """

import os
import pandas as pd
import numpy as np
import time
import lz4.frame

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 500)

#import pdb;pdb.set_trace()

#%% files and paths
cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
meta_date = '200606'
location = 'MGH'
meta_dir = os.path.join(cfr_data_root, 'metadata_'+meta_date)
file_df_file = 'echo_'+location+'_npy_feather_files_'+meta_date+'.parquet'

# Filter by the studies that we need
# echo list
#echo_list_dir = os.path.normpath('/mnt/obi0/sgoto/BWHCFREvents/echoList')
#echo_list_filename = 'BWH_2015-05-01_2015-10-31_FirstEcho.txt'
#echo_list_file = os.path.join(echo_list_dir, echo_list_filename)
#echo_list = pd.read_csv(echo_list_file, header='infer', sep='\t')
#study_list = list(echo_list.study.unique())

# Output file
meta_filename = 'echo_'+location+'_meta_'+meta_date+'.parquet'

# Some feather files contain None where the file names should be. Lets collect them.
empty_feather_list = []
empty_feather_list_file = os.path.join(meta_dir, 'empty_view_classification_'+location+'_files.txt')


#%% Load the file names
file_df = pd.read_parquet(os.path.join(meta_dir, file_df_file))

#%% Filter the feather files that are needed
feather_dsc_list_original = list(file_df.dsc.unique())
feather_dsc_list = ['video_metadata_withScale', 'viewPredictionsVideo_withRV', 'study_metadata']
file_df2 = file_df[file_df.dsc.isin(feather_dsc_list)]

# Filter by the list of echos that we need (SPECIAL)
#file_df2 = file_df2[file_df2.study.isin(study_list)]

print('Collecting metadata from df: {}'.format(file_df_file))
print('Number of unique npy files: {}'.format(len(file_df2.filename.unique())))
print('Number of studies: {}'.format(len(file_df2.study.unique())))

#%% How many studies for each meta file
df_count = file_df2.groupby('dsc')['study'].nunique()
print(df_count)
# This shows that there is about one study per metadata. There seems to be a few studies without metadata.

#%% Collect meta data from file df

def get_metadata(df_file, study, metacol):
    df_studymeta = df_file[(df_file.study == study) & (df_file.dsc == metacol)]
    if df_studymeta.shape[0] > 0:
        metafile = os.path.join(df_studymeta.meta_dir.iloc[0], df_studymeta.meta_filename.iloc[0])
        meta_df = pd.read_feather(metafile)
        # Some of the view classification feather files have empty file names
        if len(meta_df.loc[meta_df['index'].isnull()]) > 0:
            empty_feather_list.append(metafile)
            meta_df = meta_df.loc[~meta_df['index'].isnull()]
    else:
        meta_df = pd.DataFrame()
    return meta_df

def collect_meta_study(df, study):

    # Base df without the meta filenames
    df_meta_study = df[df.study == study].drop(columns = ['meta_filename', 'meta_dir', 'dsc']).\
        drop_duplicates().reset_index(drop = True)
    df_meta_study = df_meta_study.assign(fileid = df_meta_study.filename.apply(lambda f: f.split('.')[0]))

    # Add study_metadata
    mdf = get_metadata(df_file = df, study = study, metacol = 'study_metadata')
    if mdf.shape[0]>0:
        df_meta_study = df_meta_study.assign(institution = mdf.institution.values[0],
                                             model = mdf.model.values[0],
                                             manufacturer = mdf.manufacturer.values[0])

    # Skip the whole extraction if there is no video metadata or view prediction
    mdf_video = get_metadata(df_file=df, study=study, metacol='video_metadata_withScale')
    mdf_view = get_metadata(df_file=df, study=study, metacol='viewPredictionsVideo_withRV')

    if (mdf_video.shape[0] > 0) & (mdf_view.shape[0] > 0):

        # Add video_metadata
        mdf_video = mdf_video.assign(fileid = mdf_video.identifier.apply(lambda f: f.split('.')[0])).\
            drop(columns = ['study', 'identifier'])
        df_meta_study = df_meta_study.merge(right = mdf_video, on = 'fileid', how = 'left').reset_index(drop = True)

        # Add view predictions
        mdf_view = mdf_view.assign(fileid = mdf_view['index'].apply(lambda f: f.split('.')[0])).\
            drop(columns = ['index'])
        df_meta_study = df_meta_study.merge(right = mdf_view, on = 'fileid', how = 'left').reset_index(drop = True)

    else:
        # Return an empty df if important metadata are missing
        df_meta_study = pd.DataFrame()
    return df_meta_study

def collect_intensities_from_study_files(df_meta_study):
    # open the files and get the intensities
    files_s_int_list = []
    for filename in df_meta_study.filename.unique():

        file_s = df_meta_study[df_meta_study.filename==filename].iloc[0]
        file = os.path.join(file_s.dir, filename)

        try:
            with lz4.frame.open(file, 'rb') as fp:
                data = np.load(fp)
        except IOError as err:
            print('Cannot open npy file.')
            print(err)
        else:
            file_s['min_data'] = np.amin(data)
            file_s['max_data'] = np.amax(data)
            file_s['mean_data'] = np.mean(data)
            file_s['std_dat'] = np.std(data)
            files_s_int_list.append(file_s.to_frame().transpose())

    df = pd.concat(files_s_int_list, axis = 0)

    return df

#%% Run the function.
df_meta_list = [] # Collect filenames with meta data
df_missing_meta_list = [] # Collect studies with missing meta data

study_list = sorted(list(file_df2.study.unique()))
meta_missing_filename = meta_filename.split('.')[0]+'_missing.parquet'

start_time = time.time()
for s, study in enumerate(study_list):

    if (s+1) % 100 == 0:
        print(f'Study {s + 1} of {len(study_list)}')

    df_meta_study = collect_meta_study(file_df2, study = study)

    if df_meta_study.shape[0] > 0:
        # Collect the intensities
        #print(f'Collecting intensities from {len(df_meta_study.filename.unique())} files.')
        #df_meta_study_int = collect_intensities_from_study_files(df_meta_study)
        df_meta_list.append(df_meta_study)
    else:
        print('Not enough meta data for study {}. Skipping.'.format(study))
        df_missing_meta_list.append(file_df2[file_df2.study == study])
    if (s+1) % 100 == 0:
        print('Study {} of {}, time {:.1f} seconds.'.format(s + 1, len(study_list), time.time()-start_time))

# Concat all data frames
df_meta = pd.concat(df_meta_list, ignore_index = True).reset_index(drop = True)

if len(df_missing_meta_list)>0:
    df_missing_meta = pd.concat(df_missing_meta_list, ignore_index=True).reset_index(drop=True)
    # Here are the .npy files that are missing meta data
    # Missing either video_metadata_withScale OR viewPredictionsVideo_withRV
    df_missing_meta.to_parquet(os.path.join(meta_dir, meta_missing_filename))

# Make sure that we have consistent types in numeric columns
num_cols = ['frame_time', 'number_of_frames', 'heart_rate', 'deltaX', 'deltaY']
df_meta.loc[:, num_cols] = df_meta[num_cols].apply(pd.to_numeric, errors = 'coerce')
df_meta.to_parquet(os.path.join(meta_dir, meta_filename))

# List of view classification files that contain empty rows
pd.DataFrame(empty_feather_list).to_csv(empty_feather_list_file, header=False, index=False)
