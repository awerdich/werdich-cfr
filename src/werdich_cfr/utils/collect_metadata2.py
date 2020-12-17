"""
Create a metadata lookup table for ALL existing npy files
Or just the MRNs for the project """

import os
import pandas as pd
import numpy as np
import time
import lz4.frame

#import pdb;pdb.set_trace()

#%% files and paths
cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
meta_date = '201215'
location = 'MGH'
meta_dir = os.path.join(cfr_data_root, 'metadata_'+meta_date)
file_df_file = 'echo_'+location+'_npy_feather_files_'+meta_date+'.parquet'

# Use the final data set to filter the studies that we need
#echo_npyFiles_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr/predictions_echodata/FirstEcho')
#echo_npyFiles = os.path.join(echo_npyFiles_dir, 'echo_BWH_npy_feather_files_200617.parquet')
#dataset = pd.read_parquet(echo_npyFiles)
#filter_study_list = list(dataset.study.unique())

# Output file
meta_filename = 'echo_'+location+'_meta_'+meta_date+'.parquet'

# Some feather files contain None where the file names should be. Lets collect them.
empty_feather_list = []
empty_feather_list_file = os.path.join(meta_dir, 'empty_view_classification_'+location+'_files.txt')

#%% Load the file names
file_df = pd.read_parquet(os.path.join(meta_dir, file_df_file))

#%% Filter the feather files that are needed
feather_dsc_list = ['video_metadata_withScale', 'viewPredictionsVideo_withRV', 'study_metadata']
file_df2 = file_df[file_df.dsc.isin(feather_dsc_list)]

# Filter by the list of echos that we need (SPECIAL)
#file_df2 = file_df2[file_df2.study.isin(filter_study_list)]

print('Collecting metadata from df: {}'.format(file_df_file))
print('Number of unique npy files: {}'.format(len(file_df2.filename.unique())))
print('Number of studies: {}'.format(len(file_df2.study.unique())))

#%% Collect meta data from file df

# Function to load meta data for a study
def get_study_metadata(study, meta_df):
    meta_df_study = meta_df[meta_df.study == study]
    meta_dsc_list = list(set(meta_df_study.dsc.unique()))

    meta_dict = {}

    for dsc in meta_dsc_list:

        m = meta_df_study[meta_df_study.dsc == dsc][['meta_dir', 'meta_filename']].drop_duplicates()
        dsc_file = os.path.join(m.meta_dir.values[0], m.meta_filename.values[0])

        try:
            with open(dsc_file, 'rb') as fl:
                df = pd.read_feather(fl)
        except IOError as err:
            print(err)
        else:
            if df.shape[0] > 0:
                meta_dict[dsc] = df

    return meta_dict

# Get a list of studies and then collect metadata for all files in the study
default_dsc_list = tuple(['video_metadata_withScale', 'viewPredictionsVideo_withRV', 'study_metadata'])
def get_study_metadata_files(study, meta_df, meta_dsc_list=default_dsc_list):
    study_df = pd.DataFrame()
    meta_dict = get_study_metadata(study=study, meta_df=meta_df)

    if len(meta_dict) > 0:
        meta_file_difference = set(meta_dsc_list).symmetric_difference(meta_dict.keys())
        # Open meta data files only if they are all resent. Otherwise, skip.
        if len(meta_file_difference) == 0:

            # Now we can collect the meta data for each file that we expect in this study
            meta_df_study = meta_df[meta_df.study == study]
            meta_df_study = meta_df_study.loc[~meta_df_study.filename.isnull()]
            meta_df_study = meta_df_study.assign(file_base=meta_df_study.filename.apply(lambda s: s.split('.')[0]))
            # One row per file. We need this for later.
            meta_df_study_file = meta_df_study.drop(columns=['meta_filename', 'meta_dir', 'dsc']).drop_duplicates()
            file_base_list = list(meta_df_study_file.file_base.unique())

            # Video meta data
            video_df = meta_dict['video_metadata_withScale']
            video_df = video_df.loc[~video_df.identifier.isnull()]
            video_df = video_df.assign(file_base=video_df.identifier.apply(lambda s: s.split('.')[0]))
            video_df_files = video_df[video_df.file_base.isin(file_base_list)].drop(columns=['index'])

            # View classification results
            view_df = meta_dict['viewPredictionsVideo_withRV']
            view_df = view_df.loc[~view_df['index'].isnull()]
            view_df = view_df.assign(file_base=view_df['index'].apply(lambda s: s.split('.')[0]))
            view_df = view_df.drop(columns=['index'])
            view_df_files = view_df[view_df.file_base.isin(file_base_list)]

            # Combine meta data and view classification and add study meta data
            study_df = video_df_files.merge(view_df_files, on='file_base', how='outer')
            study_meta = meta_dict['study_metadata'].drop(columns=['index'])
            study_df = study_df.merge(study_meta, on='study', how='left')

            # Add the original meta data for the study (with the file names and directories)
            study_df = meta_df_study_file.merge(study_df, on=['file_base', 'study'], how='left').reset_index(drop=True)

        else:
            print(f'Meta data file {meta_file_difference} missing. Skipping this study.')
    else:
        print(f'No meta data files for study {study}. Skipping this study.')

    return study_df

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
meta_missing_filename = meta_filename.split('.')[0]+'_missing_meta.parquet'

start_time = time.time()
for s, study in enumerate(study_list):

    if (s+1) % 100 == 0:
        print(f'Study {s + 1} of {len(study_list)}')

    df_meta_study = get_study_metadata_files(study=study, meta_df=file_df2)

    if df_meta_study.shape[0] > 0:
        # Collect the intensities
        #print(f'Collecting intensities from {len(df_meta_study.filename.unique())} files.')
        #df_meta_study_int = collect_intensities_from_study_files(df_meta_study)
        df_meta_list.append(df_meta_study)
    else:
        df_missing = file_df2[file_df2.study == study][['filename', 'dir', 'study']].drop_duplicates()
        df_missing_meta_list.append(df_missing)
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
