""" Create a metadata lookup table for all npy files """
import os
import pandas as pd
import time

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 500)

#%% files and paths
cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
file_df_file = 'echo_BWH_npy_feather_files.parquet'

#%% Load the file names
file_df = pd.read_parquet(os.path.join(cfr_data_root, file_df_file))

#%% Filter the feather files that are needed
feather_dsc_list_original = list(file_df.dsc.unique())
feather_dsc_list = ['video_metadata_withScale', 'viewPredictionsVideo_withRV', 'study_metadata']
file_df2 = file_df[file_df.dsc.isin(feather_dsc_list)]
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
    mdf_video = get_metadata(df_file = df, study = study, metacol = 'video_metadata_withScale')
    mdf_view = get_metadata(df_file = df, study = study, metacol = 'viewPredictionsVideo_withRV')

    if (mdf_video.shape[0]>0) & (mdf_view.shape[0]>0):

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


#%% Run the function.
df_meta_list = [] # Collect filenames with meta data
study_list = sorted(list(file_df2.study.unique()))
meta_filename = 'echo_BWH_meta_100.parquet'
start_time = time.time()
for s, study in enumerate(study_list):
    df_meta_study = collect_meta_study(file_df2, study = study)
    if df_meta_study.shape[0] > 0:
        df_meta_list.append(df_meta_study)
    else:
        print('Not enough meta data for study {}. Skipping.'.format(study))
    if (s % 100 == 0):
        print('Study {} of {}, time {:.1f} seconds.'.format(s + 1,
                                                len(study_list),
                                                time.time()-start_time))

# Concat all data frames
df_meta = pd.concat(df_meta_list, ignore_index = True).reset_index(drop = True)
df_meta.to_parquet(os.path.join(cfr_data_root, meta_filename))
