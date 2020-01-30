""" Create a metadata lookup table for all npy files """
import os
import pandas as pd
import time

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 500)

#%% files and paths
cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr/backup')
file_df_file = 'echo_BWH_npy_feather_files.parquet'

#%% Load the file names
file_df = pd.read_parquet(os.path.join(cfr_data_root, file_df_file))

#%% Filter the feather files that are needed
feather_dsc_list_original = list(file_df.dsc.unique())
print(*feather_dsc_list_original, sep = '\n')
feather_dsc_list = ['video_metadata_withScale', 'viewPredictionsVideo_withRV', 'study_metadata']


#%% Studies without metadata
#df_none_dsc = file_df.loc[file_df.dsc.isnull()]

#%% Take a look at the meta data
study = list(file_df.study.unique())[10]
print(study)
study_df = file_df[file_df.study == study]
print(study_df.shape)

# Open meta data for the first video in this study
study_df_meta = study_df[study_df.dsc == 'viewPredictionsVideo_withRV']
feather_file = os.path.join(study_df_meta.iloc[0].meta_dir, study_df_meta.iloc[0].meta_filename)

#%% Open study_metadata
study_metadata = pd.read_feather(feather_file)






