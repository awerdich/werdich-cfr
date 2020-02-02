import os
import numpy as np
import lz4.frame
import pandas as pd

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

#%% files and directories
cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
cfr_meta_dir = os.path.join(cfr_data_root, 'metadata_200131')
cfr_meta_file = '210_getStressTest_files_dset_BWH_200131.parquet'
meta_df = pd.read_parquet(os.path.join(cfr_meta_dir, cfr_meta_file))
max_samples_per_file = 1500

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
filename = df.filename.iloc[100]
ser = df.loc[df.filename == filename, :]
dir = ser.dir.values[0]
file = os.path.join(dir, filename)
try:
    with lz4.frame.open(file, 'rb') as fp:
        data = np.load(fp)
except IOError as err:
    print('Could not open this file: {}\n {}'.format(file, err))





