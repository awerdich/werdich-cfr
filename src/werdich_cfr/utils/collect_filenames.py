""" Collect MRN and dates from echo filenames """
import os
import glob
import numpy as np
import pandas as pd
import time


#%% Files and folders
cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
cfr_echo_dir = os.path.normpath('/mnt/obi0/phi/echo/npyFiles/BWH')
#test_dir = os.path.join(cfr_echo_dir, '48bd')

def collect_files(topdir):

    """ Collects the file names in all sub-directories of dir """

    print('Start collecting files...')
    start_time = time.time()
    file_list = list()
    for (dirpath, dirnames, filenames) in os.walk(topdir):
        file_list += [os.path.join(dirpath, file) for file in filenames]

    print('Completed. This took {:.2f} seconds.'.format(time.time()-start_time))

    # Stick it in a data frame
    df = pd.DataFrame({'path': file_list})
    df = df.assign(filename=df.path.apply(lambda f: os.path.basename(f)),
                   dir=df.path.apply(lambda f: os.path.dirname(f))).drop(columns=['path']). \
        reset_index(drop=True)

    return df

#%% Save as pd data frame
df = collect_files(cfr_echo_dir)
file_list_name = 'echo_npyFiles_BWH.parquet'
df.to_parquet(os.path.join(cfr_data_root, file_list_name))
