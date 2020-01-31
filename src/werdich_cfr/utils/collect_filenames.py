""" Collect MRN and dates from echo filenames """
import os
import glob
import numpy as np
from Crypto.Cipher import AES
import pandas as pd
import time

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 500)

#%% Files, directories and parameters
cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
cfr_echo_dir = os.path.normpath('/mnt/obi0/phi/echo/npyFiles/BWH')
cfr_feather_dir = os.path.normpath('/mnt/obi0/phi/echo/featherFiles/BWH')
key='rahuldeoechobwh*'
iv='echoisexcellent*'

def collect_files(topdir, file_pattern = '*.npy.lz4'):

    """ Collects the file names in all sub-directories of dir """

    print('Start collecting files...')
    start_time = time.time()
    file_list = list()
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(topdir)):
        file_list += glob.glob(os.path.join(dirpath, file_pattern))
        #file_list += [os.path.join(dirpath, file) for file in filenames]
        if (i+1)%1000 == 0:
            print('Completed {} directories in {:.1f} seconds.'.format(i+1, time.time()-start_time))

    print('Completed. This took {:.1f} seconds.'.format(time.time()-start_time))

    # Stick it in a data frame
    df = pd.DataFrame({'path': file_list})
    df = df.assign(filename=df.path.apply(lambda f: os.path.basename(f)),
                   dir=df.path.apply(lambda f: os.path.dirname(f))).drop(columns=['path']). \
        reset_index(drop=True)

    return df

def decodeMRN(MRN, key, iv):
    decipher = AES.new(key, AES.MODE_CFB, iv)
    MRN=bytes.fromhex(MRN)
    oldmrn = decipher.decrypt(MRN).decode()
    # oldmrn = unhexlify(MRN)
    return oldmrn

def decode_file(filename):
    """ Decode a filename in [mnr, datetime] """
    base_name_list = [filename.split('_')[i] for i in range(2)]
    study = base_name_list[0] + '_' + base_name_list[1]
    mrn_date = [decodeMRN(encoded, key = key, iv = iv) for encoded in base_name_list]
    return study, mrn_date

def add_base_name_mrn_datetime(df):

    # Before we can split the filename in mrn and date
    # We want to be sure that at least it has two parts
    df = df.assign(l = df.filename.apply(lambda f: len(f.split('_'))))
    df = df.drop(df.loc[df.l<2].index).drop(columns = ['l'], axis = 1).reset_index(drop = True)

    # Now we should have only rows in the filename column that can be split into at least two parts
    df = df.assign(study=df.filename.apply(lambda s: decode_file(s)[0]),
                   mrn=df.filename.apply(lambda s: decode_file(s)[1][0]),
                   datetime=df.filename.apply(lambda s: decode_file(s)[1][1]))

    df.datetime = pd.to_datetime(df.datetime, infer_datetime_format=True)

    return df

#%% Run the search

npy_file_list_name = 'echo_npyFiles_BWH_200131.parquet'
df_npy_file = collect_files(cfr_echo_dir)
df_npy_file_2 = add_base_name_mrn_datetime(df_npy_file)
df_npy_file_2.to_parquet(os.path.join(cfr_data_root, npy_file_list_name))

feather_file_list_name = 'echo_featherFiles_BWH_200131.parquet'
df_feather_file = collect_files(cfr_feather_dir)
df_feather_file_2 = add_base_name_mrn_datetime(df_feather_file)
df_feather_file_3 = df_feather_file_2.assign(dsc = df_feather_file_2.filename.apply(
    lambda s: os.path.splitext(s)[0].replace(decode_file(s)[0], '')[1:]))
df_feather_file_3.to_parquet(os.path.join(cfr_data_root, feather_file_list_name))

df_feather_file_3 = df_feather_file_3.rename(columns = {'filename': 'meta_filename',
                                                        'dir': 'meta_dir'})
df_files = df_npy_file_2.merge(right = df_feather_file_3, how = 'left', on = ['study', 'mrn', 'datetime'])

# Join npy_file_list and feather_file_list
# Rename some of the feather data columns
npy_meta_name = 'echo_BWH_npy_feather_files_201031.parquet'
df_files.to_parquet(os.path.join(cfr_data_root, npy_meta_name))
