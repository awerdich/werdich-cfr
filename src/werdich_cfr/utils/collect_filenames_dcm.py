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
dcm_echo_dir = os.path.normpath('/mnt/obi0/phi/echo/deIdentifyedEcho/BWH')
key='rahuldeoechobwh*'
iv='echoisexcellent*'

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

    df = df.assign(study=df.fileid.apply(lambda s: decode_file(s)[0]),
                   mrn=df.fileid.apply(lambda s: decode_file(s)[1][0]),
                   datetime=df.fileid.apply(lambda s: decode_file(s)[1][1]))

    df.datetime = pd.to_datetime(df.datetime, infer_datetime_format=True)

    return df

#%% Load the file name lists

file_list_name = 'echo_deIdentifyedEcho_BWH.parquet'
#test_dir = os.path.normpath('/mnt/obi0/phi/echo/deIdentifyedEcho/BWH/48b0')
test_dir = os.path.normpath('/home/andreas/test')
df_file = collect_files(test_dir)
#df_file = collect_files(dcm_echo_dir)

# Decode MRN and STUDY DATE
df_file = df_file.assign(fileid = df_file.dir.apply(lambda f: os.path.basename(f)))
df_file = add_base_name_mrn_datetime(df_file)
#df_file.to_parquet(os.path.join(cfr_data_root, file_list_name))

