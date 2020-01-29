""" Collect MRN and dates from echo filenames """
import os
import glob
import numpy as np
import pandas as pd


#%% Files and folders
cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
cfr_echo_dir = os.path.normpath('/mnt/obi0/phi/echo/npyFiles/BWH')

listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(cfr_echo_dir):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]
