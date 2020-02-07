import os
import glob
import pickle

# Custom imports
from werdich_cfr.tfutils.Modeltrainer import VideoTrainer

#%% Support functions

def load_model_dict(file):
    with open(file, 'wb') as f:
        model_dict = pickle.load(f)
    return model_dict

#%% Directories and data sets
log_dir = os.path.normpath('/mnt/obi0/andreas/data/cfr/log')
model_dict_file = glob.glob(os.path.join(log_dir, '*_model_dict.pkl'))
model_dict = load_model_dict(model_dict_file)
