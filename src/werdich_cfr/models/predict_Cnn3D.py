import os
import glob
import pickle

from tensorflow.keras.models import load_model

# Custom imports
from werdich_cfr.tfutils.Modeltrainer import VideoTrainer

#%% Support functions

def read_model_dict(file):
    with open(file, 'rb') as f:
        model_dict = pickle.load(f)
    return model_dict

#%% Directories and data sets
cfr_data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
log_dir = os.path.join(cfr_data_root, 'log', '20020a4c_pad')
test_data_dir = os.path.join(cfr_data_root, 'tfr_200202')
model_dict_file = glob.glob(os.path.join(log_dir, '*_model_dict.pkl'))[0]
model_dict = read_model_dict(model_dict_file)

# Re-create the model from checkpoint
checkpoint_file_list = sorted(glob.glob(os.path.join(log_dir, model_dict['name']+'*_chkpt_*')))
checkpoint_file = checkpoint_file_list[14]

# Re-create the model from checkpoint
model = load_model(checkpoint_file)
model = model.load_weights(checkpoint_file)

estimator = VideoTrainer(log_dir = log_dir,
                         model_dict = model_dict,
                         train_dict = None)

test_tfr_files = sorted(glob.glob(os.path.join(test_data_dir, 'CFR_200202_view_a4c_test_*.tfrecords')))
test_parquet_files = [file.replace('.tfrecords', '.parquet') for file in test_tfr_files]







