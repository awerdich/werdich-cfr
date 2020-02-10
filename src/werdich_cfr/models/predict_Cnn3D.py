import os
import glob
import pickle
import pandas as pd

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
checkpoint_file_list_predict = [checkpoint_file_list[14], checkpoint_file_list[-1]]

test_tfr_files = sorted(glob.glob(os.path.join(test_data_dir, 'CFR_200202_view_a4c_test_*.tfrecords')))
test_parquet_files = [file.replace('.tfrecords', '.parquet') for file in test_tfr_files]
test_df = pd.concat([pd.read_parquet(file) for file in test_parquet_files], axis = 0)

#Instantiate the prediction
estimator = VideoTrainer(log_dir = log_dir,
                         model_dict = model_dict,
                         train_dict = None)

# Re-create the model from checkpoint
model = load_model(checkpoint_file_list[0])

for c, cfile in enumerate(checkpoint_file_list_predict):
    print('Running predictinos from checkpoint {} of {}: {}'.format(c+1, len(checkpoint_file_list_predict), 
                                                                    os.path.basename(cfile)))
    model.load_weights(cfile)
    class_output, score_output = estimator.predict(model, test_tfr_files, steps=None)
    test_df_predict = test_df.assign(cfr_predicted = score_output)
    predict_file = os.path.basename(cfile).split('.')[0]+'_predicted.parquet'
    test_df_predict.to_parquet(os.path.join(log_dir, predict_file))