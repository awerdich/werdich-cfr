# Imports
import os
import glob
import pickle
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
from pdb import set_trace
import tensorflow as tf

from tensorflow_cfr.tfutils.TFRtrainer import VideoTrainer

#%% Files and paths
data_root = os.path.normpath('/tf/data')
image_dir = os.path.normpath('/tf/imagedata')
log_dir = os.path.normpath('/tf/log')
model_log_dir = os.path.join(log_dir, '3catw1cfrw9do_sp2_a4c_201906030150')
checkpoint_file = os.path.join(model_log_dir, '3catw1cfrw9do_sp2_a4c_chkpt_34.hdf5')
model_name = '3catw1cfrw9do_sp2_a4c'
cfr_boundaries = (1.355, 1.874)
view = 'a4c'
test_batch_size = 12

# NETWORK AND TRAINING PARAMETERS AS DICTIONARIES
model_dict = dict() # Model parameters (e.g. filters)
model_dict['name'] = model_name
model_dict['view'] = view
model_dict['im_size'] = (200, 200, 1)
model_dict['cl_outputs'] = len(cfr_boundaries)+1
model_dict['filters'] = 32
model_dict['pool_nodes'] = 256
model_dict['fc_nodes'] = 256
model_dict['fullnet'] = True
model_dict['cat_reg_outputs'] = True

train_dict = dict() # Training parameters (e.g. optimizer, learning rate)
train_dict['learning_rate'] = 0.0001
train_dict['loss_weights_class_output'] = 1.0
train_dict['loss_weights_score_output'] = 9.0

#%% Load model from checkpoint

# Recreate the exact same model from checkpoint
model = tf.keras.models.load_model(checkpoint_file)

# Load the weights
model.load_weights(checkpoint_file)

#%% Select test data set

trainer = VideoTrainer(data_root = data_root,
                       log_dir = log_dir,
                       view = view,
                       model_dict = model_dict,
                       train_dict = train_dict,
                       cfr_boundaries = cfr_boundaries)

test_files = sorted(glob.glob(os.path.join(image_dir, 'test_' + view + '*' + '.tfrecords')))

#%%

n_test, test_set = trainer.build_datset(tfr_files = test_files,
                                        repeat_count = 1,
                                        batch_size = test_batch_size,
                                        shuffle = False)

# Get the predictions from the model
class_model, score_model = model.predict(test_set)
class_preds = list(np.argmax(class_model, axis = 1))
score_preds = list(score_model.flatten())

# Get the true labels from the data set
n_test_batches = int(np.ceil(n_test/test_batch_size))
class_labels = []
score_labels = []
for _, labels_tf in test_set.take(n_test_batches+1):
    class_batch = list(np.argmax(labels_tf['class_output'].numpy(), axis = 1))
    score_batch = list(labels_tf['score_output'].numpy())
    class_labels.extend(class_batch)
    score_labels.extend(score_batch)

# Load also the labels from the csv files
test_csv_files = trainer.tfr2csv_file_list(test_files)
df = trainer.load_csv_info(test_csv_files)

labels_df = pd.DataFrame({'cfrtfr_true': score_labels,
                          'class_true': class_labels,
                          'score_pred': score_preds,
                          'class_pred': class_preds})

df_pred = pd.concat([df, labels_df], axis = 1)

#%% Save to csv
csv_filename = model_name + '_testpreds.csv'
df_pred.to_csv(os.path.join(data_root, csv_filename), index = False)
