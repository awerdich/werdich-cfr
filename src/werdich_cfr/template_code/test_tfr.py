import os
import pickle
import gzip
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt


from tensorflow_cfr.tfutils.TFRprovider import DatasetProvider

#%% Paths and files
data_root = os.path.normpath('/tf/USBRAID/projects/echoData/tfr')
# Load .csv file
df = pd.read_csv(os.path.join(data_root, 'train_a4c.csv'))
print(df[['patient_ID', 'mode', 'cfr', 'frames', 'rows', 'cols', 'tfr_file', 'record_ID']].head(10))
cfr_threshold = 1.8

#%% Load data from TFRecords

ds = DatasetProvider([os.path.join(data_root, 'train_a4c.tfrecords')],
                     n_frames = 30,
                     output_height = 299,
                     output_width = 299,
                     cfr_threshold = cfr_threshold,
                     n_epochs = 1)

batch_size = 8
ds = ds.make_batch(batch_size = batch_size, shuffle = False)

#%% Load a batch of data

for image_tf, cls_tf, cfr_tf, record_tf in ds.take(1):
    image_batch = image_tf.numpy()
    cls_batch = cls_tf.numpy()
    cfr_batch = cfr_tf.numpy()
    rec_batch = record_tf.numpy()

idx = np.random.randint(batch_size, size=1)[0]
im = image_batch[idx]
cls = np.argmax(cls_batch[idx])
cfr = cfr_batch[idx]
rec = rec_batch[idx]

print('record_ID:', rec)
print('cfr value:', cfr)
print('Image batch shape:', image_batch.shape)
print('Image mean:', np.mean(im))
print('Image min:', np.min(im))
print('Image max:', np.max(im))

fig, ax = plt.subplots(figsize = (5, 5))
ax.imshow(im[10])
plt.show()
