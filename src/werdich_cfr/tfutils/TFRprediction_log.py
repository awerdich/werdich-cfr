""" TFR predictions on test set from log folder """
import os
import glob
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import pearsonr

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

# Custom imports
from werdich_cfr.models.Modeltrainer_Inc2 import VideoTrainer
from werdich_cfr.tfutils.tfutils import use_gpu_devices

#%% GPUs

physical_devices, device_list = use_gpu_devices(gpu_device_string='0,1,2,3')

#%% Directories and parameters
data_root = os.path.normpath('/mnt/obi0/andreas/data/cfr')
predictions_dir = os.path.join(data_root, 'predictions')
log_dir = os.path.join(data_root, 'log')
model_dir_list = glob.glob(os.path.join(log_dir, '*/'))
model_dir_list = [os.path.normpath(d) for d in model_dir_list]

#%% Functions

def get_checkpoint_file_list(model_dir, epoch_list):
    checkpoint_file_list = sorted(glob.glob(os.path.join(model_dir, '*_chkpt_*.h5')))
    checkpoint_file_list_xt = [file.split('.')[0] for file in checkpoint_file_list]
    checkpoint_file_cut = [file.rsplit('_', maxsplit=1)[0] for file in checkpoint_file_list_xt][0]
    checkpoint_epoch_list = [int(os.path.basename(file).rsplit('_')[-1]) \
                             for file in checkpoint_file_list_xt]
    mag = len(str(max(checkpoint_epoch_list)))

    # Select only those epochs that we want
    epoch_list = sorted(list(set(checkpoint_epoch_list).intersection(set(epoch_list))))
    print(f'Found checkpoints for epochs: {epoch_list}')
    epoch_checkpoint_file_list = [checkpoint_file_cut + '_' + str(epoch).zfill(mag) + '.h5' for epoch in epoch_list]

    return epoch_list, epoch_checkpoint_file_list

def predict_from_model(model_dir, epoch_list):

    # model_dict
    model_dict_name = os.path.basename(model_dir)+'_model_dict.pkl'
    model_dict_file = os.path.join(model_dir, model_dict_name)
    with open(model_dict_file, 'rb') as fl:
        model_dict = pickle.load(fl)

    # train_dict
    train_dict_name = model_dict_name.replace('_model_dict.pkl', '_train_dict.pkl')
    train_dict_file = os.path.join(model_dir, train_dict_name)
    with open(train_dict_file, 'rb') as fl:
        train_dict = pickle.load(fl)

    # test files
    train_file = train_dict['train_file_list'][0]
    test_basename = os.path.basename(train_file).rsplit('_', maxsplit=1)[0].replace('train', 'test')
    tfr_dir = os.path.dirname(train_file)
    test_tfr_file_list = sorted(glob.glob(os.path.join(tfr_dir, test_basename+'*.tfrecords')))
    test_parquet_file_list = [file.replace('.tfrecords', '.parquet') for file in test_tfr_file_list]
    test_df = pd.concat([pd.read_parquet(file) for file in test_parquet_file_list])
    print('Test files:')
    print(*test_tfr_file_list, sep='\n')

    # feature_dict
    feature_dict_file = glob.glob(os.path.join(tfr_dir, '*.pkl'))[0]
    with open(feature_dict_file, 'rb') as fl:
        feature_dict = pickle.load(fl)

    # model trainer class
    VT = VideoTrainer(log_dir=None, model_dict=model_dict, train_dict=train_dict, feature_dict=feature_dict)

    # checkpoint_files
    epoch_list, checkpoint_file_list = get_checkpoint_file_list(model_dir, epoch_list)

    # Continue with predictions only if there are checkpoint files
    if len(epoch_list)>0:
        # predictions
        df_pred_checkpoints = []
        df_cor_checkpoints = []
        for c, checkpoint_file in enumerate(checkpoint_file_list):

            print(f'Predictions from checkpoint {c+1}/{len(checkpoint_file_list)}')

            # run predictions in loop over checkpoint files
            pred_df = VT.predict_on_test(test_tfr_file_list, checkpoint_file, batch_size=12)

            # reshape df
            checkpoint_name = os.path.basename(checkpoint_file).split('.')[0]
            rename_dict = {model_dict['model_output']: 'label',
                           checkpoint_name: 'pred'}
            pred_df = pred_df.rename(columns=rename_dict).assign(model_name=model_dict['name'],
                                                                 model_output=model_dict['model_output'],
                                                                 epoch=epoch_list[c],
                                                                 checkpoint_file=checkpoint_file,
                                                                 dset=os.path.basename(tfr_dir))
            # calculate correlation coefficients
            spear_cor = spearmanr(pred_df.label, pred_df.pred)
            pear_cor = pearsonr(pred_df.label, pred_df.pred)

            cor_dict = {'model_name': [model_dict['name']],
                        'model_output': [model_dict['model_output']],
                        'epoch': [epoch_list[c]],
                        'chechkpoint_file': [checkpoint_file],
                        'spear_cor': [spear_cor[0]],
                        'spear_p': [spear_cor[1]],
                        'pear_cor': [pear_cor[0]],
                        'pear_p': [pear_cor[1]],
                        'n_samples': [pred_df.shape[0]],
                        'dset': [os.path.basename(tfr_dir)]}

            df_pred_checkpoints.append(pred_df)
            df_cor_checkpoints.append(pd.DataFrame(cor_dict))

        df_pred_model = pd.concat(df_pred_checkpoints).reset_index(drop=True)
        df_cor_model = pd.concat(df_cor_checkpoints).reset_index(drop=True)

    else:

        print('No checkpoint files found.')
        df_pred_model = pd.DataFrame()
        df_cor_model = pd.DataFrame()

    return test_df, df_pred_model, df_cor_model

#%% Run the predictions and save the outputs
epoch_list = [50, 100, 150]
for m, model_dir in enumerate(model_dir_list):

    model_name = os.path.basename(model_dir)
    pred_file_name = model_name + '_pred.parquet'
    cor_file_name = model_name + '_cor.parquet'
    test_df_file_name = model_name + '_testdf.parquet'

    # Skip predictions if already done
    if not os.path.exists(os.path.join(predictions_dir, pred_file_name)):
        print(f'Predictions for model {m+1}/{len(model_dir_list)}: {os.path.basename(model_dir)}')
        test_df, df_pred_model, df_cor_model = predict_from_model(model_dir, epoch_list)
        if len(df_pred_model)>0:
            df_pred_model.to_parquet(os.path.join(predictions_dir, pred_file_name))
            df_cor_model.to_parquet(os.path.join(predictions_dir, cor_file_name))
            test_df.to_parquet(os.path.join(predictions_dir, test_df_file_name))
    else:
        print(f'Test predictions already done for model {model_name}. Skipping')
