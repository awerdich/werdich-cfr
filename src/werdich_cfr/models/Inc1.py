import os
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.layers import BatchNormalization, Conv3D, MaxPooling3D, Dense, Flatten

#%% Model class

class Inc1model:

    kreg = None  # regularizers.l2(0.001) #None
    pad = 'valid'
    strd = None

    def __init__(self, model_dict):

        # NETWORK PARAMETERS AS DICTIONARY
        self.im_size = model_dict['im_size']
        self.n_frames = model_dict['n_frames']
        self.filters = model_dict['filters']
        self.fc_nodes = model_dict['fc_nodes']
        self.pool_nodes = model_dict['pool_nodes'] # Filters in the 1x1x1 convolutional pooling layer
        self.fullnet = model_dict['fullnet']

    def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5,
                         filters_pool_proj, trainable=True):
        conv_1x1 = Conv3D(filters_1x1, (1, 1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                          bias_initializer=bias_init, trainable=trainable)(x)
        conv_3x3 = Conv3D(filters_3x3_reduce, (1, 1, 1), padding='same', activation='relu',
                          kernel_initializer=kernel_init, bias_initializer=bias_init, trainable=trainable)(x)
        conv_1x1 = BatchNormalization(scale=False, trainable=trainable)(conv_1x1)
        conv_3x3 = BatchNormalization(scale=False, trainable=trainable)(conv_3x3)
        conv_3x3 = Conv3D(filters_3x3, (3, 3, 3), padding='same', activation='relu', kernel_initializer=kernel_init,
                          bias_initializer=bias_init, trainable=trainable)(conv_3x3)
        conv_3x3 = BatchNormalization(scale=False, trainable=trainable)(conv_3x3)
        conv_5x5 = Conv3D(filters_5x5_reduce, (1, 1, 1), padding='same', activation='relu',
                          kernel_initializer=kernel_init, bias_initializer=bias_init, trainable=trainable)(x)
        conv_5x5 = BatchNormalization(scale=False, trainable=trainable)(conv_5x5)
        conv_5x5 = Conv3D(filters_5x5, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',
                          kernel_initializer=kernel_init, bias_initializer=bias_init, trainable=trainable)(conv_5x5)
        conv_5x5 = BatchNormalization(scale=False, trainable=trainable)(conv_5x5)
        conv_5x5 = Conv3D(filters_5x5, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',
                          kernel_initializer=kernel_init, bias_initializer=bias_init, trainable=trainable)(conv_5x5)
        # conv_7x7 = BatchNormalization()(conv_5x5)
        # conv_7x7 = Conv3D(filters_5x5, (3,3,3),strides=(1,1,1), padding='same', activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_7x7)
        pool_proj = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        pool_proj = Conv3D(filters_pool_proj, (1, 1, 1), padding='same', activation='relu',
                           kernel_initializer=kernel_init, bias_initializer=bias_init, trainable=trainable)(pool_proj)
        output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=4)
        return output

    def video_encoder(self):

        video = layers.Input(shape = (self.n_frames, *self.im_size), name = 'video')

        # Block 1
        x = Conv3D(self.filters, (3, 3, 3), activation='relu')(video)
        x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=None)(x)

        x = Conv3D(self.pool_nodes, (1, 1, 1), activation='relu')(x)

        # Flatten for output
        x = Flatten()(x)
        x = BatchNormalization()(x)

        # Regression output
        net_cfr = Dense(self.fc_nodes, activation='relu')(x)
        net_cfr = BatchNormalization()(net_cfr)
        score_output = Dense(1, name='score_output')(net_cfr)

        # Combined classification (net_cat) and regression (net_cfr) outputs
        model = Model(inputs=video, outputs=score_output)

        return model

#%% Compile for testing

# GPUs
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

# NETWORK PARAMETERS AS DICTIONARY
# Model parameters
model_dict = {'name': 'testmodel',
              'im_size': (29, 29, 1),
              'im_scale_factor': None,
              'n_frames': 10,
              'filters': 32,
              'pool_nodes': 256,
              'fc_nodes': 256,
              'fullnet': True}

loss = {'score_output': tf.keras.losses.MeanSquaredError()}
metrics = {'score_output': tf.keras.metrics.MeanAbsolutePercentageError()}
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

model = Inc1model(model_dict = model_dict).video_encoder()
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=metrics)
