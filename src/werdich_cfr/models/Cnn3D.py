# Imports
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import BatchNormalization, Conv3D, MaxPooling3D, Dense, Flatten

class Convmodel:

    kreg = None  # regularizers.l2(0.001) #None
    pad = 'valid'
    strd = None

    def __init__(self, model_dict):

        # NETWORK PARAMETERS AS DICTIONARY
        self.im_size = model_dict['im_size']
        self.n_frames = model_dict['n_frames']
        self.cl_outputs = model_dict['cl_outputs']
        self.filters = model_dict['filters']
        self.fc_nodes = model_dict['fc_nodes']
        self.pool_nodes = model_dict['pool_nodes'] # Filters in the 1x1x1 convolutional pooling layer
        self.fullnet = model_dict['fullnet']

    def video_encoder(self):

        video = layers.Input(shape = (self.n_frames, *self.im_size), name = 'video')

        # Block 1
        x = Conv3D(self.filters, (3, 3, 3), activation='relu')(video)
        x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=None)(x)

        if self.fullnet:

            # Block 2
            x = Conv3D(self.filters, (3, 3, 3), activation='relu')(x)
            x = BatchNormalization()(x)

            # Block 3
            x = Conv3D(self.filters, (3, 3, 3), activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling3D(pool_size=(1, 2, 2), strides=None)(x)

            # Block 4
            x   = Conv3D(self.filters* 2, (3, 3, 3), activation='relu')(x)
            x = BatchNormalization()(x)

            # Block 5
            x = Conv3D(self.filters * 2, (3, 3, 3), activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling3D(pool_size=(1, 2, 2), strides=None)(x)

            # Block 6
            x = Conv3D(self.filters * 4, (3, 3, 3), activation='relu')(x)
            x = BatchNormalization()(x)

            # Block 7
            x = Conv3D(self.filters * 4, (3, 3, 3), activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling3D(pool_size=(2, 1, 2), strides=None)(x)

            # Block 8
            x = Conv3D(self.filters * 8, (3, 3, 3), activation='relu')(x)
            x = BatchNormalization()(x)

            # Block 9
            x = Conv3D(self.filters * 8, (3, 3, 3), activation='relu')(x)
            x = BatchNormalization()(x)

            # Block 10
            x = Conv3D(self.filters * 8, (3, 3, 3), activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling3D(pool_size=(2, 1, 2))(x)

        # Reduce model complexity by 1x1x1 convolution
        # replaces x = Dense(self.fc_nodes, activation='relu')(x)
        x = Conv3D(self.pool_nodes, (1, 1, 1), activation = 'relu')(x)
        # Flatten for output
        x = Flatten()(x)
        x = BatchNormalization()(x)

        # Categorical outputs classification
        net_cat = Dense(self.fc_nodes, activation='relu')(x)
        net_cat = BatchNormalization()(net_cat)
        class_output = Dense(self.cl_outputs, activation = 'softmax', name = 'class_output')(net_cat)

        # Regression output
        net_cfr = Dense(self.fc_nodes, activation='relu')(x)
        net_cfr = BatchNormalization()(net_cfr)
        score_output = Dense(1, name = 'score_output')(net_cfr)

        # Combined classification (net_cat) and regression (net_cfr) outputs
        model = Model(inputs = video, outputs = [class_output, score_output])

        return model
