# -*- coding: utf-8 -*-
"""Model definitions.

Instantiate your desired model and fit, evaluate, and predict using that:
    import YourModel from models
    model = YourModel()
    model.create_model()
    model.compile()
    model.train()

Available models:
    1. Conv2DModel
    2. ESTCNNModel
    3. EEGNet
    4. Proposed Model: DeepEEGAbstractor
"""
# Author: Soroush Moazed <soroush.moazed@gmail.com>

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from .custom_layers import InstanceNorm, TemporalAttention, TemporalAttentionV2, TemporalAttentionV3

import tensorflow as tf

tf_version = tf.__version__
print('tensorflow version: ', tf_version)

if tf.__version__.startswith('2'):
    from tensorflow import keras
else:
    import keras
plt.style.use('ggplot')


class BaseModel:

    def __init__(self, input_shape, model_name):
        self.input_shape_ = input_shape
        self.model_name_ = model_name
        self.model_ = None
        self.loss = keras.losses.MSE
        self.optimizer = keras.optimizers.Adam()
        self.metrics = [keras.metrics.binary_accuracy, f1_score, sensitivity, specificity]
        self.history = None

    def compile(self):
        self.model_.compile(loss=self.loss,
                            optimizer=self.optimizer,
                            metrics=self.metrics)

    def train(self, train_gen, val_gen, n_iter_train, n_iter_val, epochs, plot=True):
        history = self.model_.fit_generator(train_gen, steps_per_epoch=n_iter_train, epochs=epochs,
                                            validation_data=val_gen, validation_steps=n_iter_val)
        self.history = history
        if plot:
            self._plot_performance(history.history)
        return self.model_

    def save_model(self, path='models'):
        model_dir = os.path.join(path, self.model_name_)
        graph_path = os.path.join(model_dir, self.model_name_ + '.json')
        checkpoint_path = os.path.join(model_dir, self.model_name_ + '.h5')

        model_json = self.model_.to_json()
        with open(graph_path, 'w') as json_file:
            json_file.write(model_json)
        self.model_.save_weights(checkpoint_path)
        print('Model saved to: ', model_dir)

    def create_model(self):
        return None

    def _plot_performance(self, history):
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
        ax0.plot(history['loss'], label='Training loss')
        ax0.plot(history['val_loss'], label='Validation loss')
        ax0.set_xlabel('# epoch')
        ax0.set_ylabel('Loss')
        ax0.legend()

        fig.suptitle('Loss and Accuracy for "{}" model'.format(self.model_name_), fontsize=16)

        ax1.plot(history['binary_accuracy'], label='Training accuracy')
        ax1.plot(history['val_binary_accuracy'], label='Validation accuracy')
        ax1.set_xlabel('# epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()


class Conv2DModel(BaseModel):

    """Lightweight 2D CNN for EEG classification.
     Source paper: Cloud-aided online EEG classification system for brain healthcare - A case study of depression
      evaluation with a lightweight CNN.
        [https://onlinelibrary.wiley.com/doi/10.1002/spe.2668]
    """

    def __init__(self, input_shape, model_name='conv2d'):
        super().__init__(input_shape, model_name)
        self.optimizer = keras.optimizers.SGD(lr=0.01,
                                              momentum=0.9,
                                              decay=1e-4,
                                              nesterov=True)
        if keras.backend.image_data_format() != 'channels_last':
            keras.backend.set_image_data_format('channels_last')

    def create_model(self):
        input_tensor = keras.layers.Input(shape=self.input_shape_)
        x = keras.layers.Reshape((32, 32, self.input_shape_[-1]))(input_tensor)
        x = keras.layers.SpatialDropout2D(0.1)(x)
        x = keras.layers.Conv2D(20, (3, 3))(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(18, (3, 3))(x)
        x = keras.layers.MaxPooling2D((1, 1))(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(250, activation='sigmoid')(x)
        x = keras.layers.Dense(60, activation='sigmoid')(x)
        prediction = keras.layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(input_tensor, prediction)

        self.model_ = model
        return model


class ESTCNNModel(BaseModel):

    """Spatio-temporal CNN model.

     Source paper: EEG-Based Spatio–Temporal Convolutional Neural Network for Driver Fatigue Evaluation
        [https://ieeexplore.ieee.org/document/8607897]

    """

    def __init__(self, input_shape, model_name='st_cnn'):
        super().__init__(input_shape, model_name)
        if keras.backend.image_data_format() != 'channels_last':
            keras.backend.set_image_data_format('channels_last')

    def create_model(self):
        input_tensor = keras.layers.Input(shape=self.input_shape_)
        input1 = keras.layers.Permute((2, 1))(input_tensor)
        input1 = keras.layers.Lambda(keras.backend.expand_dims,
                                     arguments={'axis': -1},
                                     name='estcnn_input')(input1)

        x = self.core_block(input1, 16)
        x = keras.layers.MaxPooling2D((1, 2), strides=2)(x)

        x = self.core_block(x, 32)
        x = keras.layers.MaxPooling2D((1, 2), strides=2)(x)

        x = self.core_block(x, 64)
        x = keras.layers.AveragePooling2D((1, 7), strides=7)(x)

        x = keras.layers.Flatten()(x)

        x = keras.layers.Dense(50, activation='relu')(x)
        output_tensor = keras.layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(input_tensor, output_tensor)

        self.model_ = model
        return model

    @staticmethod
    def core_block(x, n_units):
        out = keras.layers.Conv2D(filters=n_units,
                                  kernel_size=(1, 3),
                                  padding='valid',
                                  kernel_initializer='glorot_normal',
                                  activation='relu')(x)
        out = keras.layers.BatchNormalization()(out)
        out = keras.layers.Conv2D(filters=n_units,
                                  kernel_size=(1, 3),
                                  padding='valid',
                                  kernel_initializer='glorot_normal',
                                  activation='relu')(out)
        out = keras.layers.BatchNormalization()(out)
        out = keras.layers.Conv2D(filters=n_units,
                                  kernel_size=(1, 3),
                                  padding='valid',
                                  kernel_initializer='glorot_normal',
                                  activation='relu')(out)
        out = keras.layers.BatchNormalization()(out)
        return out


class EEGNet(BaseModel):

    """Model proposed in:
        EEGNet A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces

    Source paper: https://arxiv.org/abs/1611.08024

    Original implementation: https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py

    Note: BatchNormalization raises this error on tensorflow v1.14 and channels_first:
    ValueError: Shape must be rank 1 but is rank 0 for 'batch_normalization_2/cond/Reshape_4' (op: 'Reshape') with input shapes: [1,8,1,1], [].
    So I changed architecture to handle channels_last config because of this error, and
     I doubled the lengthes of temporal conv2d kernels and pooling sizes because of sampling rate 256 in my dataset.
    """

    def __init__(self,
                 input_shape,
                 model_name='eegnet',
                 dropout_rate=0.5,
                 kernel_length=64,
                 f1=8,
                 d=2,
                 f2=16,
                 norm_rate=0.25):
        super().__init__(input_shape, model_name)
        self.dropout_rate = dropout_rate
        self.kernel_length = kernel_length
        self.f1 = f1
        self.d = d
        self.f2 = f2
        self.norm_rate = norm_rate
        if keras.backend.image_data_format() != 'channels_last':
            keras.backend.set_image_data_format('channels_last')

    def create_model(self):
        samples, channels = self.input_shape_
        input_tensor = keras.layers.Input(shape=self.input_shape_)
        input1 = keras.layers.Permute((2, 1))(input_tensor)
        input1 = keras.layers.Lambda(keras.backend.expand_dims,
                                     arguments={'axis': -1},
                                     name='eegnet_standard_input')(input1)

        block1 = keras.layers.Conv2D(self.f1,
                                     (1, 2 * self.kernel_length),
                                     padding='same',
                                     use_bias=False)(input1)
        block1 = keras.layers.BatchNormalization(axis=-1)(block1)
        block1 = keras.layers.DepthwiseConv2D((channels, 1),
                                              use_bias=False,
                                              depth_multiplier=self.d,
                                              depthwise_constraint=keras.constraints.max_norm(1.))(block1)
        block1 = keras.layers.BatchNormalization(axis=-1)(block1)
        block1 = keras.layers.Activation('relu')(block1)
        block1 = keras.layers.AveragePooling2D((1, 2 * 4))(block1)
        block1 = keras.layers.Dropout(self.dropout_rate)(block1)

        block2 = keras.layers.SeparableConv2D(self.f2,
                                              (1, 2 * 16),
                                              use_bias=False,
                                              padding='same')(block1)
        block2 = keras.layers.BatchNormalization(axis=-1)(block2)
        block2 = keras.layers.Activation('relu')(block2)
        block2 = keras.layers.AveragePooling2D((1, 2 * 8))(block2)
        block2 = keras.layers.Dropout(self.dropout_rate)(block2)

        flatten = keras.layers.Flatten(name='flatten')(block2)

        dense = keras.layers.Dense(1,
                                   name='dense',
                                   kernel_constraint=keras.constraints.max_norm(self.norm_rate))(flatten)
        output = keras.layers.Activation('sigmoid',
                                         name='output')(dense)

        model = keras.Model(inputs=input_tensor,
                            outputs=output)
        self.model_ = model

        return model


class SpatioTemporalWFB(BaseModel):

    """Spatio-Temporal Windowed Filter Bank CNN.

    The design is based on STFT, i.e. each layer consists of equal length filters that extract features in different
     frequencies.

    Receptive field of each unit before GAP layer is 306 time-steps, about 1.25 seconds with sampling rate of 256, i.e.
     each unit looks at 1.25 seconds of input multi-variate time-series.
    """

    def __init__(self,
                 input_shape,
                 model_name='ST-WFB-CNN'):
        super().__init__(input_shape, model_name)
        self.n_kernels = [8, 6, 4]
        self.strides = [2, 2, 1]
        self.pool_size = 2
        self.pool_stride = 2
        self.spatial_dropout_rate = 0.2
        self.dropout_rate = 0.4
        self.use_bias = False
        self.kernel_size = 16
        if keras.backend.image_data_format() != 'channels_last':
            keras.backend.set_image_data_format('channels_last')

    def create_model(self):
        input_tensor = keras.layers.Input(shape=self.input_shape_,
                                          name='input_tensor')

        # Block 1
        x = self._eeg_filter_bank(input_tensor=input_tensor,
                                  n_units=self.n_kernels[0],
                                  strides=self.strides[0])
        x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                          strides=self.pool_stride)(x)

        # Block 2
        x = keras.layers.SpatialDropout1D(self.spatial_dropout_rate)(x)
        x = self._eeg_filter_bank(input_tensor=x,
                                  n_units=self.n_kernels[1],
                                  strides=self.strides[1])
        x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                          strides=self.pool_stride)(x)

        # Block 3
        x = keras.layers.Dropout(self.dropout_rate)(x)
        x = self._eeg_filter_bank(input_tensor=x,
                                  n_units=self.n_kernels[2],
                                  strides=self.strides[2])
        x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                          strides=self.pool_stride)(x)

        # Block 3
        x = keras.layers.Dropout(self.dropout_rate)(x)
        x = self._eeg_filter_bank(input_tensor=x,
                                  n_units=self.n_kernels[3],
                                  strides=self.strides[3])

        # Temporal abstraction
        x = keras.layers.GlobalAveragePooling1D()(x)

        # Logistic regression unit
        output_tensor = keras.layers.Dense(1, activation='sigmoid', name='output')(x)

        model = keras.Model(input_tensor, output_tensor)
        self.model_ = model
        return model

    def _eeg_filter_bank(self, input_tensor, n_units, strides):
        branch_a = self._conv1d(input_tensor=input_tensor,
                                filters=n_units,
                                kernel_size=self.kernel_size,
                                dilation_rate=1,
                                strides=strides)

        branch_b = self._conv1d(input_tensor=input_tensor,
                                filters=n_units,
                                kernel_size=self.kernel_size // 2,
                                dilation_rate=2,
                                strides=strides)

        branch_c = self._conv1d(input_tensor=input_tensor,
                                filters=n_units,
                                kernel_size=self.kernel_size // 4,
                                dilation_rate=4,
                                strides=strides)

        output = keras.layers.concatenate([branch_a, branch_b, branch_c], axis=-1)
        return output

    def _conv1d(self, input_tensor, filters, kernel_size, dilation_rate, strides):
        out = keras.layers.Conv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding='same',
                                  data_format='channels_last',
                                  dilation_rate=dilation_rate,
                                  activation=None,
                                  use_bias=self.use_bias)(input_tensor)
        out = InstanceNorm(mean=0,
                           stddev=1)(out)
        out = keras.layers.ELU()(out)
        return out


class TemporalWFB(BaseModel):

    """Temporal Windowed Filter Bank CNN.

        The design is based on STFT, i.e. first layer extracts multiple temporal features in multiple scales from input
         in fixed-sized windows, which are optimized by using the error signal of all channels. Then for each channel a
         specific combination of features is created, and two layers of spatio-temporal 1D convolutional layers are used
         for generating final representations by combination of channels' representations. This final representations
         will be passed to a Global Average Pooling layer which abstracts the temporal dimension before making
         prediction through a sigmoid unit.

        Receptive field of each unit before GAP layer is 226 time-steps, about 1 second with sampling rate of 256, i.e.
         each unit looks at 1 second of input multi-variate time-series.
    """

    def __init__(self,
                 input_shape,
                 model_name='T-WFB-CNN'):
        super().__init__(input_shape, model_name)
        self.wfb_kernel_length = 32
        self.wfb_kernel_units = 8
        self.wfb_kernel_strides = 2
        self.sdropout_rate = 0.2
        self.pool_size = 2
        self.pool_strides = 2
        self.channel_wise_layer_kernel_length = 4
        self.channel_wise_layer_n_kernel = 1
        self.channel_wise_layer_strides = 1
        self.dropout_rate = 0.4
        self.st_1_kernel_length = 8
        self.st_1_n_kernel = 16
        self.st_1_strides = 1
        self.st_2_kernel_length = 8
        self.st_2_n_kernel = 10
        self.st_2_strides = 1
        self.use_bias = False
        if keras.backend.image_data_format() != 'channels_last':
            keras.backend.set_image_data_format('channels_last')

    def create_model(self):
        input_tensor = keras.layers.Input(shape=self.input_shape_)
        permuted_input = keras.layers.Permute((2, 1))(input_tensor)
        permuted_input = keras.layers.Lambda(keras.backend.expand_dims,
                                             arguments={'axis': -1},
                                             name='permuted_input')(permuted_input)

        # Block 1: Temporal dilated filter-bank for initial feature extraction
        block_1 = self._temporal_dilated_filter_bank(input_tensor=permuted_input,
                                                     n_units=self.wfb_kernel_units,
                                                     strides=self.wfb_kernel_strides)
        block_1 = keras.layers.SpatialDropout2D(self.sdropout_rate)(block_1)
        block_1 = keras.layers.Permute((3, 1))(block_1)  # out[:, :, -1] is representation of a unique input channel
        block_1 = keras.layers.AveragePooling2D(pool_size=(1, self.pool_size),
                                                strides=(1, self.pool_strides))(block_1)

        # Block 2: Make single signal out of each channel's new representations
        block_2 = self._channel_wise_mixing(input_tensor=block_1,
                                            kernel_length=self.channel_wise_layer_kernel_length,
                                            strides=self.channel_wise_layer_strides,
                                            n_kernel=self.channel_wise_layer_n_kernel)
        block_2 = keras.layers.Lambda(keras.backend.squeeze,
                                      arguments={'axis': 1},
                                      name='squeezed_block_2')(block_2)
        block_2 = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                                strides=self.pool_strides)(block_2)

        # Block 3: Spatio-temporal mixing of channels
        block_3 = keras.layers.SpatialDropout1D(self.sdropout_rate)(block_2)
        block_3 = self._st_conv1d(input_tensor=block_3,
                                  n_units=self.st_1_n_kernel,
                                  kernel_length=self.st_1_kernel_length,
                                  strides=self.st_1_strides)
        block_3 = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                                strides=self.pool_strides)(block_3)

        # Block 4
        block_4 = keras.layers.Dropout(self.dropout_rate)(block_3)
        block_4 = self._st_conv1d(input_tensor=block_4,
                                  n_units=self.st_2_n_kernel,
                                  kernel_length=self.st_2_kernel_length,
                                  strides=self.st_2_strides)

        # Temporal abstraction
        gap = keras.layers.GlobalAveragePooling1D()(block_4)

        # Prediction
        output_tensor = keras.layers.Dense(units=1, activation='sigmoid', name='output')(gap)

        model = keras.Model(input_tensor, output_tensor)
        self.model_ = model

        return model

    def _temporal_dilated_filter_bank(self, input_tensor, n_units, strides):
        branch_a = self._temporal_conv1d(input_tensor=input_tensor,
                                         n_units=n_units,
                                         kernel_length=self.wfb_kernel_length,
                                         strides=strides,
                                         dilation_rate=1)
        branch_b = self._temporal_conv1d(input_tensor=input_tensor,
                                         n_units=n_units,
                                         kernel_length=self.wfb_kernel_length // 2,
                                         strides=strides,
                                         dilation_rate=2)
        branch_c = self._temporal_conv1d(input_tensor=input_tensor,
                                         n_units=n_units,
                                         kernel_length=self.wfb_kernel_length // 4,
                                         strides=strides,
                                         dilation_rate=4)
        branch_d = self._temporal_conv1d(input_tensor=input_tensor,
                                         n_units=n_units,
                                         kernel_length=self.wfb_kernel_length // 8,
                                         strides=strides,
                                         dilation_rate=8)
        output = keras.layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)
        return output

    def _temporal_conv1d(self, input_tensor, n_units, kernel_length, strides, dilation_rate):
        x = keras.layers.Conv2D(filters=n_units,
                                kernel_size=(1, kernel_length),
                                strides=(1, strides),
                                padding='same',
                                data_format='channels_last',
                                dilation_rate=(1, dilation_rate),
                                activation=None,
                                use_bias=self.use_bias)(input_tensor)

        # Normalize outputs: normalize each input channel
        x = InstanceNorm(axis=1, mean=0, stddev=1.0)(x)

        out = keras.layers.Activation('elu')(x)
        return out

    def _st_conv1d(self, input_tensor, n_units, kernel_length, strides):
        x = keras.layers.Conv1D(filters=n_units,
                                kernel_size=kernel_length,
                                strides=strides,
                                padding='valid',
                                data_format='channels_last',
                                dilation_rate=1,
                                activation=None)(input_tensor)
        x = InstanceNorm(axis=-1, mean=0, stddev=1.0)(x)
        out = keras.layers.Activation('elu')(x)
        return out

    def _channel_wise_mixing(self, input_tensor, kernel_length, strides, n_kernel):
        n_features = keras.backend.int_shape(input_tensor)[-3]
        x = keras.layers.DepthwiseConv2D(kernel_size=(n_features, kernel_length),
                                         strides=(1, strides),
                                         padding='valid',
                                         depth_multiplier=n_kernel,
                                         data_format='channels_last',
                                         activation=None,
                                         use_bias=self.use_bias)(input_tensor)
        x = InstanceNorm(axis=-1, mean=0, stddev=1.0)(x)
        out = keras.layers.Activation('elu')(x)
        return out


class TemporalDFB(BaseModel):

    """Temporal Dilated Filter Bank CNN.

            The design is based on DWT, i.e. first layer extracts multiple temporal features in multiple
             scales from input in multiple context sizes, which are optimized by using the error signal
             of all channels. Then for each channel a specific combination of features is created, and
             two layers of spatio-temporal 1D convolutional layers are used for generating final representations
             by combination of channels' representations. This final representations will be passed to a
             Global Average Pooling layer which abstracts the temporal dimension before making
             prediction through a sigmoid unit.

            Receptive field of each unit before GAP layer is 322 time-steps, about 1.25 second with sampling rate of 256, i.e.
             each unit looks at 1 second of input multi-variate time-series.
        """

    def __init__(self,
                 input_shape,
                 model_name='T-DFB-CNN'):
        super().__init__(input_shape, model_name)
        self.dfb_kernel_length = 16
        self.dfb_kernel_units = 8
        self.dfb_kernel_strides = 2
        self.sdropout_rate = 0.2
        self.pool_size = 2
        self.pool_strides = 2
        self.channel_wise_layer_kernel_length = 4
        self.channel_wise_layer_n_kernel = 1
        self.channel_wise_layer_strides = 1
        self.dropout_rate = 0.4
        self.st_1_kernel_length = 8
        self.st_1_n_kernel = 16
        self.st_1_strides = 1
        self.st_2_kernel_length = 8
        self.st_2_n_kernel = 10
        self.st_2_strides = 1
        self.use_bias = False
        if keras.backend.image_data_format() != 'channels_last':
            keras.backend.set_image_data_format('channels_last')

    def create_model(self):
        input_tensor = keras.layers.Input(shape=self.input_shape_)
        permuted_input = keras.layers.Permute((2, 1))(input_tensor)
        permuted_input = keras.layers.Lambda(keras.backend.expand_dims,
                                             arguments={'axis': -1},
                                             name='permuted_input')(permuted_input)

        # Block 1: Temporal dilated filter-bank for initial feature extraction
        block_1 = self._st_dilated_filter_bank(input_tensor=permuted_input,
                                               n_units=self.dfb_kernel_units,
                                               strides=self.dfb_kernel_strides)
        block_1 = keras.layers.SpatialDropout2D(self.sdropout_rate)(block_1)
        block_1 = keras.layers.Permute((3, 1))(block_1)  # out[:, :, -1] is representation of a unique input channel
        block_1 = keras.layers.AveragePooling2D(pool_size=(1, self.pool_size),
                                                strides=(1, self.pool_strides))(block_1)

        # Block 2: Make single signal out of each channel's new representations
        block_2 = self._channel_wise_mixing(input_tensor=block_1,
                                            kernel_length=self.channel_wise_layer_kernel_length,
                                            strides=self.channel_wise_layer_strides,
                                            n_kernel=self.channel_wise_layer_n_kernel)
        block_2 = keras.layers.Lambda(keras.backend.squeeze,
                                      arguments={'axis': 1},
                                      name='squeezed_block_2')(block_2)
        block_2 = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                                strides=self.pool_strides)(block_2)

        # Block 3: Spatio-temporal mixing of channels
        block_3 = keras.layers.SpatialDropout1D(self.sdropout_rate)(block_2)
        block_3 = self._st_conv1d(input_tensor=block_3,
                                  n_units=self.st_1_n_kernel,
                                  kernel_length=self.st_1_kernel_length,
                                  strides=self.st_1_strides)
        block_3 = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                                strides=self.pool_strides)(block_3)

        # Block 4
        block_4 = keras.layers.Dropout(self.dropout_rate)(block_3)
        block_4 = self._st_conv1d(input_tensor=block_4,
                                  n_units=self.st_2_n_kernel,
                                  kernel_length=self.st_2_kernel_length,
                                  strides=self.st_2_strides)

        # Temporal abstraction
        gap = keras.layers.GlobalAveragePooling1D()(block_4)

        # Prediction
        output_tensor = keras.layers.Dense(units=1, activation='sigmoid', name='output')(gap)

        model = keras.Model(input_tensor, output_tensor)
        self.model_ = model

        return model

    def _st_dilated_filter_bank(self, input_tensor, n_units, strides):
        branch_a = self._temporal_conv1d(input_tensor=input_tensor,
                                         n_units=n_units,
                                         kernel_length=self.dfb_kernel_length,
                                         strides=strides,
                                         dilation_rate=1)
        branch_b = self._temporal_conv1d(input_tensor=input_tensor,
                                         n_units=n_units,
                                         kernel_length=self.dfb_kernel_length,
                                         strides=strides,
                                         dilation_rate=2)
        branch_c = self._temporal_conv1d(input_tensor=input_tensor,
                                         n_units=n_units,
                                         kernel_length=self.dfb_kernel_length,
                                         strides=strides,
                                         dilation_rate=4)
        branch_d = self._temporal_conv1d(input_tensor=input_tensor,
                                         n_units=n_units,
                                         kernel_length=self.dfb_kernel_length,
                                         strides=strides,
                                         dilation_rate=8)
        output = keras.layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)
        return output

    def _temporal_conv1d(self, input_tensor, n_units, kernel_length, strides, dilation_rate):
        x = keras.layers.Conv2D(filters=n_units,
                                kernel_size=(1, kernel_length),
                                strides=(1, strides),
                                padding='same',
                                data_format='channels_last',
                                dilation_rate=(1, dilation_rate),
                                activation=None,
                                use_bias=self.use_bias)(input_tensor)

        # Normalize outputs: normalize each input channel
        x = InstanceNorm(axis=1, mean=0, stddev=1.0)(x)

        out = keras.layers.Activation('elu')(x)
        return out

    def _st_conv1d(self, input_tensor, n_units, kernel_length, strides):
        x = keras.layers.Conv1D(filters=n_units,
                                kernel_size=kernel_length,
                                strides=strides,
                                padding='valid',
                                data_format='channels_last',
                                dilation_rate=1,
                                activation=None)(input_tensor)
        x = InstanceNorm(axis=-1, mean=0, stddev=1.0)(x)
        out = keras.layers.Activation('elu')(x)
        return out

    def _channel_wise_mixing(self, input_tensor, kernel_length, strides, n_kernel):
        n_features = keras.backend.int_shape(input_tensor)[-3]
        x = keras.layers.DepthwiseConv2D(kernel_size=(n_features, kernel_length),
                                         strides=(1, strides),
                                         padding='valid',
                                         depth_multiplier=n_kernel,
                                         data_format='channels_last',
                                         activation=None,
                                         use_bias=self.use_bias)(input_tensor)
        x = InstanceNorm(axis=-1, mean=0, stddev=1.0)(x)
        out = keras.layers.Activation('elu')(x)
        return out


class SpatioTemporalDFB(BaseModel):

    """Spatio-Temporal Windowed Filter Bank CNN.

        The design is based on DWT, i.e. each layer consists of dilated filters that extract features in different
         frequencies and different contexts.

        Receptive field of each unit before GAP layer is 481 time-steps, about 2 seconds with sampling rate of 256, i.e.
         each unit looks at 2 seconds of input multi-variate time-series.
    """

    def __init__(self,
                 input_shape,
                 model_name='ST-DFB-CNN'):
        super().__init__(input_shape, model_name)
        self.n_kernels = [8, 8, 6, 6, 4]
        self.pool_size = 2
        self.pool_stride = 2
        self.spatial_dropout_rate = 0.2
        self.dropout_rate = 0.4
        self.use_bias = False
        self.kernel_size = 4
        if keras.backend.image_data_format() != 'channels_last':
            keras.backend.set_image_data_format('channels_last')

    def create_model(self):
        input_tensor = keras.layers.Input(shape=self.input_shape_,
                                          name='input_tensor')

        # Block 1
        x = self._eeg_filter_bank(input_tensor=input_tensor,
                                  n_units=self.n_kernels[0],
                                  strides=1)
        x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                          strides=self.pool_stride)(x)

        # Block 2
        x = keras.layers.SpatialDropout1D(self.spatial_dropout_rate)(x)
        x = self._eeg_filter_bank(input_tensor=x,
                                  n_units=self.n_kernels[1],
                                  strides=1)
        x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                          strides=self.pool_stride)(x)

        # Block 3
        x = keras.layers.Dropout(self.dropout_rate)(x)
        x = self._eeg_filter_bank(input_tensor=x,
                                  n_units=self.n_kernels[2],
                                  strides=1)
        x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                          strides=self.pool_stride)(x)

        # Block 4
        x = keras.layers.Dropout(self.dropout_rate)(x)
        x = self._eeg_filter_bank(input_tensor=x,
                                  n_units=self.n_kernels[3],
                                  strides=1)
        x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                          strides=self.pool_stride)(x)

        # Block 5
        x = keras.layers.Dropout(self.dropout_rate)(x)
        x = self._eeg_filter_bank(input_tensor=x,
                                  n_units=self.n_kernels[4],
                                  strides=1)

        # Temporal abstraction
        x = keras.layers.GlobalAveragePooling1D()(x)

        # Logistic regression unit
        output_tensor = keras.layers.Dense(1, activation='sigmoid', name='output')(x)

        model = keras.Model(input_tensor, output_tensor)
        self.model_ = model
        return model

    def _eeg_filter_bank(self, input_tensor, n_units, strides):
        branch_a = self._conv1d(input_tensor=input_tensor,
                                filters=n_units,
                                kernel_size=self.kernel_size,
                                dilation_rate=1,
                                strides=strides)

        branch_b = self._conv1d(input_tensor=input_tensor,
                                filters=n_units,
                                kernel_size=self.kernel_size,
                                dilation_rate=2,
                                strides=strides)

        branch_c = self._conv1d(input_tensor=input_tensor,
                                filters=n_units,
                                kernel_size=self.kernel_size,
                                dilation_rate=4,
                                strides=strides)

        output = keras.layers.concatenate([branch_a, branch_b, branch_c], axis=-1)
        return output

    def _conv1d(self, input_tensor, filters, kernel_size, dilation_rate, strides):
        out = keras.layers.Conv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding='same',
                                  data_format='channels_last',
                                  dilation_rate=dilation_rate,
                                  activation=None,
                                  use_bias=self.use_bias)(input_tensor)
        out = InstanceNorm(mean=0,
                           stddev=1)(out)
        out = keras.layers.ELU()(out)
        return out


class TemporalInceptionResnet(BaseModel):

    # TODO: add pruning

    def __init__(self,
                 input_shape,
                 model_name='deega-gap',
                 lightweight=False,
                 units=(10, 8, 6),
                 dropout_rate=0.1,
                 pool_size=2,
                 use_bias=True,
                 dilation_rate=4,
                 kernel_size=8,
                 activation='elu'):
        super().__init__(input_shape, model_name)
        self.lightweight = lightweight
        self.units = units
        self.dropout_rate = dropout_rate
        self.pool_size = pool_size
        self.use_bias = use_bias
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.activation = activation
        if keras.backend.image_data_format() != 'channels_last':
            keras.backend.set_image_data_format('channels_last')

    def create_model(self):
        input_tensor = keras.layers.Input(shape=self.input_shape_,
                                          name='input_tensor')

        # Block 1
        x = keras.layers.SpatialDropout1D(self.dropout_rate)(input_tensor)
        y = self._dilated_inception(x,
                                    n_units=self.units[0])
        if not self.lightweight:
            y = self._dilated_inception(y,
                                        n_units=self.units[0])
        x = self._add_skip_connection(x, y)
        x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                          strides=self.pool_size)(x)

        # Block 2 - n
        for n_units in self.units[1:]:
            x = keras.layers.SpatialDropout1D(self.dropout_rate)(x)
            y = self._dilated_inception(x,
                                        n_units=n_units)
            if not self.lightweight:
                y = self._dilated_inception(y,
                                            n_units=n_units)
            x = self._add_skip_connection(x, y)
            x = keras.layers.AveragePooling1D(pool_size=self.pool_size)(x)

        x = keras.layers.GlobalAveragePooling1D()(x)

        # Logistic regression unit
        output_tensor = keras.layers.Dense(1, activation='sigmoid', name='output')(x)

        model = keras.Model(input_tensor, output_tensor)
        self.model_ = model

        return model

    def _dilated_inception(self, input_tensor, n_units):
        branch_a = self._causal_conv1d(input_tensor, n_units, self.kernel_size, 1)

        branch_b = self._causal_conv1d(input_tensor, n_units, self.kernel_size, 1)
        branch_b = self._causal_conv1d(branch_b, n_units, self.kernel_size, self.dilation_rate)

        branch_c = self._causal_conv1d(input_tensor, n_units, 1, 1)

        output = keras.layers.concatenate([branch_a, branch_b, branch_c], axis=-1)
        return output

    def _add_skip_connection(self, input_tensor, output_tensor, scale=1.0):
        channels = keras.backend.int_shape(output_tensor)[-1]

        shortcut_branch = self._causal_conv1d(input_tensor, channels, 1)
        out = self._weighted_add(shortcut_branch, output_tensor, scale)
        return keras.layers.ELU()(out)

    @staticmethod
    def _weighted_add(shortcut_branch, inception_branch, scale_factor):
        return keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                                   arguments={'scale': scale_factor})([shortcut_branch, inception_branch])

    def _causal_conv1d(self, x, filters, kernel_size, dilation_rate=1):
        out = keras.layers.Conv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=1,
                                  padding='same',
                                  data_format='channels_last',
                                  dilation_rate=dilation_rate,
                                  activation=None,
                                  use_bias=self.use_bias)(x)
        if self.activation == 'elu':
            out = InstanceNorm(mean=0, stddev=1.0)(out)
            out = keras.layers.ELU()(out)
        elif self.activation == 'relu':
            out = InstanceNorm(mean=0.5, stddev=0.5)(out)
            out = keras.layers.ReLU()(out)
        return out


class DeepEEGAbstractor(BaseModel):

    # TODO: add pruning

    def __init__(self,
                 input_shape,
                 model_name='deep_eeg_abstractor',
                 lightweight=False,
                 units=(10, 8, 6),
                 dropout_rate=0.1,
                 pool_size=2,
                 use_bias=True,
                 dilation_rate=4,
                 kernel_size=8,
                 activation='elu',
                 attention_type='v1'):
        super().__init__(input_shape, model_name)
        self.lightweight = lightweight
        self.units = units
        self.dropout_rate = dropout_rate
        self.pool_size = pool_size
        self.use_bias = use_bias
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.activation = activation
        self.attention_type = attention_type
        if keras.backend.image_data_format() != 'channels_last':
            keras.backend.set_image_data_format('channels_last')

    def create_model(self):
        input_tensor = keras.layers.Input(shape=self.input_shape_, name='input_tensor')

        # Block 1
        x = keras.layers.SpatialDropout1D(self.dropout_rate)(input_tensor)
        y = self._dilated_inception(x,
                                    n_units=self.units[0])
        if not self.lightweight:
            y = self._dilated_inception(y,
                                        n_units=self.units[0])
        x = self._add_skip_connection(x, y)
        x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                          strides=self.pool_size)(x)

        # Block 2 - n
        for n_units in self.units[1:]:
            x = keras.layers.SpatialDropout1D(self.dropout_rate)(x)
            y = self._dilated_inception(x,
                                        n_units=n_units)
            if not self.lightweight:
                y = self._dilated_inception(y,
                                            n_units=n_units)
            x = self._add_skip_connection(x, y)
            x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                              strides=self.pool_size)(x)

        # Attention layer
        if self.attention_type == 'v1':
            x = TemporalAttention(name='embedding')(x)
        elif self.attention_type == 'v2':
            x = TemporalAttentionV2(name='embedding')(x)
        else:
            x = TemporalAttentionV3(name='embedding')(x)

        # Logistic regression unit
        output_tensor = keras.layers.Dense(1, activation='sigmoid', name='output')(x)

        model = keras.Model(input_tensor, output_tensor)
        self.model_ = model

        return model

    def _dilated_inception(self, input_tensor, n_units):
        branch_a = self._causal_conv1d(input_tensor, n_units, self.kernel_size, 1)

        branch_b = self._causal_conv1d(input_tensor, n_units, self.kernel_size, 1)
        branch_b = self._causal_conv1d(branch_b, n_units, self.kernel_size, self.dilation_rate)

        branch_c = self._causal_conv1d(input_tensor, n_units, 1, 1)

        output = keras.layers.concatenate([branch_a, branch_b, branch_c], axis=-1)
        return output

    def _add_skip_connection(self, input_tensor, output_tensor, scale=1.0):
        channels = keras.backend.int_shape(output_tensor)[-1]

        shortcut_branch = self._causal_conv1d(input_tensor, channels, 1)
        out = self._weighted_add(shortcut_branch, output_tensor, scale)
        return keras.layers.ELU()(out)

    @staticmethod
    def _weighted_add(shortcut_branch, inception_branch, scale_factor):
        return keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                                   arguments={'scale': scale_factor})([shortcut_branch, inception_branch])

    def _causal_conv1d(self, x, filters, kernel_size, dilation_rate=1):
        out = keras.layers.Conv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=1,
                                  padding='same',
                                  data_format='channels_last',
                                  dilation_rate=dilation_rate,
                                  activation=None,
                                  use_bias=self.use_bias)(x)
        if self.activation == 'elu':
            out = InstanceNorm(mean=0, stddev=1.0)(out)
            out = keras.layers.ELU()(out)
        elif self.activation == 'relu':
            out = InstanceNorm(mean=0.5, stddev=0.5)(out)
            out = keras.layers.ReLU()(out)
        return out

    def generate_embeddings(self, data):
        model = keras.Model(self.model_.input, self.model_.get_layer('embedding').output)
        embeddings = model.predict(data)
        return embeddings

    def plot_embeddings(self, data, labels):
        embeddings = self.generate_embeddings(data)
        transformed = TSNE(n_components=2).fit_transform(embeddings)
        group0 = np.where(labels == 0)
        group1 = np.where(labels == 1)
        plt.plot(transformed[group0][:, 0], transformed[group0][:, 1], legend='Class 0')
        plt.plot(transformed[group1][:, 0], transformed[group1][:, 1], legend='Class 1')
        plt.show()


class DeepEEGAbstractorV2(BaseModel):

    # TODO: add pruning

    def __init__(self,
                 input_shape,
                 model_name='deep_eeg_abstractor_v2',
                 n_initial_features=32,
                 units=(10, 10),
                 dropout_rate=0.1,
                 pool_size=2,
                 use_bias=True,
                 dilation_rate=5,
                 kernel_size=5):
        super().__init__(input_shape, model_name)
        self.n_initial_features = n_initial_features
        self.units = units
        self.dropout_rate = dropout_rate
        self.pool_size = pool_size
        self.use_bias = use_bias
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        if keras.backend.image_data_format() != 'channels_last':
            keras.backend.set_image_data_format('channels_last')

    def create_model(self):
        input_tensor = keras.layers.Input(shape=self.input_shape_)

        # Starter feature extractor
        x = keras.layers.SpatialDropout1D(self.dropout_rate)(input_tensor)
        x = keras.layers.Permute((2, 1))(x)  # (?, ch, t)
        x = keras.layers.Lambda(keras.backend.expand_dims,
                                arguments={'axis': -1},
                                name='inital_feature_extractor')(x)  # (?, ch, t, 1)
        x = keras.layers.Conv2D(32,
                                (1, 5),
                                padding='same',
                                use_bias=self.use_bias,
                                data_format='channels_last',
                                dilation_rate=1,
                                activation=None)(x)
        x = InstanceNorm(axis=None, mean=0.5, stddev=0.5)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.SpatialDropout2D(self.dropout_rate)(x)
        x = keras.layers.Permute((3, 2, 1))(x)  # (?, 32, t, ch)
        x = keras.layers.DepthwiseConv2D(kernel_size=(32, 1),
                                         strides=1,
                                         padding='valid',
                                         depth_multiplier=1,
                                         use_bias=self.use_bias,
                                         activation=None)(x)
        x = InstanceNorm(axis=None, mean=0.5, stddev=0.5)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Lambda(keras.backend.squeeze,
                                arguments={'axis': 1},
                                name='inital_feature_extractor_out')(x)  # (?, t, ch)

        # Main blocks
        for n_units in self.units:
            x = keras.layers.SpatialDropout1D(self.dropout_rate)(x)
            y = self._dilated_inception(x, n_units=n_units)
            x = self._add_skip_connection(x, y)
            x = keras.layers.MaxPooling1D(pool_size=self.pool_size,
                                          strides=self.pool_size)(x)

        x = TemporalAttention(name='embedding')(x)
        output_tensor = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(input_tensor, output_tensor)
        self.model_ = model

        return model

    def _dilated_inception(self, input_tensor, n_units):
        branch_a = self._causal_conv1d(input_tensor, n_units, self.kernel_size, 1)

        branch_b = self._causal_conv1d(input_tensor, n_units, self.kernel_size, 1)
        branch_b = self._causal_conv1d(branch_b, n_units, self.kernel_size, self.dilation_rate)

        branch_c = self._causal_conv1d(input_tensor, n_units, 1, 1)

        # branch_d = keras.layers.AveragePooling1D(self.kernel_size, strides=1, padding='same')(input_tensor)
        # branch_d = self._causal_conv1d(branch_d, n_units, 1, 1)

        output = keras.layers.concatenate([branch_a, branch_b, branch_c], axis=-1)
        return output

    def _add_skip_connection(self, input_tensor, output_tensor, scale=0.5):
        channels = keras.backend.int_shape(output_tensor)[-1]

        shortcut_branch = self._causal_conv1d(input_tensor, channels, 1)
        out = self._weighted_add(shortcut_branch, output_tensor, scale)
        return keras.layers.ReLU()(out)

    @staticmethod
    def _weighted_add(shortcut_branch, inception_branch, scale_factor):
        return keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                                   arguments={'scale': scale_factor})([shortcut_branch, inception_branch])

    def _causal_conv1d(self, x, filters, kernel_size, dilation_rate=1):
        out = keras.layers.Conv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=1,
                                  padding='causal',
                                  data_format='channels_last',
                                  dilation_rate=dilation_rate,
                                  activation=None,
                                  use_bias=self.use_bias)(x)
        out = InstanceNorm(mean=0.5, stddev=0.5)(out)
        out = keras.layers.ReLU()(out)
        return out


def f1_score(y_true, y_pred):
    true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true, 0, 1)))
    predicted_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + keras.backend.epsilon())
    recall = true_positives / (possible_positives + keras.backend.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + keras.backend.epsilon())
    return f1_val


def sensitivity(y_true, y_pred):
    # recall: true_p / possible_p
    true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + keras.backend.epsilon())


def specificity(y_true, y_pred):
    # true_n / possible_n
    true_negatives = keras.backend.sum(keras.backend.round(keras.backend.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = keras.backend.sum(keras.backend.round(keras.backend.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + keras.backend.epsilon())