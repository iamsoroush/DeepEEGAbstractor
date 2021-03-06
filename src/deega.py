# -*- coding: utf-8 -*-
"""Proposed model as a single class.
"""
# Author: Soroush Moazed <soroush.moazed@gmail.com>

import os

from .helpers import f1_score, sensitivity, specificity
from .custom_layers import InstanceNorm, TemporalAttention

from . import keras


class KerasModel:

    def __init__(self, model_name, models_dir):
        self.model_name = model_name

        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)

        self.model_dir = os.path.join(models_dir, model_name)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.logs_dir = os.path.join(self.model_dir, 'logs')
        if not os.path.exists(self.logs_dir):
            os.mkdir(self.logs_dir)

        self.checkpoints = list()
        self.logs = list()
        self._update_logs_checkpoints_paths()

        self.model_ = None

    def _update_logs_checkpoints_paths(self):
        self.checkpoints = [os.path.join(self.model_dir, f) for f in os.listdir(self.model_dir)]
        self.logs = [os.path.join(self.logs_dir, f) for f in os.listdir(self.logs_dir)]

    def load_model(self, checkpoint_name=None):
        if checkpoint_name is None:
            chkpt_path = self.checkpoints[-1]
        else:
            chkpt_path = os.path.join(self.model_dir, checkpoint_name)
            assert chkpt_path in self.checkpoints, "Model {} can not found in {}.".format(checkpoint_name,
                                                                                          self.model_dir)
        self._create_model()
        self.model_.load_weights(chkpt_path)
        print('Model loaded successfully.')

    def _create_model(self):
        pass


class DeepEEGAbstractor(KerasModel):
    """Wrapper for EEGNet deep CNN.

    Use like this:
        * deega = DeepEEGAbstractor() # Instantiate an DeepEEGAbstractor object
        * deega.load_model('201908...') # Load model from a pre-trained checkpoint, or:
        * deega.create_model() # : create model from scratch
        * deega.compile_and_train() # Compile and train the model on your data
    """

    # TODO: add pruning

    def __init__(self,
                 time_steps=None,
                 channels=20,
                 model_name='deep_eeg_abstractor',
                 lightweight=False,
                 units=(6, 8, 6),
                 dropout_rate=0.1,
                 pool_size=2,
                 models_dir='./models',
                 use_bias=True,
                 dilation_rate=5,
                 kernel_size=5):
        super(DeepEEGAbstractor, self).__init__(model_name, models_dir)
        self.time_steps = time_steps
        self.channels = channels
        self.lightweight = lightweight
        self.units = units
        self.dropout_rate = dropout_rate
        self.pool_size = pool_size
        self.use_bias = use_bias
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.model_ = None
        if keras.backend.image_data_format() != 'channels_last':
            keras.backend.set_image_data_format('channels_last')

    def create_model(self):
        self._create_model()
        print('Model created.')

    def get_model(self):
        return self.model_

    def compile_and_train(self,
                          train_gen,
                          test_gen,
                          n_iter_train,
                          n_iter_test,
                          task,
                          epochs=50,
                          x_embed=None,
                          y_embed=None,
                          metadata=None,
                          run_prefix='deega'):
        assert self.model_ is not None, "Create or load model first."

        # dt = datetime.now().strftime('%Y%m%d-%H%M%S')
        prefix = '{}-{}-{}epochs'.format(run_prefix, task, epochs)

        checkpoint_path = os.path.join(self.model_dir,
                                       'weights-' + prefix + '-{val_loss:.2f}.hdf5')
        self.checkpoints.append(checkpoint_path)

        log_dir = os.path.join(self.logs_dir, prefix)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        if x_embed is None:
            emb_metadata, embedding_freq = None, None
        else:
            emb_metadata = 'metadata.tsv'
            embedding_freq = 1
            if metadata is not None:
                with open(os.path.join(log_dir, emb_metadata), 'w') as f:
                    f.write(metadata[0])
                    for row in metadata[1:]:
                        f.write(row)
            else:
                if task == 'rnr':
                    labels = ['Responder' if i == 1 else 'NonResponder' for i in y_embed]
                    index = list(range(len(y_embed)))
                elif task == 'hmdd':
                    labels = ['MDD' if i == 1 else 'Healthy' for i in y_embed]
                    index = list(range(len(y_embed)))
                else:
                    raise Exception('task must be rnr or hmdd')
                with open(os.path.join(log_dir, emb_metadata), 'w') as f:
                    f.write("Index\tLabel\n")
                    for ind, label in zip(index, labels):
                        f.write("{}\t{}\n".format(ind, label))
        self.logs.append(log_dir)

        checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                              monitor='val_loss',
                                                              verbose=0,
                                                              save_best_only=True,
                                                              save_weights_only=True,
                                                              mode='auto',
                                                              period=1)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                           histogram_freq=0,
                                                           write_graph=False,
                                                           write_grads=False,
                                                           write_images=True,
                                                           embeddings_freq=embedding_freq,
                                                           embeddings_layer_names=['embedding'],
                                                           embeddings_metadata=emb_metadata,
                                                           embeddings_data=x_embed,
                                                           update_freq='epoch')
        reducelr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                              factor=0.5,
                                                              patience=3,
                                                              verbose=0,
                                                              mode='auto',
                                                              min_delta=0.0001,
                                                              cooldown=0,
                                                              min_lr=0.00001)
        callbacks = [checkpoint_callback, tensorboard_callback, reducelr_callback]

        self._compile()
        self.model_.fit_generator(train_gen,
                                  steps_per_epoch=n_iter_train,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  validation_data=test_gen,
                                  validation_steps=n_iter_test)
        return log_dir

    def _compile(self):
        loss = keras.losses.MSE
        optimizer = keras.optimizers.Adam(lr=0.001)
        metrics = [keras.metrics.binary_accuracy,
                   f1_score,
                   sensitivity,
                   specificity]
        self.model_.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def _create_model(self):
        input_tensor = keras.layers.Input(shape=(self.time_steps, self.channels))

        # Block 1
        x = keras.layers.SpatialDropout1D(self.dropout_rate)(input_tensor)
        y = self._dilated_inception(x,
                                    n_units=self.units[0])
        if not self.lightweight:
            y = self._dilated_inception(y,
                                        n_units=self.units[0])
        x = self._add_skip_connection(x, y)
        x = keras.layers.MaxPooling1D(pool_size=self.pool_size,
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
            x = keras.layers.MaxPooling1D(pool_size=self.pool_size,
                                          strides=self.pool_size)(x)

        # Attention layer
        x = TemporalAttention(name='embedding')(x)

        # Logistic regression unit
        output_tensor = keras.layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(input_tensor, output_tensor)
        self.model_ = model

    def _dilated_inception(self, input_tensor, n_units):
        branch_a = self._causal_conv1d(input_tensor, n_units, self.kernel_size, 1)

        branch_b = self._causal_conv1d(input_tensor, n_units, self.kernel_size, 1)
        branch_b = self._causal_conv1d(branch_b, n_units, self.kernel_size, self.dilation_rate)

        branch_c = self._causal_conv1d(input_tensor, n_units, 1, 1)

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


class CausalConv1D(keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 dilation_rate,
                 use_bias,
                 norm_mean=0.0,
                 norm_std=1.0,
                 **kwargs):
        super(CausalConv1D, self).__init__(**kwargs)
        self.causal_conv1d = keras.layers.Conv1D(filters=filters,
                                                 kernel_size=kernel_size,
                                                 strides=1,
                                                 padding='same',
                                                 data_format='channels_last',
                                                 dilation_rate=dilation_rate,
                                                 activation=None,
                                                 use_bias=use_bias)
        self.instance_normalization = InstanceNorm(mean=norm_mean, stddev=norm_std)
        self.activation = keras.layers.ELU()

    def call(self, inputs):
        x = self.causal_conv1d(inputs)
        x = self.instance_normalization(x)
        out = self.activation(x)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape


class EEGInceptionBlock(keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 dilation_rate,
                 use_bias,
                 **kwargs):
        super(EEGInceptionBlock, self).__init__(**kwargs)
        self.causal_conv1d_dilation1 = CausalConv1D(filters=filters,
                                                    kernel_size=kernel_size,
                                                    dilation_rate=1,
                                                    use_bias=use_bias)
        self.causal_conv1d_dilation5 = CausalConv1D(filters=filters,
                                                    kernel_size=kernel_size,
                                                    dilation_rate=dilation_rate,
                                                    use_bias=use_bias)
        self.pointwise_conv1d = CausalConv1D(filters=filters,
                                             kernel_size=1,
                                             dilation_rate=1,
                                             use_bias=use_bias)
        self.concatenate = keras.layers.Concatenate(axis=-1)
        self.n_output_channels = 3 * filters

    def call(self, inputs):
        branch_a = self.causal_conv1d_dilation1(inputs)

        branch_b = self.causal_conv1d_dilation1(inputs)
        branch_b = self.causal_conv1d_dilation5(branch_b)

        branch_c = self.pointwise_conv1d(inputs)

        out = self.concatenate([branch_a, branch_b, branch_c])
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n_output_channels)


class ResEEGInceptionBlock(keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 dilation_rate,
                 use_bias,
                 scale_factor=0.5,
                 **kwargs):
        super(ResEEGInceptionBlock, self).__init__(**kwargs)
        self.inception_1 = EEGInceptionBlock(filters=filters,
                                             kernel_size=kernel_size,
                                             dilation_rate=dilation_rate,
                                             use_bias=use_bias)
        self.inception_2 = EEGInceptionBlock(filters=filters,
                                             kernel_size=kernel_size,
                                             dilation_rate=dilation_rate,
                                             use_bias=use_bias)
        n_channels = self.inception_2.n_output_channels
        self.point_conv = CausalConv1D(filters=n_channels,
                                       kernel_size=1,
                                       dilation_rate=1,
                                       use_bias=use_bias)
        self.add_skip = keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                                            arguments={'scale': scale_factor})

    def call(self, inputs):
        x = self.inception_1(inputs)
        x = self.inception_2(x)
        shortcut_branch = self.point_conv(inputs)
        x = self.add_skip([shortcut_branch, x])
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class CwCDInception1D(keras.Model):

    def __init__(self,
                 filters,
                 kernel_size,
                 dilation_rate,
                 use_bias,
                 **kwargs):
        super(CwCDInception1D, self).__init__(**kwargs)
        self.causal_conv1d = CausalConv1D(filters,
                                          kernel_size,
                                          dilation_rate,
                                          use_bias)
        self.inception1d = EEGInceptionBlock(filters,
                                             kernel_size,
                                             dilation_rate,
                                             use_bias)
        #         self.splits = [keras.layers.Lambda(lambda h: h[:, :, i]) for i in range(20)]
        self.expand_dims = keras.layers.Lambda(keras.backend.expand_dims, arguments={'axis': 2})
        self.concat = keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        input_shape = keras.backend.int_shape(inputs)
        outputs = list()
        channels = input_shape[-1]
        for i in range(channels):
            x = keras.layers.Lambda(lambda h: h[:, :, i])(inputs)
            x = self.expand_dims(x)
            x = self.inception1d(x)
            outputs.append(x)

        out = self.concat(outputs)
        return out


def get_deega(n_channels=19,
              time_steps=None,
              model_name='deep_eeg_abstractor',
              units=(6, 8, 10),
              dropout_rate=0.1,
              pool_size=4,
              use_bias=True,
              dilation_rate=4,
              kernel_size=8,
              weighted_add_scale=0.5):

    """Use this only with tf 2.0"""
    input_tensor = keras.Input(shape=(time_steps, n_channels), name='input_tensor')

    x = keras.layers.SpatialDropout1D(dropout_rate,
                                      name='block1-spatial_dropout')(input_tensor)
    x = ResEEGInceptionBlock(filters=units[0],
                             kernel_size=kernel_size,
                             dilation_rate=dilation_rate,
                             use_bias=use_bias,
                             scale_factor=weighted_add_scale,
                             name='block1-res_eeg_inception1d_block')(x)
    if pool_size is not None:
        x = keras.layers.AveragePooling1D(pool_size=pool_size,
                                          strides=pool_size,
                                          name='block1-pooling1d')(x)

    # Block 2 - n
    for i, n in enumerate(units[1:]):
        block_name = 'block' + str(i + 2)
        x = keras.layers.SpatialDropout1D(dropout_rate / 2,
                                          name=block_name + '-spatial_dropout')(x)
        x = ResEEGInceptionBlock(filters=n,
                                 kernel_size=kernel_size,
                                 dilation_rate=dilation_rate,
                                 use_bias=use_bias,
                                 scale_factor=weighted_add_scale,
                                 name=block_name + '-res_eeg_inception1d_block')(x)
        if pool_size is not None:
            x = keras.layers.AveragePooling1D(pool_size=pool_size,
                                              strides=pool_size,
                                              name=block_name + '-pooling1d')(x)

    # Attention layer
    x = TemporalAttention(name='embedding')(x)

    # Logistic regression unit
    output_tensor = keras.layers.Dense(1, activation='sigmoid', name='prediction')(x)
    model = keras.models.Model(input_tensor, output_tensor, name=model_name)
    return model
