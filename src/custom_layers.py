# -*- coding: utf-8 -*-
"""Keras custom layers."""
# Author: Soroush Moazed <soroush.moazed@gmail.com>


import tensorflow as tf
if tf.__version__.startswith('2'):
    from tensorflow import keras
else:
    import keras


class TemporalAttention(keras.layers.Layer):
    """Attention layer for conv1d networks.

    Source paper: arXiv:1512.08756v5

    Summarizes temporal axis and outputs a vector with the same length as channels.
    Note that the unweighted attention will be simple GlobalAveragePooling1D

    Make sure to pass the inputs to this layer in "channels_last" format.

    use like this at top of a conv1d layer:
        x = TemporalAttention()(x)
    """

    def __init__(self,
                 w_regularizer=None, b_regularizer=None,
                 w_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = keras.initializers.get('glorot_uniform')

        self.w_regularizer = keras.regularizers.get(w_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        self.w_constraint = keras.constraints.get(w_constraint)
        self.b_constraint = keras.constraints.get(b_constraint)

        self.bias = bias

        self.w = None
        self.b = None
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.w = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.w_regularizer,
                                 constraint=self.w_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(1,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        super(TemporalAttention, self).build(input_shape)

    def compute_mask(self, input_tensor, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # x: T*D , W: D*1 ==> a: T*1
        a = self._dot_product(x, self.w)

        if self.bias:
            a += self.b

        a = keras.backend.tanh(a)

        alpha = keras.backend.exp(a)
        denominator = keras.backend.sum(alpha, axis=-1, keepdims=True) + keras.backend.epsilon()
        alpha /= denominator

        # x: T*D, alpha: T*1
        weighted_input = x * keras.backend.expand_dims(alpha)
        return keras.backend.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    @staticmethod
    def _dot_product(x, kernel):
        """Wrapper for dot product operation,

        Args:
            x: input
            kernel: weights
        """
        return keras.backend.squeeze(keras.backend.dot(x, keras.backend.expand_dims(kernel)), axis=-1)


class InstanceNorm(keras.layers.Layer):
    """Instance normalization layer.

    Normalizes the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    of each feature map for each instance in batch close to 0 and the standard
    deviation close to 1.

    Args:
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv1D` layer with
            `data_format="channels_last"`,
            set `axis=-1` in `InstanceNorm`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        Input shape: Arbitrary. Use the keyword argument `input_shape`
            (tuple of integers, does not include the samples axis)
            when using this layer as the first layer in a Sequential model.
        Output shape: Same shape as input.

    References:
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """

    def __init__(self,
                 axis=-1,
                 epsilon=1e-3,
                 mean=0,
                 stddev=1,
                 **kwargs):
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.mean = mean
        self.stddev = stddev
        super(InstanceNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = keras.layers.InputSpec(ndim=ndim)
        super(InstanceNorm, self).build(input_shape)

    def call(self, inputs, training=None):
        input_shape = keras.backend.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = keras.backend.mean(inputs, reduction_axes, keepdims=True)
        stddev = keras.backend.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean + self.mean) / stddev * self.stddev
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
        }
        base_config = super(InstanceNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Attention(keras.layers.Layer):
    def __init__(self,
                 step_dim=128,
                 w_regularizer=None,
                 b_regularizer=None,
                 w_constraint=None,
                 b_constraint=None,
                 bias=True,
                 **kwargs):
        self.supports_masking = True
        self.init = keras.initializers.get('glorot_uniform')

        self.w_regularizer = keras.regularizers.get(w_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        self.w_constraint = keras.constraints.get(w_constraint)
        self.b_constraint = keras.constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        self.w = None
        self.b = None
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.w = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.w_regularizer,
                                 constraint=self.w_constraint)
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.built = True

    def compute_mask(self, input_tensor, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = keras.backend.reshape(keras.backend.dot(keras.backend.reshape(x, (-1, features_dim)),
                                                      keras.backend.reshape(self.w, (features_dim, 1))),
                                    (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = keras.backend.tanh(eij)
        a = keras.backend.exp(eij)
        if mask is not None:
            a *= keras.backend.cast(mask, keras.backend.floatx())
        a /= keras.backend.cast(keras.backend.sum(a, axis=1, keepdims=True) + keras.backend.epsilon(),
                                keras.backend.floatx())
        a = keras.backend.expand_dims(a)
        weighted_input = x * a
        return keras.backend.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

    def get_config(self):
        config = {
            'step_dim': self.step_dim,
            'w_regularizer': self.w_regularizer,
            'b_regularizer': self.b_regularizer,
            'w_constraint': self.w_constraint,
            'b_constraint': self.b_constraint,
            'bias': self.bias
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttentionWithContext(keras.layers.Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 w_regularizer=None, u_regularizer=None, b_regularizer=None,
                 w_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = keras.initializers.get('glorot_uniform')

        self.w_regularizer = keras.regularizers.get(w_regularizer)
        self.u_regularizer = keras.regularizers.get(u_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        self.w_constraint = keras.constraints.get(w_constraint)
        self.u_constraint = keras.constraints.get(u_constraint)
        self.b_constraint = keras.constraints.get(b_constraint)

        self.bias = bias

        self.w = None
        self.b = None
        self.u = None
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.w = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.w_regularizer,
                                 constraint=self.w_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input_tensor, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = self._dot_product(x, self.w)

        if self.bias:
            uit += self.b

        uit = keras.backend.tanh(uit)
        ait = self._dot_product(uit, self.u)

        a = keras.backend.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= keras.backend.cast(mask, keras.backend.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= keras.backend.cast(keras.backend.sum(a, axis=1, keepdims=True), keras.backend.floatx())
        a /= keras.backend.cast(keras.backend.sum(a, axis=1, keepdims=True) + keras.backend.epsilon(),
                                keras.backend.floatx())

        a = keras.backend.expand_dims(a)
        weighted_input = x * a
        return keras.backend.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    @staticmethod
    def _dot_product(x, kernel):
        """Wrapper for dot product operation, in order to be compatible with both
           Theano and Tensorflow

        Args:
            x: input
            kernel: weights
        """
        if keras.backend.backend() == 'tensorflow':
            return keras.backend.squeeze(keras.backend.dot(x, keras.backend.expand_dims(kernel)), axis=-1)
        else:
            return keras.backend.dot(x, kernel)


class InstanceNormalization(keras.layers.Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv1D` layer with
            `data_format="channels_last"`,
            set `axis=-1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """

    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.gamma = None
        self.beta = None
        super(InstanceNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = keras.layers.InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = keras.backend.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = keras.backend.mean(inputs, reduction_axes, keepdims=True)
        stddev = keras.backend.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = keras.backend.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = keras.backend.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
