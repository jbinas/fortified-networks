"""
Whitebox adversarial training code for the publication

 Fortified Networks: Improving the Robustness of Deep Networks
 by Modeling the Manifold of Hidden Representations.

 Alex Lamb, Jonathan Binas, Anirudh Goyal,
 Dmitriy Serdyuk, Sandeep Subramanian, Ioannis Mitliagkas, Yoshua Bengio

 https://arxiv.org/pdf/1804.02485
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from cleverhans.model import Model


#  |
#  |  CHANGE MODEL ARCHITECTURE DOWNSTAIRS
#  |  (at the end of this file)
#  V


class MLP(Model):
    """
    An example of a bare bones multilayer perceptron (MLP) class.
    """
    def __init__(self, layers, input_shape):
        super(MLP, self).__init__()

        self.layer_names = []
        self.layers = layers
        self.input_shape = input_shape
        if isinstance(layers[-1], Softmax):
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
                layer.name = name
            self.layer_names.append(name)

            layer.set_input_shape(input_shape)
            input_shape = layer.get_output_shape()

    def fprop(self, x, set_ref=False):
        states = []
        for layer in self.layers:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x)
            assert x is not None
            states.append(x)
        states = dict(zip(self.get_layer_names(), states))
        return states


class ECMLP(MLP):
    ''' error-correcting MLP '''
    def fprop(self, x, set_ref=False):
        states = []
        for layer in self.layers:
            if set_ref:
                layer.ref = x
            if isinstance(layer, Autoencoder):
                layer.reset_states()
                x_pre = 1. * x
                for i in range(layer.iterations):
                    layer.set_pre_state(x_pre)
                    x = layer.fprop(x, add_noise=(i == 0))
                    layer.set_post_state(x)
            else:
                x = layer.fprop(x)
            assert x is not None
            states.append(x)
        states = dict(zip(self.get_layer_names(), states))
        return states

    def get_ae_states(self):
        ''' return states affected by AE '''
        out_pre, out_post = [], []
        for layer in self.layers:
            if isinstance(layer, Autoencoder):
                out_pre.extend(layer.pre_state)
                out_post.extend(layer.post_state)
        return out_pre, out_post


class ECMLP_block(ECMLP):
    ''' with blocking gradients '''
    def fprop(self, x, set_ref=False):
        states = []
        for layer in self.layers:
            if set_ref:
                layer.ref = x
            if isinstance(layer, Autoencoder):
                y_pre = tf.stop_gradient(x)
                y = 1. * y_pre
                layer.reset_states()
                for i in range(layer.iterations):
                    layer.set_pre_state(y_pre)
                    y = layer.fprop(y, add_noise=(i == 0))
                    layer.set_post_state(y)
                x = layer.fprop(x)
            else:
                x = layer.fprop(x)
            assert x is not None
            states.append(x)
        states = dict(zip(self.get_layer_names(), states))
        return states


class Layer(object):
    ''' layer base class '''
    def get_output_shape(self):
        return self.output_shape


class Autoencoder(Layer):
    ''' implements a one-layer AE '''
    def __init__(self, n_hidden, activation=None, shared_weights=True, noise=0.1, iterations=1):
        self.dim_z = n_hidden
        self.activation = activation if activation is not None else tf.nn.tanh
        self.shared_weights = shared_weights
        self.noise = noise
        self.iterations = iterations
        self.reset_states()

    def set_input_shape(self, input_shape):
        batch_size, dim_x = input_shape
        self.input_shape = input_shape
        self.output_shape = input_shape

        self.w = tf.Variable(tf.random_uniform((dim_x, self.dim_z),
                        -1.0 / np.sqrt(dim_x + self.dim_z),
                        1.0 / np.sqrt(dim_x + self.dim_z)))
        self.b_en = tf.Variable(tf.zeros(self.dim_z))
        self.b_de = tf.Variable(tf.zeros(dim_x))
        self.params = {'w': self.w, 'b1': self.b_en, 'b2': self.b_de}

        if not self.shared_weights:
            self.w_de = tf.Variable(tf.random_uniform((self.dim_z, dim_x),
                        -1.0 / np.sqrt(dim_x + self.dim_z),
                        1.0 / np.sqrt(dim_x + self.dim_z)))
            self.params['w_de'] = self.w_de

    def fprop(self, x, add_noise=True):
        if add_noise:
            x = x + tf.cast(tf.random_normal(shape=tf.shape(x), stddev=self.noise), tf.float32)
        z_ = self.activation(tf.matmul(x, self.w) + self.b_en)
        if self.shared_weights:
            x_ = self.activation(tf.matmul(z_, self.w, transpose_b=True) + self.b_de)
        else:
            x_ = self.activation(tf.matmul(z_, self.w_de) + self.b_de)
        return x_

    def reset_states(self):
        self.pre_state = []
        self.post_state = []

    def set_pre_state(self, x):
        self.pre_state.append(x)

    def set_post_state(self, x):
        self.post_state.append(x)


class DeepAutoencoder(Autoencoder):
    ''' implements a deep AE '''
    def __init__(self, n_hidden, n_hidden_2, activation=None,
            noise=0.1, bottleneck_noise=0.1):
        self.dim_z = n_hidden
        self.dim_z_2 = n_hidden_2
        self.activation = activation if activation is not None else tf.nn.tanh
        self.noise = noise
        self.bottleneck_noise = bottleneck_noise

    def set_input_shape(self, input_shape):
        batch_size, dim_x = input_shape
        self.input_shape = input_shape
        self.output_shape = input_shape

        self.w1 = tf.Variable(tf.random_uniform(
            (dim_x, self.dim_z),
            -1.0 / np.sqrt(dim_x + self.dim_z),
            1.0 / np.sqrt(dim_x + self.dim_z)))

        self.w2 = tf.Variable(tf.random_uniform(
            (self.dim_z, self.dim_z_2),
            -1.0 / np.sqrt(self.dim_z + self.dim_z_2),
            1.0 / np.sqrt(self.dim_z + self.dim_z_2)))

        self.b1_en = tf.Variable(tf.zeros(self.dim_z))
        self.b2_en = tf.Variable(tf.zeros(self.dim_z_2))
        self.b1_de = tf.Variable(tf.zeros(self.dim_z))
        self.b2_de = tf.Variable(tf.zeros(dim_x))
        self.params = {
            'w1': self.w1, 'w2': self.w2,
            'b1_en': self.b1_en, 'b1_de': self.b1_de,
            'b2_en': self.b2_en, 'b2_de': self.b2_de}

    def fprop(self, x, add_noise=True):
        if add_noise:
            x = x + tf.cast(tf.random_normal(shape=tf.shape(x), stddev=self.noise), tf.float32)
        z1_ = self.activation(tf.matmul(x, self.w1) + self.b1_en)
        z2_ = self.activation(tf.matmul(z1_, self.w2) + self.b2_en)
        z2_ = z2_ + tf.cast(tf.random_normal(shape=tf.shape(z2_), stddev=self.bottleneck_noise), tf.float32)
        d1_ = self.activation(tf.matmul(z2_, self.w2, transpose_b=True) + self.b1_de)
        x_ = self.activation(tf.matmul(d1_, self.w1, transpose_b=True) + self.b2_de)
        return x_


class Linear(Layer):
    def __init__(self, num_hid, w_name=None):
        self.num_hid = num_hid
        if w_name is not None:
            self.w_name = w_name

    def set_input_shape(self, input_shape):
        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]
        init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                   keep_dims=True))
        self.W = tf.Variable(init)
        self.b = tf.Variable(np.zeros((self.num_hid,)).astype('float32'))

    def fprop(self, x):
        return tf.matmul(x, self.W) + self.b


class Conv2D(Layer):
    def __init__(self, output_channels, kernel_shape, strides, padding, w_name=None):
        self.__dict__.update(locals())
        del self.self
        if w_name is not None:
            self.w_name = w_name

    def set_input_shape(self, input_shape):
        batch_size, rows, cols, input_channels = input_shape
        kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                   self.output_channels)
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
        init = tf.random_normal(kernel_shape, dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                   axis=(0, 1, 2)))
        self.kernels = tf.Variable(init)
        self.b = tf.Variable(
            np.zeros((self.output_channels,)).astype('float32'))
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = 1
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.conv2d(x, self.kernels, (1,) + tuple(self.strides) + (1,),
                            self.padding) + self.b


class ConvAutoencoder(Autoencoder):
    def __init__(self, output_channels, kernel_shape, strides, padding, **kwargs):
        self.__dict__.update(locals())
        del self.self
        self.__dict__.update(kwargs)

    def set_input_shape(self, input_shape):
        batch_size, rows, cols, input_channels = input_shape
        kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                   self.output_channels)
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
        init = tf.random_normal(kernel_shape, dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                   axis=(0, 1, 2)))
        self.kernels = tf.Variable(init)
        self.b = tf.Variable(
            np.zeros((self.output_channels,)).astype('float32'))
        self.b2 = tf.Variable(
            np.zeros((input_channels,)).astype('float32'))
        self.output_shape = input_shape

    def fprop(self, x):
        input_shape = [d.value for d in x.shape]
        if input_shape[0] is None:
            input_shape[0] = 128 #XXX hack: hard-coded batch-size
        strides_ = (1,) + tuple(self.strides) + (1,)
        if add_noise:
            x = x + tf.cast(tf.random_normal(shape=tf.shape(x), stddev=self.noise), tf.float32)
        x_ = tf.nn.conv2d(x, self.kernels, strides_, self.padding) + self.b
        x_ = tf.nn.conv2d_transpose(x_, self.kernels, input_shape,
                strides_, self.padding) + self.b2
        return x_


class MaxPooling(Layer):
    def __init__(self, kernel_shape, strides, padding):
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.padding = padding

    def set_input_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = 1
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.max_pool(x,
                (1,) + self.kernel_shape + (1,),
                (1,) + self.strides + (1,),
                self.padding)


class LayerNorm(Layer):
    def __init__(self):
        super(LayerNorm, self).__init__()
        self.fprop = self.fprop_noscope #XXX not sure this should be done

    def set_input_shape(self, input_shape):
        self.input_shape = list(input_shape)
        params_shape = [input_shape[-1]]
        self.params_shape = params_shape

        self.beta = tf.get_variable(
            'beta', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32))
        self.gamma = tf.get_variable(
            'gamma', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32))

    def fprop_noscope(self, x):
        mean = tf.reduce_mean(x, (1, 2), keep_dims=True)
        x = x - mean
        std = tf.sqrt(1e-7 +
                      tf.reduce_mean(tf.square(x), (1, 2), keep_dims=True))
        x = x / std
        return x * self.gamma + self.beta


class ReLU(Layer):
    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x):
        return tf.nn.relu(x)


class LeakyReLU(ReLU):
    def __init__(self):
        super(LeakyReLU, self).__init__()

    def fprop(self, x):
        return tf.nn.leaky_relu(x)

class Tanh(Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x):
        return tf.nn.tanh(x)


class Softmax(Layer):
    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.softmax(x)


class Flatten(Layer):
    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        output_width = 1
        for factor in shape[1:]:
            output_width *= factor
        self.output_width = output_width
        self.output_shape = [None, output_width]

    def fprop(self, x):
        return tf.reshape(x, [-1, self.output_width])


def make_basic_model(nb_filters=64, nb_classes=10, input_shape=(None, 28, 28, 1),
                     model_arch='fcn', model_class=None, blocking_option=False):

    if model_class is None:
        if blocking_option:
            model_class = ECMLP_block
        else:
            model_class = ECMLP

    if model_arch == 'resnet':
        return make_resnet_model(nb_filters, nb_classes, input_shape)

    # ---
    # add new model definitions here
    # they will be slectable through --arch and --arch_sub command line args
    # ---
    model_layers = {
        'fcn': [ # default FC model used in experiments
            Flatten(),
            #Autoencoder(512, activation=tf.nn.leaky_relu,noise=0.1),
            Linear(512),
            LeakyReLU(),
            #Autoencoder(64, activation=tf.nn.leaky_relu,noise=0.1),
            Linear(512),
            LeakyReLU(),
            #Autoencoder(64, activation=tf.nn.leaky_relu,noise=0.1),
            #Autoencoder(512, activation=tf.nn.leaky_relu, noise=1e-7),
            Linear(512),
            LeakyReLU(),
            #Autoencoder(1024, activation=tf.nn.leaky_relu, noise=0.001),
            Linear(512),
            LeakyReLU(),
            Autoencoder(256, activation=tf.nn.leaky_relu, noise=0.1, iterations=1),
            Linear(nb_classes),
            Softmax()
            ],
        'fcn_sub': [ # substitute FC model
            Flatten(),
            Linear(512),
            ReLU(),
            Linear(512),
            ReLU(),
            Linear(nb_classes),
            Softmax()
            ],
        'fcn_sub_orig': [ # substitute FC model
            Flatten(),
            Linear(200),
            ReLU(),
            Linear(200),
            ReLU(),
            Linear(nb_classes),
            Softmax()
            ],
        'cnn': [# CNN model used in experiments
            Conv2D(nb_filters, (8, 8), (2, 2), "SAME"),
            LeakyReLU(),
            Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID"),
            LeakyReLU(),
            Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID"),
            LeakyReLU(),
            #ConvAutoencoder(16, (5, 5), (1, 1), "SAME", activation=tf.nn.leaky_relu), #XXX testing...
            Flatten(),
            Autoencoder(256, activation=tf.nn.leaky_relu, noise=0.1, iterations=1),
            Linear(nb_classes),
            Softmax()
            ],
        'cnn_orig': [ # CNN model used in experiments
            Conv2D(nb_filters, (8, 8), (2, 2), "SAME"),
            ReLU(),
            Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID"),
            ReLU(),
            Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID"),
            ReLU(),
            Flatten(),
            Autoencoder(256, activation=tf.tanh),
            Linear(nb_classes),
            Softmax()
            ],
        'deep_cnn': [
            Conv2D(nb_filters, (5, 5), (1, 1), "SAME"),
            ReLU(),
            MaxPooling((3, 3), (2, 2), "SAME"),
            Conv2D(nb_filters, (5, 5), (1, 1), "SAME"),
            ReLU(),
            MaxPooling((3, 3), (2, 2), "SAME"),
            Conv2D(nb_filters * 2, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Conv2D(nb_filters * 2, (3, 3), (1, 1), "SAME"),
            LeakyReLU(),
            Conv2D(nb_filters * 2, (3, 3), (1, 1), "SAME"),
            ReLU(),
            MaxPooling((3, 3), (2, 2), "SAME"),
            Flatten(),
            Autoencoder(128, activation=tf.nn.leaky_relu),
            Linear(384),
            ReLU(),
            Linear(192),
            ReLU(),
            Linear(nb_classes),
            Softmax()
        ],
        'cnn_sub': [ # substitute CNN model
            Conv2D(nb_filters, (8, 8), (2, 2), "SAME"),
            ReLU(),
            Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID"),
            ReLU(),
            Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID"),
            ReLU(),
            Flatten(),
            Linear(nb_classes),
            Softmax()
            ],
        'cnn_sub_small': [ # substitute CNN model
            Conv2D(nb_filters, (5, 5), (2, 2), "SAME"),
            ReLU(),
            Conv2D(nb_filters * 2, (5, 5), (2, 2), "VALID"),
            ReLU(),
            Flatten(),
            Linear(128),
            ReLU(),
            Linear(nb_classes),
            Softmax()
            ],
        }

    return model_class(model_layers[model_arch], input_shape)

def make_resnet_model(nb_filters=64, nb_classes=10, input_shape=(None, 28, 28, 1)):
    from . import resnet_tf
    return resnet_tf.ResNetTF()
