"""
###########################################################
# The purpose of this script is to model the LSTM network #
###########################################################
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn


def normalized_columns_initializer(std=1.0):
    """
    Higher order function that returns a normalized initializer

    Parameters
    ----------
    std : float
        standard initializer
    """
    def _initializer(shape, dtype=None, partition_info=None):
        """
        Private initializer function to be normalized.

        Parameters
        ----------
        env_id : str
            environment id to be registered in Gym
        client_id : str
            Client ID
        remotes : str
            BLANK
        kwargs : dict
            BLANK
        """
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def linear(x, size, name, initializer=None, bias_init=0):
    """
    Utility linear function of the form f = wx + b;
    where w = weights and b = biases; these may be matrices.

    Parameters
    ----------
    x : int
        set of attributes to be used in linear function calculation
    size : int
        size of data to be handled
    name : str
        variable name
    initializer : function
        initializer function
    bias_init : int
        bias initialization value
    """
    w = tf.get_variable(name + "/w",
                        [x.get_shape()[1],
                         size],
                        initializer=initializer)
    b = tf.get_variable(name + "/b", [size],
                        initializer=tf.constant_initializer(
                            bias_init))
    return tf.matmul(x, w) + b


def conv2d(x,
           num_filters,
           name,
           filter_size=(3, 3),
           stride=(1, 1),
           pad="SAME",
           dtype=tf.float32,
           collections=None):
    """
    2-D convolution function.

    Parameters
    ----------
    x : int
        set of attributes to be used in the convolution
    num_filters : int
        number of filters to be used in convolution
    name : str
        variable name
    filter_size : 2-tuple
        size of filter to be used in convolution
    stride : 2-tuple
        convolution stride
    pad : str
        padding used in convolution
    dtype : datatype
        tensorflow datatype
    collections : object
        BLANK
    """
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1],
                        int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype,
                            tf.random_uniform_initializer(
                                -w_bound,
                                w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters],
                            initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def build_lstm(x, size, name, step_size):
    """
    Function used to build the LSTM Feudal network.

    Parameters
    ----------
    x : int
        set of attributes to be used in the network
    size : int
        size of LSTM network
    step_size : str
        step size of LSTM network
    """
    lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)

    c_init = np.zeros((1, lstm.state_size.c), np.float32)
    h_init = np.zeros((1, lstm.state_size.h), np.float32)
    state_init = [c_init, h_init]

    c_in = tf.placeholder(
        tf.float32,
        shape=[1, lstm.state_size.c],
        name='c_in')
    h_in = tf.placeholder(tf.float32,
                          shape=[1, lstm.state_size.h],
                          name='h_in')
    state_in = [c_in, h_in]

    state_in = rnn.LSTMStateTuple(c_in, h_in)

    lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
        lstm, x, initial_state=state_in, sequence_length=step_size,
        time_major=False)
    lstm_outputs = tf.reshape(lstm_outputs, [-1, size])

    lstm_c, lstm_h = lstm_state
    state_out = [lstm_c[:1, :], lstm_h[:1, :]]
    return lstm_outputs, state_init, state_in, state_out


class SingleStepLSTM(object):
    """
    Class to be used to build the LSTM network.

    """

    def __init__(self, x, size, step_size):
        """
        Instantiate a regular LSTM network.

        Parameters
        ----------
        x : int
            set of attributes to be used in the network
        size : int
            size of the network
        step_size : int
            step size of the network
        """
        lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]

        c_in = tf.placeholder(tf.float32,
                              shape=[1, lstm.state_size.c],
                              name='c_in')
        h_in = tf.placeholder(tf.float32,
                              shape=[1, lstm.state_size.h],
                              name='h_in')
        self.state_in = [c_in, h_in]

        state_in = rnn.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_outputs = tf.reshape(lstm_outputs, [-1, size])

        lstm_c, lstm_h = lstm_state
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.output = lstm_outputs
