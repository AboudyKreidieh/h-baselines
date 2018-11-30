import tensorflow as tf
import numpy as np


def build_lstm(inputs,
               num_outputs,
               hidden_size,
               scope,
               reuse,
               nonlinearity=None,
               weights_initializer=tf.random_uniform_initializer(
                   minval=-3e-3, maxval=3e-3),
               bias_initializer=tf.zeros_initializer()):
    """Initialize an LSTM model.

    The LSTM model is coupled with a linear combination of nodes at the end to
    return an output of the re

    Parameters
    ----------
    inputs : tf.placeholder
        input placeholder
    num_outputs : int
        number of outputs from the neural network
    hidden_size : int
        specified the shape of the neural network
    scope : str
        scope of the model
    reuse : bool
        whether to reuse the variables
    nonlinearity : tf.nn.*
        activation nonlinearity for the output of the model
    weights_initializer : tf.Operation
        initialization operation for the weights of the model
    bias_initializer : tf.Operation
        initialization operation for the biases of the model
    # TODO: add prefix
    """
    with tf.variable_scope(scope, reuse=reuse):
        lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        state_init = [c_init, h_init]

        c_in = tf.placeholder(tf.float32, shape=[None, lstm.state_size.c],
                              name='c_in')
        h_in = tf.placeholder(tf.float32, shape=[None, lstm.state_size.h],
                              name='h_in')
        state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, inputs,
            initial_state=state_in,
            # sequence_length=tf.shape(inputs)[:1],
            # time_major=False
        )
        lstm_outputs = tf.reshape(lstm_outputs, [-1, hidden_size])

        lstm_c, lstm_h = lstm_state
        state_out = [lstm_c[:1, :], lstm_h[:1, :]]

        w = tf.get_variable("W", [hidden_size, num_outputs],
                            initializer=weights_initializer)
        b = tf.get_variable("b", [num_outputs],
                            initializer=bias_initializer)

        if nonlinearity is None:
            output = tf.matmul(lstm_outputs, w) + b
        else:
            output = nonlinearity(tf.matmul(lstm_outputs, w) + b)

    return output, state_init, state_in, state_out
