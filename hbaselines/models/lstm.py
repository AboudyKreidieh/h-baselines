import tensorflow as tf
import numpy as np


def build_lstm(inputs,
               hidden_size,
               step_size,
               scope,
               reuse):
    """Initialize an LSTM model.

    Parameters
    ----------
    inputs : tf.placeholder
        input placeholder
    hidden_size : list of int
        a list that specified the shape of the neural network
    step_size : int
        TODO: fill in
    scope : str
        scope of the model
    reuse : bool
        whether to reuse the variables
    # TODO: add prefix
    """
    with tf.variable_scope(scope, reuse=reuse):
        lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        state_init = [c_init, h_init]

        c_in = tf.placeholder(tf.float32,
                              shape=[1, lstm.state_size.c],
                              name='c_in')
        h_in = tf.placeholder(tf.float32,
                              shape=[1, lstm.state_size.h],
                              name='h_in')
        # state_in = [c_in, h_in]  FIXME?
        state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, inputs,
            initial_state=state_in,
            sequence_length=step_size,
            time_major=False)
        lstm_outputs = tf.reshape(lstm_outputs, [-1, hidden_size])

        lstm_c, lstm_h = lstm_state
        state_out = [lstm_c[:1, :], lstm_h[:1, :]]

    return lstm_outputs, state_init, state_in, state_out
