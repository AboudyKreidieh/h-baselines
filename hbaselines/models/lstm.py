from hbaselines.models.base import Model
import tensorflow as tf


class LSTM(Model):

    def __init__(self,
                 sess,
                 inputs,
                 num_outputs,
                 hidden_size,
                 stochastic,
                 scope):
        """Initialize an LSTM model.

        Parameters
        ----------
        sess : tf.Session
            tf session (shared between the model and algorithm
        inputs : tf.placeholder
            input placeholder
        num_outputs : int
            number of outputs from the neural network
        hidden_size : list of int
            a list that specified the shape of the neural network
        stochastic : bool
            a boolean operator that specifies whether the policy is meant to be
            stochastic or deterministic. If it is stochastic, an additional
            trainable variable is created to compute the logstd
        scope : str
            scope of the model
        """
        Model.__init__(self, sess, inputs)

        # store information on whether the policy is stochastic
        self.stochastic = stochastic

        with tf.variable_scope(scope):
            # Part 1. create the LSTM hidden layers

            # create the hidden layers
            lstm_layers = [tf.contrib.rnn.BasicLSTMCell(size)
                           for size in hidden_size]

            # stack up multiple LSTM layers
            cell = tf.contrib.rnn.MultiRNNCell(lstm_layers)

            # getting an initial state of all zeros
            lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, inputs)
            num_lstm_outputs = None  # FIXME

            # get the output from the hidden layers
            lstm_outputs = tf.reshape(lstm_outputs, [-1, num_lstm_outputs])

            # Part 2. Create the output from the LSTM model

            # initialize the weights and biases
            out_w = tf.get_variable(
                "W",
                [num_lstm_outputs, num_outputs],
                initializer=tf.truncated_normal_initializer())
            out_b = tf.get_variable(
                "b",
                [num_outputs],
                initializer=tf.zeros_initializer())

            # compute the output from the neural network model
            self.output_ph = tf.matmul(lstm_outputs[-1], out_w) + out_b

            # Part 3. Create the logstd for stochastic policies

            if stochastic:
                # Create a trainable variable whose output is the same size as
                # the action space. This variable will represent the output log
                # standard deviation of your stochastic policy.
                self.output_logstd = tf.get_variable(name="action_logstd",
                                                     shape=[num_outputs],
                                                     trainable=True)
