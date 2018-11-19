from hbaselines.models.base import Model
import tensorflow as tf


class LinearModel(Model):

    def __init__(self,
                 sess,
                 inputs,
                 num_outputs,
                 scope,
                 weights_initializer=None,
                 bias_initializer=0):
        """Create a linear model.

        Parameters
        ----------
        sess : tf.Session
            tf session (shared between the model and algorithm
        inputs : tf.placeholder
            input placeholder
        num_outputs : int
            number of outputs from the neural network
        scope : str
            scope of the model
        weights_initializer : tf.Operation
            initialization operation for the weights of the model
        bias_initializer : tf.Operation
            initialization operation for the biases of the model
        """
        Model.__init__(self, sess, inputs)

        with tf.variable_scope(scope):
            w = tf.get_variable("W", [inputs.get_shape()[1], num_outputs],
                                initializer=weights_initializer)
            b = tf.get_variable("b", [num_outputs],
                                initializer=tf.constant_initializer(
                                    bias_initializer))

        self.output_ph = tf.matmul(inputs, w) + b

    def get_action(self, state):
        """See parent class."""
        self.sess.run(self.output_ph, feed_dict={self.input_ph: state})
