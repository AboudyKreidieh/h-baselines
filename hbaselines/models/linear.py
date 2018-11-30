import tensorflow as tf


def build_linear(inputs,
                 num_outputs,
                 scope,
                 reuse,
                 nonlinearity=None,
                 weights_initializer=None,
                 bias_initializer=0):
    """Create a linear model.

    Parameters
    ----------
    inputs : tf.placeholder
        input placeholder
    num_outputs : int
        number of outputs from the neural network
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
    """
    with tf.variable_scope(scope, reuse=reuse):
        w = tf.get_variable("W", [inputs.get_shape()[1], num_outputs],
                            initializer=weights_initializer)
        b = tf.get_variable("b", [num_outputs],
                            initializer=tf.constant_initializer(
                                bias_initializer))

        if nonlinearity is None:
            output = tf.matmul(inputs, w) + b
        else:
            output = nonlinearity(tf.matmul(inputs, w) + b)

    return output
