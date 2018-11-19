import tensorflow as tf


def linear(inputs,
           num_outputs,
           scope,
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
    weights_initializer : tf.Operation
        initialization operation for the weights of the model
    bias_initializer : tf.Operation
        initialization operation for the biases of the model

    Returns
    -------
    tf.Variable
        output from the model
    """
    with tf.variable_scope(scope):
        w = tf.get_variable("W", [inputs.get_shape()[1], num_outputs],
                            initializer=weights_initializer)
        b = tf.get_variable("b", [num_outputs],
                            initializer=tf.constant_initializer(
                                bias_initializer))

    return tf.matmul(inputs, w) + b
