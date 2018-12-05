import tensorflow as tf


def build_fcnet(inputs,
                num_outputs,
                hidden_size,
                hidden_nonlinearity,
                output_nonlinearity,
                scope,
                reuse,
                prefix='fc'):
    """Create a deep feedforward neural network model.

    Parameters
    ----------
    inputs : tf.placeholder
        input placeholder
    num_outputs : int
        number of outputs from the neural network
    hidden_size : list of int
        a list that specified the shape of the neural network
    hidden_nonlinearity : tf.nn.*
        activation nonlinearity for the hidden nodes
    output_nonlinearity : tf.nn.*
        activation nonlinearity for the output nodes
    scope : str
        scope of the model
    reuse : bool
        whether to reuse the variables
    prefix : str, optional
        prefix to the names of variables

    Returns
    -------
    tf.Variable
        output from the model
    """
    with tf.variable_scope(scope, reuse=reuse):
        # input to the system
        last_layer = inputs

        # create the hidden layers
        for i, hidden in enumerate(hidden_size):
            last_layer = tf.layers.dense(
                inputs=last_layer,
                units=hidden,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='{}_{}'.format(prefix, i),
                activation=hidden_nonlinearity
            )

        # create the output layer
        policy = tf.layers.dense(
            inputs=last_layer,
            units=num_outputs,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-3e-3, maxval=3e-3),
            name='{}_output'.format(prefix),
            activation=output_nonlinearity
        )

    return policy
