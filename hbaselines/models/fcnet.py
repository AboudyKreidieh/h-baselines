import tensorflow as tf


def fully_connected_network(inputs,
                            num_outputs,
                            hidden_size,
                            nonlinearity,
                            stochastic,
                            scope):
    """Create a deep feedforward neural network model.

    Parameters
    ----------
    inputs : tf.placeholder
        input placeholder
    num_outputs : int
        number of outputs from the neural network
    hidden_size : list of int
        a list that specified the shape of the neural network
    nonlinearity : tf.nn.*
        activation nonlinearity
    stochastic : bool
        a boolean operator that specifies whether the policy is meant to be
        stochastic or deterministic. If it is stochastic, an additional
        trainable variable is created to compute the logstd
    scope : str
        scope of the model

    Returns
    -------
    tf.Variable
        output mean from the model
    tf.Variable or None
        output std from the model. If the model is not stochastic, then this
        output is set to None.
    """
    with tf.variable_scope(scope):
        # create the hidden layers
        last_layer = inputs
        for i, hidden in enumerate(hidden_size):
            last_layer = tf.layers.dense(
                inputs=last_layer,
                units=hidden,
                activation=nonlinearity)

        # create the output layer
        output_mean = tf.layers.dense(
            inputs=last_layer,
            units=num_outputs,
            activation=None)

    if stochastic:
        # Create a trainable variable whose output is the same size as the
        # action space. This variable will represent the output log standard
        # deviation of your stochastic policy.
        output_logstd = tf.get_variable(name="action_logstd",
                                        shape=[num_outputs],
                                        trainable=True)
    else:
        output_logstd = None

    return output_mean, output_logstd
