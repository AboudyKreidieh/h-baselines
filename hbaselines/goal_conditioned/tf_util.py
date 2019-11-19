"""TensorFlow utility methods."""
import tensorflow as tf


def make_session(num_cpu, graph=None):
    """Return a session that will use <num_cpu> CPU's only.

    Parameters
    ----------
    num_cpu : int
        number of CPUs to use for TensorFlow
    graph : tf.Graph
        the graph of the session

    Returns
    -------
    tf.compat.v1.Session
        a tensorflow session
    """
    tf_config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    # Prevent tensorflow from taking all the gpu memory
    tf_config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=tf_config, graph=graph)


def get_trainable_vars(name=None):
    """Return the trainable variables.

    Parameters
    ----------
    name : str
        the scope

    Returns
    -------
    list of tf.Variable
        trainable variables
    """
    return tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)


def get_globals_vars(name=None):
    """Return the global variables.

    Parameters
    ----------
    name : str
        the scope

    Returns
    -------
    list of tf.Variable
        global variables
    """
    return tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=name)


def reduce_std(tensor, axis=None, keepdims=False):
    """Get the standard deviation of a Tensor.

    Parameters
    ----------
    tensor : tf.Tensor or tf.Variable
        the input tensor
    axis : int or list of int
        the axis to itterate the std over
    keepdims : bool
        keep the other dimensions the same

    Returns
    -------
    tf.Tensor
        the std of the tensor
    """
    return tf.sqrt(reduce_var(tensor, axis=axis, keepdims=keepdims))


def reduce_var(tensor, axis=None, keepdims=False):
    """Get the variance of a Tensor.

    Parameters
    ----------
    tensor : tf.Tensor
        the input tensor
    axis : int or list of int
        the axis to itterate the variance over
    keepdims : bool
        keep the other dimensions the same

    Returns
    -------
    tf.Tensor
        the variance of the tensor
    """
    tensor_mean = tf.reduce_mean(tensor, axis=axis, keepdims=True)
    devs_squared = tf.square(tensor - tensor_mean)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def get_target_updates(_vars, target_vars, tau, verbose=0):
    """Get target update operations.

    Parameters
    ----------
    _vars : list of tf.Tensor
        the initial variables
    target_vars : list of tf.Tensor
        the target variables
    tau : float
        the soft update coefficient (keep old values, between 0 and 1)
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug

    Returns
    -------
    tf.Operation
        initial update
    tf.Operation
        soft update
    """
    if verbose >= 2:
        print('setting up target updates ...')

    soft_updates = []
    init_updates = []
    assert len(_vars) == len(target_vars)

    for var, target_var in zip(_vars, target_vars):
        if verbose >= 2:
            print('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.compat.v1.assign(target_var, var))
        soft_updates.append(
            tf.compat.v1.assign(target_var, (1.-tau) * target_var + tau * var))

    assert len(init_updates) == len(_vars)
    assert len(soft_updates) == len(_vars)

    return tf.group(*init_updates), tf.group(*soft_updates)
