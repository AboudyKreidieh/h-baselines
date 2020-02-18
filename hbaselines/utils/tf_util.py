"""TensorFlow utility methods."""
import tensorflow as tf
import numpy as np
from functools import reduce

# Stabilizing term to avoid NaN (prevents division by zero or log of zero)
EPS = 1e-6


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


def gaussian_likelihood(input_, mu_, log_std):
    """Compute log likelihood of a gaussian.

    Here we assume this is a Diagonal Gaussian.

    Parameters
    ----------
    input_ : tf.Variable
        the action by the policy
    mu_ : tf.Variable
        the policy mean
    log_std : tf.Variable
        the policy log std

    Returns
    -------
    tf.Variable
        the log-probability of a given observation given the output action
        from the policy
    """
    pre_sum = -0.5 * (((input_ - mu_) / (
                tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(
        2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def apply_squashing_func(mu_, pi_, logp_pi):
    """Squash the output of the Gaussian distribution.

    This method also accounts for that in the log probability. The squashed
    mean is also returned for using deterministic actions.

    Parameters
    ----------
    mu_ : tf.Variable
        mean of the gaussian
    pi_ : tf.Variable
        output of the policy (or action) before squashing
    logp_pi : tf.Variable
        log probability before squashing

    Returns
    -------
    tf.Variable
        the output from the squashed deterministic policy
    tf.Variable
        the output from the squashed stochastic policy
    tf.Variable
        the log probability of a given squashed action
    """
    # Squash the output
    deterministic_policy = tf.nn.tanh(mu_)
    policy = tf.nn.tanh(pi_)

    # Squash correction (from original implementation)
    logp_pi -= tf.reduce_sum(tf.math.log(1 - policy ** 2 + EPS), axis=1)

    return deterministic_policy, policy, logp_pi


def print_params_shape(scope, param_type):
    """Print parameter shapes and number of parameters.

    Parameters
    ----------
    scope : str
        scope containing the parameters
    param_type : str
        the name of the parameter
    """
    shapes = [var.get_shape().as_list() for var in get_trainable_vars(scope)]
    nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in shapes])
    print('  {} shapes: {}'.format(param_type, shapes))
    print('  {} params: {}'.format(param_type, nb_params))
