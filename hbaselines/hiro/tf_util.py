import tensorflow as tf
from stable_baselines import logger
import numpy as np
import os
import multiprocessing


def make_session(num_cpu=None, make_default=False, graph=None):
    """Return a session that will use <num_cpu> CPU's only.

    Parameters
    ----------
    num_cpu : int
        number of CPUs to use for TensorFlow
    make_default : bool
        if this should return an InteractiveSession or a normal Session
    graph : tf.Graph
        the graph of the session

    Returns
    -------
    tf.Session
        a tensorflow session
    """
    if num_cpu is None:
        num_cpu = int(os.getenv("RCALL_NUM_CPU", multiprocessing.cpu_count()))
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    # Prevent tensorflow from taking all the gpu memory
    tf_config.gpu_options.allow_growth = True
    if make_default:
        return tf.InteractiveSession(config=tf_config, graph=graph)
    else:
        return tf.Session(config=tf_config, graph=graph)


def get_trainable_vars(name):
    """Returns the trainable variable.

    Parameters
    ----------
    name : str
        the scope

    Returns
    -------
    list of tf.Variable
        trainable variables
    """
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)


def get_globals_vars(name):
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
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)


def normalize(tensor, stats):
    """Normalize a tensor using a running mean and std.

    Parameters
    ----------
    tensor : tf.Tensor
        the input tensor
    stats : RunningMeanStd
        the running mean and std of the input to normalize

    Returns
    -------
    tf.Tensor
        the normalized tensor
    """
    if stats is None:
        return tensor
    return (tensor - stats.mean) / stats.std


def denormalize(tensor, stats):
    """Denormalize a tensor using a running mean and std.

    Parameters
    ----------
    tensor : tf.Variable
        the normalized tensor
    stats : RunningMeanStd
        the running mean and std of the input to normalize

    Returns
    -------
    tf.Tensor
        the restored tensor
    """
    if stats is None:
        return tensor
    return tensor * stats.std + stats.mean


def reduce_std(tensor, axis=None, keepdims=False):
    """Get the standard deviation of a Tensor.

    Parameters
    ----------
    tensor : tf.Tensor
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
        logger.info('setting up target updates ...')

    soft_updates = []
    init_updates = []
    assert len(_vars) == len(target_vars)

    for var, target_var in zip(_vars, target_vars):
        if verbose >= 2:
            logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(
            tf.assign(target_var, (1. - tau) * target_var + tau * var))

    assert len(init_updates) == len(_vars)
    assert len(soft_updates) == len(_vars)

    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbed_actor_updates(actor,
                                perturbed_actor,
                                param_noise_stddev,
                                verbose=0):
    """Get the actor update, with noise.

    Parameters
    ----------
    actor : str
        the actor
    perturbed_actor : str
        the perturbed actor
    param_noise_stddev : float
        the std of the parameter noise
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug

    Returns
    -------
    tf.Operation
        the update function
    """
    assert len(get_globals_vars(actor)) == \
        len(get_globals_vars(perturbed_actor))
    assert \
        len([var for var in get_trainable_vars(actor)
             if 'LayerNorm' not in var.name]) == \
        len([var for var in get_trainable_vars(perturbed_actor)
             if 'LayerNorm' not in var.name])

    updates = []
    for var, perturbed_var in zip(get_globals_vars(actor),
                                  get_globals_vars(perturbed_actor)):
        if var in [var for var in get_trainable_vars(actor)
                   if 'LayerNorm' not in var.name]:
            if verbose >= 2:
                logger.info('  {} <- {} + noise'.format(
                    perturbed_var.name, var.name))
            updates.append(
                tf.assign(perturbed_var, var + tf.random_normal(
                    tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            if verbose >= 2:
                logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(get_globals_vars(actor))
    return tf.group(*updates)


def var_shape(tensor):
    """Get TensorFlow Tensor shape.

    Parameters
    ----------
    tensor : tf.Tensor
        the input tensor

    Returns
    -------
    list of int
        the shape
    """
    out = tensor.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(tensor):
    """Get TensorFlow Tensor's number of elements.

    Parameters
    ----------
    tensor : tf.Tensor
        the input tensor

    Returns
    -------
    int
        the number of elements
    """
    return intprod(var_shape(tensor))


def intprod(tensor):
    """Calculate the product of all the elements in a list.

    Parameters
    ----------
    tensor : array_like
        the list of elements

    Returns
    -------
    int
        the product truncated
    """
    return int(np.prod(tensor))


def flatgrad(loss, var_list, clip_norm=None):
    """Calculate the gradient and flatten it.

    Parameters
    ----------
    loss : float
        the loss value
    var_list : list of tf.Tensor
        the variables
    clip_norm : float
        clip the gradients (disabled if None)

    Returns
    -------
    list of tf.Tensor
        flattened gradient
    """
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])
