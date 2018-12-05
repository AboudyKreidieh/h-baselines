"""A few statistical methods to be performed on tensors."""

import tensorflow as tf


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
    tensor : tf.Tensor
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
        the axis to iterate the variance over
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
