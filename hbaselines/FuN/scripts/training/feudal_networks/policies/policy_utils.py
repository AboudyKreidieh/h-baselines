"""
###########################################################################
# Script containing utility functions for the Neural Network policy class #
###########################################################################
"""


import numpy as np
import tensorflow as tf


def flatten(x):
    """
     Flattening function for the neural network

    Parameters
    ----------
    x : object
        Layer to be flattened in neural network
    """
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def categorical_sample(logits, d):
    """
     Flattening function for the neural network

    Parameters
    ----------
    logits : object
        Log-odd function
    d : object
        BLANK
    """
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(
        logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)
