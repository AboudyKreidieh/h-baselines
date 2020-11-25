"""Utility method for the exploration strategies classes."""
import numpy as np


def argmax_random(x):
    """Return the index of the max value of array x, if multiple occurance, return one randomly"""
    max_reward = np.amax(x)
    argmax_reward = np.argwhere(x == max_reward).flatten()
    return np.random.choice(argmax_reward)
