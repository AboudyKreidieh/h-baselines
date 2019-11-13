"""Miscellaneous utility methods for this repository,"""
import os
import errno
import numpy as np
from gym.spaces import Box


def ensure_dir(path):
    """Ensure that the directory specified exists, and if not, create it."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise  # pragma: no cover
    return path


def get_manager_ac_space(ob_space,
                         relative_goals,
                         env_name,
                         use_fingerprints,
                         fingerprint_dim):
    """Compute the action space for the Manager.
    If the fingerprint terms are being appended onto the observations, this
    should be removed from the action space.
    Parameters
    ----------
    ob_space : gym.spaces.*
        the observation space of the environment
    relative_goals : bool
        specifies whether the goal issued by the Manager is meant to be a
        relative or absolute goal, i.e. specific state or change in state
    env_name : str
        the name of the environment. Used for special cases to assign the
        Manager action space to only ego observations in the observation space.
    use_fingerprints : bool
        specifies whether to add a time-dependent fingerprint to the
        observations
    fingerprint_dim : tuple of int
        the shape of the fingerprint elements, if they are being used
    Returns
    -------
    gym.spaces.Box
        the action space of the Manager policy
    """
    if env_name in ["AntMaze", "AntPush", "AntFall", "AntGather"]:
        manager_ac_space = Box(
            low=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3, -0.5,
                          -0.3, -0.5, -0.3, -0.5, -0.3]),
            high=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3,
                           0.5, 0.3, 0.5, 0.3]),
            dtype=np.float32,
        )
    elif env_name == "UR5":
        manager_ac_space = Box(
            low=np.array([-2 * np.pi, -2 * np.pi, -2 * np.pi, -4, -4, -4]),
            high=np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 4, 4, 4]),
            dtype=np.float32,
        )
    elif env_name == "Pendulum":
        manager_ac_space = Box(
            low=np.array([-np.pi, -15]),
            high=np.array([np.pi, 15]),
            dtype=np.float32
        )
    elif env_name == "figureeight0":
        if relative_goals:
            manager_ac_space = Box(-.5, .5, shape=(1,), dtype=np.float32)
        else:
            manager_ac_space = Box(0, 1, shape=(1,), dtype=np.float32)
    elif env_name == "figureeight1":
        if relative_goals:
            manager_ac_space = Box(-.5, .5, shape=(7,), dtype=np.float32)
        else:
            manager_ac_space = Box(0, 1, shape=(7,), dtype=np.float32)
    elif env_name == "figureeight2":
        if relative_goals:
            manager_ac_space = Box(-.5, .5, shape=(14,), dtype=np.float32)
        else:
            manager_ac_space = Box(0, 1, shape=(14,), dtype=np.float32)
    elif env_name == "merge0":
        if relative_goals:
            manager_ac_space = Box(-.5, .5, shape=(5,), dtype=np.float32)
        else:
            manager_ac_space = Box(0, 1, shape=(5,), dtype=np.float32)
    elif env_name == "merge1":
        if relative_goals:
            manager_ac_space = Box(-.5, .5, shape=(13,), dtype=np.float32)
        else:
            manager_ac_space = Box(0, 1, shape=(13,), dtype=np.float32)
    elif env_name == "merge2":
        if relative_goals:
            manager_ac_space = Box(-.5, .5, shape=(17,), dtype=np.float32)
        else:
            manager_ac_space = Box(0, 1, shape=(17,), dtype=np.float32)
    else:
        if use_fingerprints:
            low = np.array(ob_space.low)[:-fingerprint_dim[0]]
            high = ob_space.high[:-fingerprint_dim[0]]
            manager_ac_space = Box(low=low, high=high, dtype=np.float32)
        else:
            manager_ac_space = ob_space

    return manager_ac_space
