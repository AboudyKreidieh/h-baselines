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
    elif env_name in ["ring0", "ring1"]:
        if relative_goals:
            manager_ac_space = Box(-.5, .5, shape=(1,), dtype=np.float32)
        else:
            manager_ac_space = Box(0, 1, shape=(1,), dtype=np.float32)
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


def get_state_indices(ob_space,
                      env_name,
                      use_fingerprints,
                      fingerprint_dim):
    """Return the state indices for the worker rewards.

    This assigns the indices of the state that are assigned goals, and
    subsequently rewarded for performing those goals.

    Parameters
    ----------
    ob_space : gym.spaces.*
        the observation space of the environment
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
    list of int
        the state indices that are assigned goals
    """
    # remove the last element to compute the reward FIXME
    if use_fingerprints:
        state_indices = list(np.arange(
            0, ob_space.shape[0] - fingerprint_dim[0]))
    else:
        state_indices = None

    if env_name in ["AntMaze", "AntPush", "AntFall", "AntGather"]:
        state_indices = list(np.arange(0, 15))
    elif env_name == "UR5":
        state_indices = None
    elif env_name == "Pendulum":
        state_indices = [0, 2]
    elif env_name in ["ring0", "ring1"]:
        state_indices = [0]
    elif env_name == "figureeight0":
        state_indices = [13]
    elif env_name == "figureeight1":
        state_indices = [i for i in range(1, 14, 2)]
    elif env_name == "figureeight2":
        state_indices = [i for i in range(14)]
    elif env_name == "merge0":
        state_indices = [5 * i for i in range(5)]
    elif env_name == "merge1":
        state_indices = [5 * i for i in range(13)]
    elif env_name == "merge2":
        state_indices = [5 * i for i in range(17)]

    return state_indices
