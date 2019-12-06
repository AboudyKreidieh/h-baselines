"""
Helpers for scripts like run_atari.py.
"""
import random
import numpy as np
import tensorflow as tf
import os

import gym
from gym.wrappers import FlattenObservation, FilterObservation
from hbaselines.ppo.bench.monitor import Monitor
from hbaselines.ppo.vec_env.subproc_vec_env import SubprocVecEnv
from hbaselines.ppo.vec_env.dummy_vec_env import DummyVecEnv
from hbaselines.ppo.common.wrappers import ClipActionsWrapper


def make_vec_env(env_id,
                 num_env,
                 seed,
                 env_kwargs=None,
                 start_index=0,
                 flatten_dict_observations=True,
                 initializer=None,
                 force_dummy=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    env_kwargs = env_kwargs or {}
    log_dir = ""  # FIXME

    def make_thunk(rank, initializer=None):
        return lambda: make_env(
            env_id=env_id,
            subrank=rank,
            seed=seed,
            flatten_dict_observations=flatten_dict_observations,
            env_kwargs=env_kwargs,
            logger_dir=log_dir,
            initializer=initializer
        )

    # set global seeds
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not force_dummy and num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index,
                                         initializer=initializer)
                              for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(i + start_index, initializer=None)
                            for i in range(num_env)])


def make_env(env_id,
             mpi_rank=0,
             subrank=0,
             seed=None,
             flatten_dict_observations=True,
             env_kwargs=None,
             logger_dir=None,
             initializer=None):
    if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank)

    env_kwargs = env_kwargs or {}
    env = gym.make(env_id, **env_kwargs)

    if flatten_dict_observations and isinstance(env.observation_space,
                                                gym.spaces.Dict):
        env = FlattenObservation(env)

    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(
        env,
        logger_dir and os.path.join(logger_dir,
                                    str(mpi_rank) + '.' + str(subrank)),
        allow_early_resets=True)

    if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)

    return env


def make_mujoco_env(env_id, seed):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    # set global seeds
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(env_id)
    logger_path = None  # FIXME
    env = Monitor(env, logger_path, allow_early_resets=True)
    env.seed(seed)
    return env


def make_robotics_env(env_id, seed, rank=0):
    """Create a wrapped, monitored gym.Env for MuJoCo."""
    # set global seeds
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    logger_path = None  # FIXME

    env = gym.make(env_id)
    env = FlattenObservation(
        FilterObservation(env, ['observation', 'desired_goal']))
    env = Monitor(env, logger_path, info_keywords=('is_success',))
    env.seed(seed)
    return env


def arg_parser():
    """Create an empty argparse.ArgumentParser."""
    import argparse
    return argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def common_arg_parser():
    """Create an argparse.ArgumentParser for run_mujoco.py."""
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str,
                        default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--num_env', default=None, type=int,
                        help='Number of environment copies being run in '
                             'parallel. When not specified, set to number of '
                             'cpus for Atari, and to 1 for Mujoco')
    parser.add_argument('--reward_scale', default=1.0, type=float,
                        help='Reward scale factor. Default: 1.0')
    return parser


def robotics_arg_parser():
    """Create an argparse.ArgumentParser for run_mujoco.py."""
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str,
                        default='FetchReach-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser
