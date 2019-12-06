import sys
import multiprocessing
import os
from time import strftime
from importlib import import_module

from hbaselines.ppo.vec_env import VecNormalize
from hbaselines.ppo.cmd_util import common_arg_parser, make_vec_env
from hbaselines.ppo.algorithm import PPO


def train(args):
    env_id = args.env

    # Define the log directory.
    dir_name = os.path.join(
        "results", "{}/{}".format(env_id, strftime("%Y-%m-%d-%H:%M:%S")))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    alg_kwargs = get_learn_function_defaults('ppo2')

    env = build_env(args)

    if "log_interval" in alg_kwargs:
        log_interval = alg_kwargs.pop("log_interval")
    else:
        log_interval = 1000

    if "save_interval" in alg_kwargs:
        save_interval = alg_kwargs.pop("save_interval")
    else:
        save_interval = 0

    # Perform the training operation.
    alg = PPO(env=env, **alg_kwargs)
    model = alg.learn(
        total_timesteps=total_timesteps,
        log_dir=dir_name,
        seed=seed,
        log_interval=log_interval,
        save_interval=save_interval
    )

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin':
        ncpu //= 2
    seed = args.seed

    env_id = args.env

    env = make_vec_env(
        env_id,
        args.num_env or 1,
        seed,
        flatten_dict_observations=True
    )

    # Return the normalized form of the environment.
    env = VecNormalize(env, use_tf=True)

    return env


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_algs', alg, submodule]))

    return alg_module


def get_learn_function_defaults(alg):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, 'mujoco')()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)

    _, _ = train(args)


if __name__ == '__main__':
    main(sys.argv)
