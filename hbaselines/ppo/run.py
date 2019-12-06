import sys
import multiprocessing
import os
from time import strftime
import gym
from collections import defaultdict

from hbaselines.ppo.vec_env import VecNormalize
from hbaselines.ppo.vec_env.vec_video_recorder import VecVideoRecorder
from hbaselines.ppo.cmd_util import common_arg_parser, parse_unknown_args, \
    make_vec_env
from importlib import import_module
from hbaselines.ppo.algorithm import PPO

_game_envs = defaultdict(set)
for env_i in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_i_type = env_i.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_i_type].add(env_i.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_id = args.env

    # Define the log directory.
    dir_name = os.path.join(
        "results", "{}/{}".format(env_id, strftime("%Y-%m-%d-%H:%M:%S")))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    alg_kwargs = get_learn_function_defaults(args.alg)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(
            env,
            os.path.join(dir_name, "videos"),
            record_video_trigger=lambda x: x % args.save_video_interval == 0,
            video_length=args.save_video_length
        )

    print('Training {} on {} with arguments \n{}'.format(
        args.alg, env_id, alg_kwargs))

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

    env_type = 'mujoco'  # FIXME
    if env_type == 'mujoco':
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


def parse_cmdline_kwargs(args):
    """Convert a list of '='-spaced command-line arguments to a dictionary.

    Evaluate python objects when possible.
    """
    def parse(v):
        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}


def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    _, _ = train(args, extra_args)


if __name__ == '__main__':
    main(sys.argv)
