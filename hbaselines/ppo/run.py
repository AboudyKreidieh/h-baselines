import sys
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from hbaselines.ppo.vec_env import VecNormalize, VecEnv
from hbaselines.ppo.vec_env.vec_video_recorder import VecVideoRecorder
from hbaselines.ppo.cmd_util import common_arg_parser, parse_unknown_args, \
    make_vec_env
from hbaselines.ppo.tf_util import get_session
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

    log_dir = ""  # FIXME

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    alg_kwargs = get_learn_function_defaults(args.alg)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(
            env,
            osp.join(log_dir, "videos"),
            record_video_trigger=lambda x: x % args.save_video_interval == 0,
            video_length=args.save_video_length
        )

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network()

    print('Training {} on {} with arguments \n{}'.format(
        args.alg, env_id, alg_kwargs))

    if "log_interval" in alg_kwargs:
        log_interval = alg_kwargs.pop("log_interval")
    if "save_interval" in alg_kwargs:
        save_interval = alg_kwargs.pop("save_interval")

    alg = PPO(env=env, **alg_kwargs)
    model = alg.learn(total_timesteps=total_timesteps,
                      seed=seed,
                      log_interval=log_interval)

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin':
        ncpu //= 2
    alg = args.alg
    seed = args.seed

    env_id = args.env

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    flatten_dict_observations = alg not in {'her'}
    env = make_vec_env(
        env_id,
        args.num_env or 1,
        seed,
        flatten_dict_observations=flatten_dict_observations
    )

    env_type = 'mujoco'  # FIXME
    if env_type == 'mujoco':
        env = VecNormalize(env, use_tf=True)

    return env


def get_default_network():
    return 'mlp'


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

    model, env = train(args, extra_args)

    if args.save_path is not None:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        print("Running trained model")
        obs = env.reset()

        state = getattr(model, 'initial_state', None)
        dones = np.zeros((1,))

        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) \
            else np.zeros(1)
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew
            env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                for i in np.nonzero(done)[0]:
                    print('episode_rew={}'.format(episode_rew[i]))
                    episode_rew[i] = 0

    env.close()

    return model


if __name__ == '__main__':
    main(sys.argv)
