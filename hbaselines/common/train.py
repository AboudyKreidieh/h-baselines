"""Utility methods when performing training."""
import argparse
import numpy as np
import os
import errno

DEFAULT_TD3_HP = dict(
    gamma=0.99,
    nb_train_steps=1,
    nb_rollout_steps=1,
    nb_eval_episodes=50,
    normalize_observations=False,
    tau=0.001,
    batch_size=128,
    normalize_returns=False,
    observation_range=(-5., 5.),
    critic_l2_reg=0.,
    return_range=(-np.inf, np.inf),
    actor_lr=1e-4,
    critic_lr=1e-3,
    clip_norm=None,
    reward_scale=1.,
    render=False,
    render_eval=False,
    buffer_size=100000,
    random_exploration=0.0,
    verbose=2,
    _init_setup_model=True,
    policy_kwargs=None
)


def get_hyperparameters(args):
    """Return the hyperparameters of a training algorithm from the parser."""
    hp = DEFAULT_TD3_HP.copy()

    hp.update({
        "gamma": args.gamma,
        "nb_train_steps": args.nb_train_steps,
        "nb_rollout_steps": args.nb_rollout_steps,
        "nb_eval_episodes": args.nb_eval_episodes,
        "normalize_observations": args.normalize_observations,
        "tau": args.tau,
        "batch_size": args.batch_size,
        # "normalize_returns": args.normalize_returns,
        "observation_range": (-5, 5),  # TODO: figure this out
        "critic_l2_reg": args.critic_l2_reg,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "clip_norm": args.clip_norm,
        "reward_scale": args.reward_scale,
        "render": args.render,
        "buffer_size": int(args.buffer_size),
        "verbose": args.verbose
    })

    return hp


def parse_options(description, example_usage, args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=description, epilog=example_usage)

    # required input parameters
    parser.add_argument(
        'env_name', type=str,
        help='Name of the gym environment. This environment must either be '
             'registered in gym, be available in the computation framework '
             'Flow, or be available within the hbaselines/envs folder.')

    # optional input parameters
    parser.add_argument(
        '--n_training', type=int, default=1,
        help='Number of training operations to perform. Each training '
             'operation is performed on a new seed. Defaults to 1.')
    parser.add_argument(
        '--steps',  type=int, default=1e6,
        help='Total number of timesteps used during training.')

    # algorithm-specific hyperparameters
    parser = create_td3_parser(parser)

    flags, _ = parser.parse_known_args(args)

    return flags


def create_td3_parser(parser):
    """Add the TD3 hyperparameters to the parser."""
    parser.add_argument('--gamma',
                        type=float,
                        default=DEFAULT_TD3_HP['gamma'],
                        help='the discount rate')
    parser.add_argument('--tau',
                        type=float,
                        default=DEFAULT_TD3_HP['tau'],
                        help='the soft update coefficient (keep old values, '
                             'between 0 and 1)')
    parser.add_argument('--batch_size',
                        type=int,
                        default=DEFAULT_TD3_HP['batch_size'],
                        help='the size of the batch for learning the policy')
    parser.add_argument('--reward_scale',
                        type=float,
                        default=DEFAULT_TD3_HP['reward_scale'],
                        help='the value the reward should be scaled by')
    parser.add_argument('--actor_lr',
                        type=float,
                        default=DEFAULT_TD3_HP['actor_lr'],
                        help='the actor learning rate')
    parser.add_argument('--critic_lr',
                        type=float,
                        default=DEFAULT_TD3_HP['critic_lr'],
                        help='the critic learning rate')
    parser.add_argument('--critic_l2_reg',
                        type=float,
                        default=DEFAULT_TD3_HP['critic_l2_reg'],
                        help='l2 regularizer coefficient')
    parser.add_argument('--clip_norm',
                        type=float,
                        default=DEFAULT_TD3_HP['clip_norm'],
                        help='clip the gradients (disabled if None)')
    parser.add_argument('--nb_train_steps',
                        type=int,
                        default=DEFAULT_TD3_HP['nb_train_steps'],
                        help='the number of training steps')
    parser.add_argument('--nb_rollout_steps',
                        type=int,
                        default=DEFAULT_TD3_HP['nb_rollout_steps'],
                        help='the number of rollout steps')
    parser.add_argument('--nb_eval_episodes',
                        type=int,
                        default=DEFAULT_TD3_HP['nb_eval_episodes'],
                        help='the number of evaluation episodes')
    parser.add_argument('--normalize_observations',
                        action='store_true',
                        help='should the observation be normalized')
    parser.add_argument('--render',
                        action='store_true',
                        help='enable rendering of the environment')
    parser.add_argument('--verbose',
                        type=int,
                        default=DEFAULT_TD3_HP['verbose'],
                        help='the verbosity level: 0 none, 1 training '
                             'information, 2 tensorflow debug')
    parser.add_argument('--buffer_size',
                        type=int,
                        default=DEFAULT_TD3_HP['buffer_size'],
                        help='the max number of transitions to store')
    parser.add_argument('--evaluate',
                        action='store_true',
                        help='add an evaluation environment')

    return parser


def ensure_dir(path):
    """Ensure that the directory specified exists, and if not, create it."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path
