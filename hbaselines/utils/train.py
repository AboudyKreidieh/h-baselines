import argparse
import numpy as np

DEFAULT_DDPG_HP = dict(
    gamma=0.99,
    memory_policy=None,
    nb_train_steps=50,
    nb_rollout_steps=100,
    action_noise=None,
    normalize_observations=False,
    tau=0.001,
    batch_size=128,
    normalize_returns=False,
    observation_range=(-5, 5),
    critic_l2_reg=0.,
    return_range=(-np.inf, np.inf),
    actor_lr=1e-4,
    critic_lr=1e-3,
    clip_norm=None,
    reward_scale=1.,
    render=False,
    memory_limit=100,
    verbose=0,
    tensorboard_log=None,
    _init_setup_model=True
)


def get_hyperparameters(args, discrete):
    """Return the hyperparameters of a training algorithm from the parser."""
    if discrete:
        # hyperparameters for DQN
        hp = {}
    else:
        # hyperparameters for DDPG
        hp = DEFAULT_DDPG_HP.copy()
        hp.update(
            {"gamma": args.gamma,
             "nb_train_steps": args.nb_train_steps,
             "nb_rollout_steps": args.nb_rollout_steps,
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
             "memory_limit": int(args.memory_limit),
             "verbose": args.verbose}
        )

    return hp


def create_parser(description, example_usage):
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
    parser = create_ddpg_parser(parser)
    parser = create_dqn_parser(parser)

    return parser


def create_dqn_parser(parser):
    """Add the DQN hyperparameters to the parser."""
    return parser


def create_ddpg_parser(parser):
    """Add the DDPG hyperparameters to the parser."""
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount rate')
    parser.add_argument('--tau', type=float, default=0.001,
                        help='the soft update coefficient (keep old values, '
                             'between 0 and 1)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='the size of the batch for learning the policy')
    parser.add_argument('--reward_scale', type=float, default=1,
                        help='the value the reward should be scaled by')
    parser.add_argument('--actor_lr', type=float, default=1e-4,
                        help='the actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=1e-3,
                        help='the critic learning rate')

    parser.add_argument('--critic_l2_reg', type=float, default=0,
                        help='l2 regularizer coefficient')
    parser.add_argument('--clip_norm', type=float, default=None,
                        help='clip the gradients (disabled if None)')
    parser.add_argument('--nb_train_steps', type=int, default=50,
                        help='the number of training steps')
    parser.add_argument('--nb_rollout_steps', type=int, default=100,
                        help='the number of rollout steps')
    parser.add_argument('--normalize_observations', action='store_true',
                        help='should the observation be normalized')

    parser.add_argument('--render', action='store_true',
                        help='enable rendering of the environment')
    parser.add_argument('--verbose', type=int, default=2,
                        help='the verbosity level: 0 none, 1 training '
                             'information, 2 tensorflow debug')
    parser.add_argument('--memory_limit', type=int, default=1e5,
                        help='the max number of transitions to store')

    return parser
