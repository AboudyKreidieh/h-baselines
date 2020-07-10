"""Utility methods when performing training."""
import argparse
from hbaselines.algorithms.off_policy import TD3_PARAMS
from hbaselines.algorithms.off_policy import SAC_PARAMS
from hbaselines.algorithms.off_policy import FEEDFORWARD_PARAMS
from hbaselines.algorithms.off_policy import GOAL_CONDITIONED_PARAMS
from hbaselines.algorithms.utils import is_sac_policy, is_td3_policy
from hbaselines.algorithms.utils import is_goal_conditioned_policy
from hbaselines.algorithms.utils import is_multiagent_policy


def get_hyperparameters(args, policy):
    """Return the hyperparameters of a training algorithm from the parser."""
    algorithm_params = {
        "nb_train_steps": args.nb_train_steps,
        "nb_rollout_steps": args.nb_rollout_steps,
        "nb_eval_episodes": args.nb_eval_episodes,
        "actor_update_freq": args.actor_update_freq,
        "meta_update_freq": args.meta_update_freq,
        "reward_scale": args.reward_scale,
        "render": args.render,
        "render_eval": args.render_eval,
        "verbose": args.verbose,
        "num_envs": args.num_envs,
        "_init_setup_model": True,
    }

    # add FeedForwardPolicy parameters
    policy_kwargs = {
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "tau": args.tau,
        "gamma": args.gamma,
        "layer_norm": args.layer_norm,
        "use_huber": args.use_huber,
    }

    # add TD3 parameters
    if is_td3_policy(policy):
        policy_kwargs.update({
            "noise": args.noise,
            "target_policy_noise": args.target_policy_noise,
            "target_noise_clip": args.target_noise_clip,
        })

    # add SAC parameters
    if is_sac_policy(policy):
        policy_kwargs.update({
            "target_entropy": args.target_entropy,
        })

    # add GoalConditionedPolicy parameters
    if is_goal_conditioned_policy(policy):
        policy_kwargs.update({
            "num_levels": args.num_levels,
            "meta_period": args.meta_period,
            "intrinsic_reward_type": args.intrinsic_reward_type,
            "intrinsic_reward_scale": args.intrinsic_reward_scale,
            "relative_goals": args.relative_goals,
            "off_policy_corrections": args.off_policy_corrections,
            "hindsight": args.hindsight,
            "subgoal_testing_rate": args.subgoal_testing_rate,
            "connected_gradients": args.connected_gradients,
            "cg_weights": args.cg_weights,
            "use_fingerprints": args.use_fingerprints,
            "centralized_value_functions": args.centralized_value_functions,
        })

    # add MultiActorCriticPolicy parameters
    if is_multiagent_policy(policy):
        policy_kwargs.update({
            "shared": args.shared,
            "maddpg": args.maddpg,
        })

    # add the policy_kwargs term to the algorithm parameters
    algorithm_params['policy_kwargs'] = policy_kwargs

    return algorithm_params


def parse_options(description, example_usage, args):
    """Parse training options user can specify in command line.

    Parameters
    ----------
    description : str
        the description of the script using this parser
    example_usage : str
        an example of the runner script being used
    args : list of str
        command-line arguments

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        description=description, epilog=example_usage)

    # required input parameters
    parser.add_argument(
        'env_name', type=str,
        help='Name of the gym environment. This environment must either be '
             'registered in gym, be available in the computation framework '
             'Flow, or be available within the hbaselines/envs folder.')

    # optional input parameters
    parser.add_argument(
        '--alg', type=str, default='TD3',
        help='The algorithm to use. Must be one of [TD3, SAC].')
    parser.add_argument(
        '--evaluate', action='store_true',
        help='add an evaluation environment')
    parser.add_argument(
        '--n_training', type=int, default=1,
        help='Number of training operations to perform. Each training '
             'operation is performed on a new seed. Defaults to 1.')
    parser.add_argument(
        '--total_steps',  type=int, default=1000000,
        help='Total number of timesteps used during training.')
    parser.add_argument(
        '--seed', type=int, default=1,
        help='Sets the seed for numpy, tensorflow, and random.')
    parser.add_argument(
        '--log_interval', type=int, default=2000,
        help='the number of training steps before logging training results')
    parser.add_argument(
        '--eval_interval', type=int, default=50000,
        help='number of simulation steps in the training environment before '
             'an evaluation is performed')
    parser.add_argument(
        '--save_interval', type=int, default=50000,
        help='number of simulation steps in the training environment before '
             'the model is saved')
    parser.add_argument(
        '--initial_exploration_steps', type=int, default=10000,
        help='number of timesteps that the policy is run before training to '
             'initialize the replay buffer with samples')

    # algorithm-specific hyperparameters
    parser = create_algorithm_parser(parser)
    parser = create_td3_parser(parser)
    parser = create_sac_parser(parser)
    parser = create_feedforward_parser(parser)
    parser = create_goal_conditioned_parser(parser)
    parser = create_multi_feedforward_parser(parser)

    flags, _ = parser.parse_known_args(args)

    return flags


def create_algorithm_parser(parser):
    """Add the algorithm hyperparameters to the parser."""
    parser.add_argument(
        '--nb_train_steps', type=int, default=1,
        help='the number of training steps')
    parser.add_argument(
        '--nb_rollout_steps', type=int, default=1,
        help='the number of rollout steps')
    parser.add_argument(
        '--nb_eval_episodes', type=int, default=50,
        help='the number of evaluation episodes')
    parser.add_argument(
        '--reward_scale', type=float, default=1,
        help='the value the reward should be scaled by')
    parser.add_argument(
        '--render', action='store_true',
        help='enable rendering of the environment')
    parser.add_argument(
        '--render_eval', action='store_true',
        help='enable rendering of the evaluation environment')
    parser.add_argument(
        '--num_envs', type=int, default=1,
        help='number of environments used to run simulations in parallel. '
             'Each environment is run on a separate CPUS and uses the same '
             'policy as the rest. Must be less than or equal to '
             'nb_rollout_steps.')
    parser.add_argument(
        '--verbose', type=int, default=2,
        help='the verbosity level: 0 none, 1 training information, '
             '2 tensorflow debug')
    parser.add_argument(
        '--actor_update_freq', type=int, default=2,
        help='number of training steps per actor policy update step. The '
             'critic policy is updated every training step.')
    parser.add_argument(
        '--meta_update_freq', type=int, default=10,
        help='number of training steps per meta policy update step. The actor '
             'policy of the meta-policy is further updated at the frequency '
             'provided by the actor_update_freq variable. Note that this value'
             ' is only relevant when using the GoalConditionedPolicy policy.')

    return parser


def create_td3_parser(parser):
    """Add the TD3 hyperparameters to the parser."""
    parser.add_argument(
        "--noise",
        type=float,
        default=TD3_PARAMS["noise"],
        help="scaling term to the range of the action space, that is "
             "subsequently used as the standard deviation of Gaussian noise "
             "added to the action if `apply_noise` is set to True in "
             "`get_action`")
    parser.add_argument(
        "--target_policy_noise",
        type=float,
        default=TD3_PARAMS["target_policy_noise"],
        help="standard deviation term to the noise from the output of the "
             "target actor policy. See TD3 paper for more.")
    parser.add_argument(
        "--target_noise_clip",
        type=float,
        default=TD3_PARAMS["target_noise_clip"],
        help="clipping term for the noise injected in the target actor policy")

    return parser


def create_sac_parser(parser):
    """Add the SAC hyperparameters to the parser."""
    parser.add_argument(
        "--target_entropy",
        type=float,
        default=SAC_PARAMS["target_entropy"],
        help="target entropy used when learning the entropy coefficient. If "
             "set to None, a heuristic value is used.")

    return parser


def create_feedforward_parser(parser):
    """Add the feedforward policy hyperparameters to the parser."""
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=FEEDFORWARD_PARAMS["buffer_size"],
        help="the max number of transitions to store")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=FEEDFORWARD_PARAMS["batch_size"],
        help="the size of the batch for learning the policy")
    parser.add_argument(
        "--actor_lr",
        type=float,
        default=FEEDFORWARD_PARAMS["actor_lr"],
        help="the actor learning rate")
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=FEEDFORWARD_PARAMS["critic_lr"],
        help="the critic learning rate")
    parser.add_argument(
        "--tau",
        type=float,
        default=FEEDFORWARD_PARAMS["tau"],
        help="the soft update coefficient (keep old values, between 0 and 1)")
    parser.add_argument(
        "--gamma",
        type=float,
        default=FEEDFORWARD_PARAMS["gamma"],
        help="the discount rate")
    parser.add_argument(
        "--layer_norm",
        action="store_true",
        help="enable layer normalisation")
    parser.add_argument(
        "--use_huber",
        action="store_true",
        help="specifies whether to use the huber distance function as the "
             "loss for the critic. If set to False, the mean-squared error "
             "metric is used instead")

    return parser


def create_goal_conditioned_parser(parser):
    """Add the goal-conditioned policy hyperparameters to the parser."""
    parser.add_argument(
        "--num_levels",
        type=int,
        default=GOAL_CONDITIONED_PARAMS["num_levels"],
        help="number of levels within the hierarchy. Must be greater than 1. "
             "Two levels  correspond to a Manager/Worker paradigm.")
    parser.add_argument(
        "--meta_period",
        type=int,
        default=GOAL_CONDITIONED_PARAMS["meta_period"],
        help="meta-policy action period")
    parser.add_argument(
        "--intrinsic_reward_type",
        type=str,
        default=GOAL_CONDITIONED_PARAMS["intrinsic_reward_type"],
        help="the reward function to be used by the lower-level policies. See "
             "the base goal-conditioned policy for a description.")
    parser.add_argument(
        "--intrinsic_reward_scale",
        type=float,
        default=GOAL_CONDITIONED_PARAMS["intrinsic_reward_scale"],
        help="the value that the intrinsic reward should be scaled by")
    parser.add_argument(
        "--relative_goals",
        action="store_true",
        help="specifies whether the goal issued by the higher-level policies "
             "is meant to be a relative or absolute goal, i.e. specific state "
             "or change in state")
    parser.add_argument(
        "--off_policy_corrections",
        action="store_true",
        help="whether to use off-policy corrections during the update "
             "procedure. See: https://arxiv.org/abs/1805.08296")
    parser.add_argument(
        "--hindsight",
        action="store_true",
        help="whether to include hindsight action and goal transitions in the "
             "replay buffer. See: https://arxiv.org/abs/1712.00948")
    parser.add_argument(
        "--subgoal_testing_rate",
        type=float,
        default=GOAL_CONDITIONED_PARAMS["subgoal_testing_rate"],
        help="rate at which the original (non-hindsight) sample is stored in "
             "the replay buffer as well. Used only if `hindsight` is set to "
             "True.")
    parser.add_argument(
        "--use_fingerprints",
        action="store_true",
        help="specifies whether to add a time-dependent fingerprint to the "
             "observations")
    parser.add_argument(
        "--centralized_value_functions",
        action="store_true",
        help="specifies whether to use centralized value functions")
    parser.add_argument(
        "--connected_gradients",
        action="store_true",
        help="whether to use the connected gradient update actor update "
             "procedure to the higher-level policy. See: "
             "https://arxiv.org/abs/1912.02368v1")
    parser.add_argument(
        "--cg_weights",
        type=float,
        default=GOAL_CONDITIONED_PARAMS["cg_weights"],
        help="weights for the gradients of the loss of the lower-level "
             "policies with respect to the parameters of the higher-level "
             "policies. Only used if `connected_gradients` is set to True.")

    return parser


def create_multi_feedforward_parser(parser):
    """Add the multi-agent policy hyperparameters to the parser."""
    parser.add_argument(
        "--shared",
        action="store_true",
        help="whether to use a shared policy for all agents")
    parser.add_argument(
        "--maddpg",
        action="store_true",
        help="whether to use an algorithm-specific variant of the MADDPG "
             "algorithm")

    return parser
