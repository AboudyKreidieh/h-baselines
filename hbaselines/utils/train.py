"""Utility methods when performing training."""
import argparse
from hbaselines.algorithms.rl_algorithm import TD3_PARAMS
from hbaselines.algorithms.rl_algorithm import SAC_PARAMS
from hbaselines.algorithms.rl_algorithm import PPO_PARAMS
from hbaselines.algorithms.rl_algorithm import FEEDFORWARD_PARAMS
from hbaselines.algorithms.rl_algorithm import GOAL_CONDITIONED_PARAMS
from hbaselines.algorithms.utils import is_sac_policy
from hbaselines.algorithms.utils import is_td3_policy
from hbaselines.algorithms.utils import is_ppo_policy
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
        "save_replay_buffer": args.save_replay_buffer,
        "verbose": args.verbose,
        "num_envs": args.num_envs,
        "_init_setup_model": True,
    }

    # add FeedForwardPolicy parameters
    policy_kwargs = {
        "model_params": {
            "model_type": getattr(args, "model_params:model_type"),
            "layer_norm": getattr(args, "model_params:layer_norm"),
            "ignore_image": getattr(args, "model_params:ignore_image"),
            "image_height": getattr(args, "model_params:image_height"),
            "image_width": getattr(args, "model_params:image_width"),
            "image_channels": getattr(args, "model_params:image_channels"),
            "ignore_flat_channels":
                getattr(args, "model_params:ignore_flat_channels") or
                FEEDFORWARD_PARAMS["model_params"]["ignore_flat_channels"],
            "filters":
                getattr(args, "model_params:filters") or
                FEEDFORWARD_PARAMS["model_params"]["filters"],
            "kernel_sizes":
                getattr(args, "model_params:kernel_sizes") or
                FEEDFORWARD_PARAMS["model_params"]["kernel_sizes"],
            "strides":
                getattr(args, "model_params:strides") or
                FEEDFORWARD_PARAMS["model_params"]["strides"],
            "layers":
                getattr(args, "model_params:layers") or
                FEEDFORWARD_PARAMS["model_params"]["layers"],
        }
    }

    # add TD3 parameters
    if is_td3_policy(policy):
        policy_kwargs.update({
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "tau": args.tau,
            "gamma": args.gamma,
            "use_huber": args.use_huber,
            "noise": args.noise,
            "target_policy_noise": args.target_policy_noise,
            "target_noise_clip": args.target_noise_clip,
        })

    # add SAC parameters
    if is_sac_policy(policy):
        policy_kwargs.update({
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "tau": args.tau,
            "gamma": args.gamma,
            "use_huber": args.use_huber,
            "target_entropy": args.target_entropy,
        })

    # add PPO parameters
    if is_ppo_policy(policy):
        policy_kwargs.update({
            "learning_rate": args.learning_rate,
            "n_minibatches": args.n_minibatches,
            "n_opt_epochs": args.n_opt_epochs,
            "gamma": args.gamma,
            "lam": args.lam,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
            "cliprange": args.cliprange,
            "cliprange_vf": args.cliprange_vf,
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
            "cooperative_gradients": args.cooperative_gradients,
            "cg_weights": args.cg_weights,
            "pretrain_worker": args.pretrain_worker,
            "pretrain_path": args.pretrain_path,
            "pretrain_ckpt": args.pretrain_ckpt,
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


def parse_options(description,
                  example_usage,
                  args,
                  multiagent=True,
                  hierarchical=True):
    """Parse training options user can specify in command line.

    Parameters
    ----------
    description : str
        the description of the script using this parser
    example_usage : str
        an example of the runner script being used
    args : list of str
        command-line arguments
    multiagent : bool
        whether the policy is supposed to be multiagent
    hierarchical : bool
        whether the policy is supposed to be hierarchical

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser_algorithm = argparse.ArgumentParser(
        description=description, epilog=example_usage)

    # required input parameters
    parser_algorithm.add_argument(
        'env_name', type=str,
        help='Name of the gym environment. This environment must either be '
             'registered in gym, be available in the computation framework '
             'Flow, or be available within the hbaselines/envs folder.')

    # optional input parameters
    parser_algorithm.add_argument(
        '--alg', type=str, default='TD3',
        help='The algorithm to use. Must be one of [TD3, SAC].')
    parser_algorithm.add_argument(
        '--evaluate', action='store_true',
        help='add an evaluation environment')
    parser_algorithm.add_argument(
        '--n_training', type=int, default=1,
        help='Number of training operations to perform. Each training '
             'operation is performed on a new seed. Defaults to 1.')
    parser_algorithm.add_argument(
        '--total_steps',  type=int, default=1000000,
        help='Total number of timesteps used during training.')
    parser_algorithm.add_argument(
        '--seed', type=int, default=1,
        help='Sets the seed for numpy, tensorflow, and random.')
    parser_algorithm.add_argument(
        '--log_interval', type=int, default=2000,
        help='the number of training steps before logging training results')
    parser_algorithm.add_argument(
        '--eval_interval', type=int, default=50000,
        help='number of simulation steps in the training environment before '
             'an evaluation is performed')
    parser_algorithm.add_argument(
        '--save_interval', type=int, default=50000,
        help='number of simulation steps in the training environment before '
             'the model is saved')
    parser_algorithm.add_argument(
        '--initial_exploration_steps', type=int, default=10000,
        help='number of timesteps that the policy is run before training to '
             'initialize the replay buffer with samples')

    parser_algorithm = create_algorithm_parser(parser_algorithm)
    [args_alg, extras_alg] = parser_algorithm.parse_known_args(args)

    # policy-specific hyperparameters
    parser_policy = argparse.ArgumentParser(
        description=description, epilog=example_usage)
    if args_alg.alg == "TD3":
        parser_policy = create_td3_parser(parser_policy)
    elif args_alg.alg == "SAC":
        parser_policy = create_sac_parser(parser_policy)
    elif args_alg.alg == "PPO":
        parser_policy = create_ppo_parser(parser_policy)

    # arguments for different model architectures
    parser_policy = create_feedforward_parser(parser_policy)
    if hierarchical:
        parser_policy = create_goal_conditioned_parser(parser_policy)
    if multiagent:
        parser_policy = create_multi_feedforward_parser(parser_policy)

    return parser_policy.parse_args(extras_alg, namespace=args_alg)


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
        '--save_replay_buffer', action='store_true',
        help='whether to save the data from the replay buffer, at the '
             'frequency that the model is saved. Only the most recent replay '
             'buffer is stored.')
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
        "--buffer_size",
        type=int,
        default=TD3_PARAMS["buffer_size"],
        help="the max number of transitions to store")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=TD3_PARAMS["batch_size"],
        help="the size of the batch for learning the policy")
    parser.add_argument(
        "--actor_lr",
        type=float,
        default=TD3_PARAMS["actor_lr"],
        help="the actor learning rate")
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=TD3_PARAMS["critic_lr"],
        help="the critic learning rate")
    parser.add_argument(
        "--tau",
        type=float,
        default=TD3_PARAMS["tau"],
        help="the soft update coefficient (keep old values, between 0 and 1)")
    parser.add_argument(
        "--gamma",
        type=float,
        default=TD3_PARAMS["gamma"],
        help="the discount rate")
    parser.add_argument(
        "--use_huber",
        action="store_true",
        help="specifies whether to use the huber distance function as the "
             "loss for the critic. If set to False, the mean-squared error "
             "metric is used instead")
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
        "--buffer_size",
        type=int,
        default=SAC_PARAMS["buffer_size"],
        help="the max number of transitions to store")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=SAC_PARAMS["batch_size"],
        help="the size of the batch for learning the policy")
    parser.add_argument(
        "--actor_lr",
        type=float,
        default=SAC_PARAMS["actor_lr"],
        help="the actor learning rate")
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=SAC_PARAMS["critic_lr"],
        help="the critic learning rate")
    parser.add_argument(
        "--tau",
        type=float,
        default=SAC_PARAMS["tau"],
        help="the soft update coefficient (keep old values, between 0 and 1)")
    parser.add_argument(
        "--gamma",
        type=float,
        default=SAC_PARAMS["gamma"],
        help="the discount rate")
    parser.add_argument(
        "--use_huber",
        action="store_true",
        help="specifies whether to use the huber distance function as the "
             "loss for the critic. If set to False, the mean-squared error "
             "metric is used instead")
    parser.add_argument(
        "--target_entropy",
        type=float,
        default=SAC_PARAMS["target_entropy"],
        help="target entropy used when learning the entropy coefficient. If "
             "set to None, a heuristic value is used.")

    return parser


def create_ppo_parser(parser):
    """Add the PPO hyperparameters to the parser."""
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=PPO_PARAMS["learning_rate"],
        help="the learning rate")
    parser.add_argument(
        "--n_minibatches",
        type=int,
        default=PPO_PARAMS["n_minibatches"],
        help="number of training minibatches per update")
    parser.add_argument(
        "--n_opt_epochs",
        type=int,
        default=PPO_PARAMS["n_opt_epochs"],
        help="number of training epochs per update procedure")
    parser.add_argument(
        "--gamma",
        type=float,
        default=PPO_PARAMS["gamma"],
        help="the discount factor")
    parser.add_argument(
        "--lam",
        type=float,
        default=PPO_PARAMS["lam"],
        help="factor for trade-off of bias vs variance for Generalized "
             "Advantage Estimator")
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=PPO_PARAMS["ent_coef"],
        help="entropy coefficient for the loss calculation")
    parser.add_argument(
        "--vf_coef",
        type=float,
        default=PPO_PARAMS["vf_coef"],
        help="value function coefficient for the loss calculation")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=PPO_PARAMS["max_grad_norm"],
        help="the maximum value for the gradient clipping")
    parser.add_argument(
        "--cliprange",
        type=float,
        default=PPO_PARAMS["cliprange"],
        help="clipping parameter, it can be a function")
    parser.add_argument(
        "--cliprange_vf",
        type=float,
        default=PPO_PARAMS["cliprange_vf"],
        help="clipping parameter for the value function, it can be a "
             "function. This is a parameter specific to the OpenAI "
             "implementation. If None is passed (default), then `cliprange` "
             "(that is used for the policy) will be used. IMPORTANT: this "
             "clipping depends on the reward scaling. To deactivate value "
             "function clipping (and recover the original PPO "
             "implementation), you have to pass a negative value (e.g. -1).")

    return parser


def create_feedforward_parser(parser):
    """Add the feedforward policy hyperparameters to the parser."""
    parser.add_argument(
        "--model_params:model_type",
        type=str,
        default=FEEDFORWARD_PARAMS["model_params"]["model_type"],
        help="the type of model to use. Must be one of {\"fcnet\", \"conv\"}.")
    parser.add_argument(
        "--model_params:layer_norm",
        action="store_true",
        help="enable layer normalisation")
    parser.add_argument(
        "--model_params:layers",
        type=int,
        nargs="+",
        help="the size of the neural network for the policy")
    parser.add_argument(
        "--model_params:ignore_flat_channels",
        type=int,
        nargs="+",
        help="specifies which channels of the observation to ignore")
    parser.add_argument(
        "--model_params:ignore_image",
        action="store_true",
        help="specifies whether the image in the observation "
             "should be ignored and removed")
    parser.add_argument(
        "--model_params:image_height",
        type=int,
        default=FEEDFORWARD_PARAMS["model_params"]["image_height"],
        help="the height of the image observation")
    parser.add_argument(
        "--model_params:image_width",
        type=int,
        default=FEEDFORWARD_PARAMS["model_params"]["image_width"],
        help="the width of the image observation")
    parser.add_argument(
        "--model_params:image_channels",
        type=int,
        default=FEEDFORWARD_PARAMS["model_params"]["image_channels"],
        help="the number of channels of the image observation")
    parser.add_argument(
        "--model_params:filters",
        type=int,
        nargs="+",
        help="specifies the convolutional filters per layer")
    parser.add_argument(
        "--model_params:kernel_sizes",
        type=int,
        nargs="+",
        help="specifies the convolutional kernel sizes per layer")
    parser.add_argument(
        "--model_params:strides",
        type=int,
        nargs="+",
        help="specifies the convolutional strides per layer")

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
        "--cooperative_gradients",
        action="store_true",
        help="whether to use the cooperative gradient update procedure for the"
             " higher-level policy. See: https://arxiv.org/abs/1912.02368v1")
    parser.add_argument(
        "--cg_weights",
        type=float,
        default=GOAL_CONDITIONED_PARAMS["cg_weights"],
        help="weights for the gradients of the loss of the lower-level "
             "policies with respect to the parameters of the higher-level "
             "policies. Only used if `cooperative_gradients` is set to True.")
    parser.add_argument(
        "--pretrain_worker",
        action="store_true",
        help="specifies whether you are pre-training the lower-level "
             "policies. Actions by the high-level policy are randomly sampled "
             "from its action space.")
    parser.add_argument(
        "--pretrain_path",
        type=str,
        default=GOAL_CONDITIONED_PARAMS["pretrain_path"],
        help="path to the pre-trained worker policy checkpoints")
    parser.add_argument(
        "--pretrain_ckpt",
        type=int,
        default=GOAL_CONDITIONED_PARAMS["pretrain_ckpt"],
        help="checkpoint number to use within the worker policy path. If set "
             "to None, the most recent checkpoint is used.")

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
