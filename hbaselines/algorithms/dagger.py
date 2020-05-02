"""Script contain the DAgger training algorithm.

See: TODO
"""
import tensorflow as tf
import numpy as np
import random
import os
from collections import deque
from copy import deepcopy

from hbaselines.algorithms.utils import is_feedforward_policy
from hbaselines.algorithms.utils import is_goal_conditioned_policy
from hbaselines.utils.tf_util import make_session
from hbaselines.utils.misc import ensure_dir
from hbaselines.utils.env_util import create_env


# =========================================================================== #
#                   Policy parameters for FeedForwardPolicy                   #
# =========================================================================== #

FEEDFORWARD_PARAMS = dict(
    # the max number of transitions to store
    buffer_size=200000,
    # the size of the batch for learning the policy
    batch_size=128,
    # the learning rate of the policy
    learning_rate=3e-4,
    # enable layer normalization
    layer_norm=False,
    # the size of the neural network for the policy
    layers=[256, 256],
    # the activation function to use in the neural network
    act_fun=tf.nn.relu,
    # specifies whether to use the huber distance function as the loss
    # function. If set to False, the mean-squared error metric is used instead
    use_huber=False,
    # specifies whether the policies are stochastic or deterministic
    stochastic=False,
)

# =========================================================================== #
#     Policy parameters for GoalConditionedPolicy (shared by TD3 and SAC)     #
# =========================================================================== #

GOAL_CONDITIONED_PARAMS = FEEDFORWARD_PARAMS.copy()
GOAL_CONDITIONED_PARAMS.update(dict(
    # number of levels within the hierarchy. Must be greater than 1. Two levels
    # correspond to a Manager/Worker paradigm.
    num_levels=2,
    # meta-policy action period
    meta_period=10,
    # the value that the intrinsic reward should be scaled by
    intrinsic_reward_scale=1,
    # specifies whether the goal issued by the higher-level policies is meant
    # to be a relative or absolute goal, i.e. specific state or change in state
    relative_goals=False,
))


class DAggerAlgorithm(object):
    """DAgger training algorithm.

    Attributes
    ----------
    policy : type [ hbaselines.base_policies.ImitationLearningPolicy ]
        the policy model to use
    env_name : str
        name of the environment. Affects the action bounds of the higher-level
        policies
    env : TODO
        TODO
    expert : str or None
        the path to the expert policy parameter that need to loaded into the
        environment. If set to None, the expert is assumed to be already loaded
    aggr_update_freq : TODO
        TODO
    aggr_update_steps : TODO
        TODO
    nb_train_steps : TODO
        TODO
    nb_rollout_steps : TODO
        TODO
    verbose :
        TODO
    policy_kwargs : TODO
        TODO
    """

    def __init__(self,
                 policy,
                 env,
                 render=False,
                 expert=None,
                 aggr_update_freq=10000,
                 aggr_update_steps=1000,
                 nb_train_steps=1,
                 nb_rollout_steps=1,
                 verbose=0,
                 policy_kwargs=None):
        """Instantiate the DAgger training algorithm.

        Parameters
        ----------
        policy : type [ hbaselines.base_policies.ImitationLearningPolicy ]
            the policy model to use
        env : gym.Env or str
            the environment to learn from (if registered in Gym, can be str)
        render : bool
            enable rendering of the training environment
        expert : str or None
            the path to the expert policy parameter that need to loaded into
            the environment. If set to None, the expert is assumed to be
            already loaded.
        aggr_update_freq : int
            TODO
        aggr_update_steps : int
            TODO
        nb_train_steps : int
            the number of training steps
        nb_rollout_steps : int
            the number of rollout steps
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        policy_kwargs : dict
            policy-specific hyperparameters
        """
        self.policy = policy
        self.env_name = deepcopy(env) if isinstance(env, str) \
            else env.__str__()
        self.env = create_env(
            env, render, shared=False, maddpg=False, evaluate=False)
        self.expert = expert
        self.aggr_update_freq = aggr_update_freq
        self.aggr_update_steps = aggr_update_steps
        self.nb_train_steps = nb_train_steps
        self.nb_rollout_steps = nb_rollout_steps
        self.verbose = verbose
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.context_space = getattr(self.env, "context_space", None)
        self.policy_kwargs = {'verbose': verbose}

        # add the default policy kwargs to the policy_kwargs term
        if is_feedforward_policy(policy):
            self.policy_kwargs.update(FEEDFORWARD_PARAMS.copy())
        elif is_goal_conditioned_policy(policy):
            self.policy_kwargs.update(GOAL_CONDITIONED_PARAMS.copy())
            self.policy_kwargs['env_name'] = self.env_name.__str__()

        self.policy_kwargs.update(policy_kwargs or {})

        # Compute the time horizon, which is used to check if an environment
        # terminated early and used to compute the done mask as per TD3
        # implementation (see appendix A of their paper). If the horizon cannot
        # be found, it is assumed to be 500 (default value for most gym
        # environments).
        if hasattr(self.env, "horizon"):
            self.horizon = self.env.horizon
        elif hasattr(self.env, "_max_episode_steps"):
            self.horizon = self.env._max_episode_steps
        elif hasattr(self.env, "env_params"):
            # for Flow environments
            self.horizon = self.env.env_params.horizon
        else:
            raise ValueError("Horizon attribute not found.")

        # Load the export policy to the environment, if needed.
        if expert is not None:
            self.env.load_expert(expert)

        # init
        self.graph = None
        self.policy_tf = None
        self.sess = None
        self.summary = None
        self.obs = None
        self.episode_step = 0
        self.total_steps = 0
        self.epoch = 0
        self.rew_history = deque(maxlen=100)
        self.rew_ph = None
        self.rew_history_ph = None
        self.saver = None

        # Create the model variables and operations.
        self.trainable_vars = self.setup_model()

    def setup_model(self):
        """Create the graph, session, policy, and summary objects."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Create the tensorflow session.
            self.sess = make_session(num_cpu=3, graph=self.graph)

            # Create the policy.
            self.policy_tf = self.policy(
                self.sess,
                self.observation_space,
                self.action_space,
                self.context_space,
                **self.policy_kwargs
            )

            # for tensorboard logging
            with tf.compat.v1.variable_scope("Train"):
                self.rew_ph = tf.compat.v1.placeholder(tf.float32)
                self.rew_history_ph = tf.compat.v1.placeholder(tf.float32)

            # Add tensorboard scalars for the return, return history, and
            # success rate.
            tf.compat.v1.summary.scalar("Train/return", self.rew_ph)
            tf.compat.v1.summary.scalar("Train/return_history",
                                        self.rew_history_ph)

            # Create the tensorboard summary.
            self.summary = tf.compat.v1.summary.merge_all()

            # Initialize the model parameters and optimizers.
            with self.sess.as_default():
                self.sess.run(tf.compat.v1.global_variables_initializer())
                self.policy_tf.initialize()

            return tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

    def learn(self,
              total_timesteps,
              log_dir,
              seed,
              log_interval,
              save_interval,
              initial_sample_steps):
        """Perform the complete training operation.

        Parameters
        ----------
        total_timesteps : int
            the total number of samples to train on
        log_dir : str
            the directory where the training and evaluation statistics, as well
            as the tensorboard log, should be stored
        seed : int or None
            the initial seed for training, if None: keep current seed
        log_interval : int
            the number of training steps before logging training results
        save_interval : int
            number of simulation steps in the training environment before the
            model is saved
        initial_sample_steps : int
            TODO
        """
        # Create a saver object.
        self.saver = tf.compat.v1.train.Saver(
            self.trainable_vars,
            max_to_keep=total_timesteps // save_interval
        )

        # Make sure that the log directory exists, and if not, make it.
        ensure_dir(log_dir)
        ensure_dir(os.path.join(log_dir, "checkpoints"))

        # Create a tensorboard object for logging.
        save_path = os.path.join(log_dir, "tb_log")
        writer = tf.compat.v1.summary.FileWriter(save_path)

        # file path for training and evaluation results
        train_filepath = os.path.join(log_dir, "train.csv")

        # Setup the seed value.
        random.seed(seed)
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

        if self.verbose >= 2:
            print('Using agent with the following configuration:')
            print(str(self.__dict__.items()))

        with self.sess.as_default(), self.graph.as_default():
            # Prepare everything.
            self.obs = self.env.reset()

            # Collect the initial samples by the expert.
            print("Collecting initial expert samples...")
            self._collect_samples(total_timesteps,
                                  run_steps=initial_sample_steps)
            print("Done!")

            # Reset total statistics variables.
            self.total_steps = 0
            self.rew_history = deque(maxlen=100)

            for i in range(total_timesteps):
                # Perform the training operation.
                self._train()

                # Collect new data to add to the replay buffer.
                if i > 0 and i % self.aggr_update_freq == 0:
                    pass

                # Run and store summary.
                if i > 0 and i % log_interval == 0:
                    pass

                # Save a checkpoint of the model.
                if i > 0 and i % save_interval == 0:
                    pass

    def _collect_samples(self, total_timesteps, run_steps=None):
        """Perform the sample collection operation.

        This method is responsible for executing rollouts for a number of steps
        before training is executed. The data from the rollouts is stored in
        the policy's replay buffer(s).

        Parameters
        ----------
        total_timesteps : int
            the total number of samples to train on. Used by the fingerprint
            element
        run_steps : int, optional
            number of steps to collect samples from. If not provided, the value
            defaults to `self.nb_rollout_steps`.
        """
        pass

    def _train(self):
        """Perform the training operation."""
        for t_train in range(self.nb_train_steps):
            _ = self.policy_tf.update()
