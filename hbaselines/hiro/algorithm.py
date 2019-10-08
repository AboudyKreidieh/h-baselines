"""Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.

This algorithm also contains modifications to support contextual environments
and hierarchical policies.

See: https://arxiv.org/pdf/1802.09477.pdf
"""
import os
import time
from collections import deque
import csv
import random
from copy import deepcopy
import gym
from gym.spaces import Box
import numpy as np
import tensorflow as tf

from hbaselines.hiro.tf_util import make_session
from hbaselines.hiro.policy import FeedForwardPolicy, GoalDirectedPolicy
from hbaselines.common.utils import ensure_dir
try:
    from flow.utils.registry import make_create_env
except (ImportError, ModuleNotFoundError):
    pass
from hbaselines.envs.efficient_hrl.envs import AntMaze, AntFall, AntPush
from hbaselines.envs.hac.envs import UR5, Pendulum


# =========================================================================== #
#                   Policy parameters for FeedForwardPolicy                   #
# =========================================================================== #

FEEDFORWARD_POLICY_KWARGS = dict(
    # the max number of transitions to store
    buffer_size=200000,
    # the size of the batch for learning the policy
    batch_size=128,
    # the actor learning rate
    actor_lr=3e-4,
    # the critic learning rate
    critic_lr=3e-4,
    # the soft update coefficient (keep old values, between 0 and 1)
    tau=0.005,
    # the discount rate
    gamma=0.99,
    # scaling term to the range of the action space, that is subsequently used
    # as the standard deviation of Gaussian noise added to the action if
    # `apply_noise` is set to True in `get_action`
    noise=0.1,
    # standard deviation term to the noise from the output of the target actor
    # policy. See TD3 paper for more.
    target_policy_noise=0.2,
    # clipping term for the noise injected in the target actor policy
    target_noise_clip=0.5,
    # enable layer normalisation
    layer_norm=False,
    # the size of the neural network for the policy
    layers=None,
    # the activation function to use in the neural network
    act_fun=tf.nn.relu,
    # specifies whether to use the huber distance function as the loss for the
    # critic. If set to False, the mean-squared error metric is used instead
    use_huber=False,
)


# =========================================================================== #
#                  Policy parameters for GoalDirectedPolicy                   #
# =========================================================================== #

GOAL_DIRECTED_POLICY_KWARGS = FEEDFORWARD_POLICY_KWARGS.copy()
GOAL_DIRECTED_POLICY_KWARGS.update(dict(
    # manger action period
    meta_period=10,
    # specifies whether the goal issued by the Manager is meant to be a
    # relative or absolute goal, i.e. specific state or change in state
    relative_goals=False,
    # whether to use off-policy corrections during the update procedure. See:
    # https://arxiv.org/abs/1805.08296
    off_policy_corrections=False,
    # specifies whether to add a time-dependent fingerprint to the observations
    use_fingerprints=False,
    # the low and high values for each fingerprint element, if they are being
    # used
    fingerprint_range=([0, 0], [5, 5]),
    # specifies whether to use centralized value functions for the Manager and
    # Worker critic functions
    centralized_value_functions=False,
    # whether to connect the graph between the manager and worker
    connected_gradients=False,
))


def as_scalar(scalar):
    """Check and return the input if it is a scalar.

    If it is not scalar, raise a ValueError.

    Parameters
    ----------
    scalar : Any
        the object to check

    Returns
    -------
    float
        the scalar if x is a scalar
    """
    if isinstance(scalar, np.ndarray):
        assert scalar.size == 1
        return scalar[0]
    elif np.isscalar(scalar):
        return scalar
    else:
        raise ValueError('expected scalar, got %s' % scalar)


class TD3(object):
    """Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.

    See: https://arxiv.org/pdf/1802.09477.pdf

    Attributes
    ----------
    policy : type [ hbaselines.hiro.policy.ActorCriticPolicy ]
        the policy model to use
    env_name : str
        name of the environment. Affects the action bounds of the Manager
        policies
    env : gym.Env or str
        the environment to learn from (if registered in Gym, can be str)
    num_cpus : int
        number of CPUs to be used during the training procedure
    eval_env : gym.Env or str
        the environment to evaluate from (if registered in Gym, can be str)
    nb_train_steps : int
        the number of training steps
    nb_rollout_steps : int
        the number of rollout steps
    nb_eval_episodes : int
        the number of evaluation episodes
    actor_update_freq : int
        number of training steps per actor policy update step. The critic
        policy is updated every training step.
    meta_update_freq : int
        number of training steps per meta policy update step. The actor policy
        of the meta-policy is further updated at the frequency provided by the
        actor_update_freq variable. Note that this value is only relevant when
        using the GoalDirectedPolicy policy.
    reward_scale : float
        the value the reward should be scaled by
    render : bool
        enable rendering of the training environment
    render_eval : bool
        enable rendering of the evaluation environment
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    action_space : gym.spaces.*
        the action space of the training environment
    observation_space : gym.spaces.*
        the observation space of the training environment
    context_space : gym.spaces.*
        the context space of the training environment (i.e. the same of the
        desired environmental goal)
    policy_kwargs : dict
        policy-specific hyperparameters
    horizon : int
        time horizon, which is used to check if an environment terminated early
        and used to compute the done mask as per TD3 implementation (see
        appendix A of their paper). If the horizon cannot be found, it is
        assumed to be 500 (default value for most gym environments).
    graph : tf.Graph
        the current tensorflow graph
    policy_tf : hbaselines.hiro.policy.ActorCriticPolicy
        the policy object
    sess : tf.compat.v1.Session
        the current tensorflow session
    summary : tf.Summary
        tensorboard summary object
    obs : array_like
        the most recent training observation
    episode_step : int
        the number of steps since the most recent rollout began
    episodes : int
        the total number of rollouts performed since training began
    total_steps : int
        the total number of steps that have been executed since training began
    epoch_episode_rewards : list of float
        a list of cumulative rollout rewards from the most recent training
        iterations
    epoch_episode_steps : list of int
        a list of rollout lengths from the most recent training iterations
    epoch_actor_losses : list of float
        the actor loss values from each SGD step in the most recent training
        iteration
    epoch_critic_losses : list of float
        the critic loss values from each SGD step in the most recent training
        iteration
    epoch_actions : list of array_like
        a list of the actions that were performed during the most recent
        training iteration
    epoch_q1s : list of float
        a list of the Q1 values that were calculated during the most recent
        training iteration
    epoch_q2s : list of float
        a list of the Q2 values that were calculated during the most recent
    epoch_episodes : int
        the total number of rollouts performed since the most recent training
        iteration began
    epoch : int
        the total number of training iterations
    episode_rewards_history : list of float
        the cumulative return from the last 100 training episodes
    episode_reward : float
        the cumulative reward since the most reward began
    saver : tf.compat.v1.train.Saver
        tensorflow saver object
    rew_ph : tf.compat.v1.placeholder
        a placeholder for the average training return for the last epoch. Used
        for logging purposes.
    rew_history_ph : tf.compat.v1.placeholder
        a placeholder for the average training return for the last 100
        episodes. Used for logging purposes.
    eval_rew_ph : tf.compat.v1.placeholder
        placeholder for the average evaluation return from the last time
        evaluations occurred. Used for logging purposes.
    eval_success_ph : tf.compat.v1.placeholder
        placeholder for the average evaluation success rate from the last time
        evaluations occurred. Used for logging purposes.
    """

    def __init__(self,
                 policy,
                 env,
                 num_cpus=1,
                 eval_env=None,
                 nb_train_steps=1,
                 nb_rollout_steps=1,
                 nb_eval_episodes=50,
                 actor_update_freq=2,
                 meta_update_freq=10,
                 reward_scale=1.,
                 render=False,
                 render_eval=False,
                 verbose=0,
                 policy_kwargs=None,
                 _init_setup_model=True):
        """Instantiate the algorithm object.

        Parameters
        ----------
        policy : type [ hbaselines.hiro.policy.ActorCriticPolicy ]
            the policy model to use
        env : gym.Env or str
            the environment to learn from (if registered in Gym, can be str)
        num_cpus : int, optional
            number of CPUs to be used during the training procedure. Defaults
            to 1.
        eval_env : gym.Env or str
            the environment to evaluate from (if registered in Gym, can be str)
        nb_train_steps : int
            the number of training steps
        nb_rollout_steps : int
            the number of rollout steps
        nb_eval_episodes : int
            the number of evaluation episodes
        actor_update_freq : int
            number of training steps per actor policy update step. The critic
            policy is updated every training step.
        meta_update_freq : int
            number of training steps per meta policy update step. The actor
            policy of the meta-policy is further updated at the frequency
            provided by the actor_update_freq variable. Note that this value is
            only relevant when using the GoalDirectedPolicy policy.
        reward_scale : float
            the value the reward should be scaled by
        render : bool
            enable rendering of the training environment
        render_eval : bool
            enable rendering of the evaluation environment
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        policy_kwargs : dict
            policy-specific hyperparameters
        _init_setup_model : bool
            Whether or not to build the network at the creation of the instance
        """
        self.policy = policy
        self.env_name = deepcopy(env)
        self.env = self._create_env(env, evaluate=False)
        self.num_cpus = num_cpus
        self.eval_env = self._create_env(eval_env, evaluate=True)
        self.nb_train_steps = nb_train_steps
        self.nb_rollout_steps = nb_rollout_steps
        self.nb_eval_episodes = nb_eval_episodes
        self.actor_update_freq = actor_update_freq
        self.meta_update_freq = meta_update_freq
        self.reward_scale = reward_scale
        self.render = render
        self.render_eval = render_eval
        self.verbose = verbose
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.context_space = getattr(self.env, "context_space", None)

        # add the default policy kwargs to the policy_kwargs term
        if policy == FeedForwardPolicy:
            self.policy_kwargs = FEEDFORWARD_POLICY_KWARGS.copy()
        elif policy == GoalDirectedPolicy:
            self.policy_kwargs = GOAL_DIRECTED_POLICY_KWARGS.copy()
            self.policy_kwargs['env_name'] = self.env_name.__str__()
        else:
            self.policy_kwargs = {}

        self.policy_kwargs.update(policy_kwargs or {})
        self.policy_kwargs['verbose'] = verbose

        # Compute the time horizon, which is used to check if an environment
        # terminated early and used to compute the done mask as per TD3
        # implementation (see appendix A of their paper). If the horizon cannot
        # be found, it is assumed to be 500 (default value for most gym
        # environments).
        if hasattr(self.env, "horizon"):
            self.horizon = self.env.horizon
        else:
            print("Warning: self.env.horizon not found. Setting self.horizon "
                  "in the algorithm class to 500.")
            self.horizon = 500

        # init
        self.graph = None
        self.policy_tf = None
        self.sess = None
        self.summary = None
        self.obs = None
        self.episode_step = None
        self.episodes = None
        self.total_steps = None
        self.epoch_episode_rewards = None
        self.epoch_episode_steps = None
        self.epoch_actor_losses = None
        self.epoch_critic_losses = None
        self.epoch_actions = None
        self.epoch_q1s = None
        self.epoch_q2s = None
        self.epoch_episodes = None
        self.epoch = None
        self.episode_rewards_history = None
        self.episode_reward = None
        self.rew_ph = None
        self.rew_history_ph = None
        self.eval_rew_ph = None
        self.eval_success_ph = None

        # Append the fingerprint dimension to the observation dimension, if
        # needed.
        if self.policy_kwargs.get("use_fingerprints", False):
            fingerprint_range = self.policy_kwargs["fingerprint_range"]
            low = np.concatenate(
                (self.observation_space.low, fingerprint_range[0]))
            high = np.concatenate(
                (self.observation_space.high, fingerprint_range[1]))
            self.observation_space = Box(low=low, high=high)

        if _init_setup_model:
            # Create the model variables and operations.
            trainable_vars = self.setup_model()

            # Create a saver object.
            self.saver = tf.compat.v1.train.Saver(trainable_vars)

    @staticmethod
    def _create_env(env, evaluate=False):
        """Return, and potentially create, the environment.

        Parameters
        ----------
        env : str or gym.Env
            the environment, or the name of a registered environment.
        evaluate : bool, optional
            specifies whether this is a training or evaluation environment

        Returns
        -------
        gym.Env or list of gym.Env
            gym-compatible environment(s)
        """
        if env == "AntMaze":
            if evaluate:
                env = [AntMaze(use_contexts=True, context_range=[16, 0]),
                       AntMaze(use_contexts=True, context_range=[16, 16]),
                       AntMaze(use_contexts=True, context_range=[0, 16])]
            else:
                env = AntMaze(use_contexts=True,
                              random_contexts=True,
                              context_range=[(-4, 20), (-4, 20)])

        elif env == "AntPush":
            if evaluate:
                env = AntPush(use_contexts=True, context_range=[0, 19])
            else:
                env = AntPush(use_contexts=True, context_range=[0, 19])
                # env = AntPush(use_contexts=True,
                #               random_contexts=True,
                #               context_range=[(-16, 16), (-4, 20)])

        elif env == "AntFall":
            if evaluate:
                env = AntFall(use_contexts=True, context_range=[0, 27, 4.5])
            else:
                env = AntFall(use_contexts=True, context_range=[0, 27, 4.5])
                # env = AntFall(use_contexts=True,
                #               random_contexts=True,
                #               context_range=[(-4, 12), (-4, 28), (0, 5)])

        elif env == "UR5":
            if evaluate:
                env = UR5(use_contexts=True,
                          random_contexts=True,
                          context_range=[(-np.pi, np.pi),
                                         (-np.pi / 4, 0),
                                         (-np.pi / 4, np.pi / 4)])
            else:
                env = UR5(use_contexts=True,
                          random_contexts=True,
                          context_range=[(-np.pi, np.pi),
                                         (-np.pi / 4, 0),
                                         (-np.pi / 4, np.pi / 4)])

        elif env == "Pendulum":
            if evaluate:
                env = Pendulum(use_contexts=True, context_range=[0, 0])
            else:
                env = Pendulum(use_contexts=True,
                               random_contexts=True,
                               context_range=[
                                   (np.deg2rad(-16), np.deg2rad(16)),
                                   (-0.6, 0.6)])

        elif env in ["figureeight0", "figureeight1", "figureeight2", "merge0",
                     "merge1", "merge2", "bottleneck0", "bottleneck1",
                     "bottleneck2", "grid0", "grid1"]:
            # Import the benchmark and fetch its flow_params
            benchmark = __import__("flow.benchmarks.{}".format(env),
                                   fromlist=["flow_params"])
            flow_params = benchmark.flow_params

            # Get the env name and a creator for the environment.
            create_env, _ = make_create_env(flow_params, version=0)

            # Create the environment.
            env = create_env()

        elif isinstance(env, str):
            # This is assuming the environment is registered with OpenAI gym.
            env = gym.make(env)

        # Reset the environment.
        if env is not None:
            if isinstance(env, list):
                for next_env in env:
                    next_env.reset()
            else:
                env.reset()

        return env

    def setup_model(self):
        """Create the graph, session, policy, and summary objects."""
        # determine whether the action space is continuous
        assert isinstance(self.action_space, Box), \
            "Error: TD3 cannot output a {} action space, only " \
            "spaces.Box is supported.".format(self.action_space)

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Create the tensorflow session.
            # self.sess = make_session(num_cpu=self.num_cpus, graph=self.graph)
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
            with tf.compat.v1.variable_scope("Evaluate"):
                self.eval_rew_ph = tf.compat.v1.placeholder(tf.float32)
                self.eval_success_ph = tf.compat.v1.placeholder(tf.float32)

            # Add tensorboard scalars for the return, return history, and
            # success rate.
            tf.compat.v1.summary.scalar("Train/return", self.rew_ph)
            tf.compat.v1.summary.scalar("Train/return_history",
                                        self.rew_history_ph)
            # FIXME
            # if self.eval_env is not None:
            #     eval_success_ph = self.eval_success_ph
            #     tf.compat.v1.summary.scalar("Evaluate/return",
            #                                 self.eval_rew_ph)
            #     tf.compat.v1.summary.scalar("Evaluate/success_rate",
            #                                 eval_success_ph)

            # Create the tensorboard summary.
            self.summary = tf.compat.v1.summary.merge_all()

            # Initialize the model parameters and optimizers.
            with self.sess.as_default():
                self.sess.run(tf.global_variables_initializer())
                self.policy_tf.initialize()

            return tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

    def _policy(self,
                obs,
                apply_noise=True,
                compute_q=True,
                random_actions=False,
                **kwargs):
        """Get the actions and critic output, from a given observation.

        Parameters
        ----------
        obs : list of float or list of int
            the observation
        apply_noise : bool
            enable the noise
        compute_q : bool
            compute the critic output
        random_actions : bool
            if set to True, actions are sampled randomly from the action space
            instead of being computed by the policy. This is used for
            exploration purposes.

        Returns
        -------
        list of float
            the action value
        float
            the critic value
        """
        obs = np.array(obs).reshape((-1,) + self.observation_space.shape)

        action = self.policy_tf.get_action(
            obs,
            apply_noise=apply_noise,
            random_actions=random_actions,
            total_steps=self.total_steps,
            context_obs=kwargs["context"])

        q_value = self.policy_tf.value(
            obs, action, context_obs=kwargs["context"]) if compute_q else None

        return action.flatten(), q_value

    def _store_transition(self, obs0, action, reward, obs1, terminal1):
        """Store a transition in the replay buffer.

        Parameters
        ----------
        obs0 : list of float or list of int
            the last observation
        action : list of float or np.ndarray
            the action
        reward : float
            the reward
        obs1 : list fo float or list of int
            the current observation
        terminal1 : bool
            is the episode done
        """
        # Get the contextual term.
        context = getattr(self.env, "current_context", None)

        # Scale the rewards by the provided term.
        reward *= self.reward_scale

        self.policy_tf.store_transition(obs0, action, reward, obs1, terminal1,
                                        context_obs0=context,
                                        context_obs1=context)

    def learn(self,
              total_timesteps,
              log_dir=None,
              seed=None,
              log_interval=100,
              eval_interval=5e4,
              start_timesteps=10000):
        """Return a trained model.

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
        eval_interval : int
            number of simulation steps in the training environment before an
            evaluation is performed
        start_timesteps : int, optional
            number of timesteps that the policy is run before training to
            initialize the replay buffer with samples
        """
        # Make sure that the log directory exists, and if not, make it.
        ensure_dir(log_dir)
        ensure_dir(os.path.join(log_dir, "checkpoints"))

        # Create a tensorboard object for logging.
        save_path = os.path.join(log_dir, "tb_log")
        writer = tf.compat.v1.summary.FileWriter(save_path)

        # file path for training and evaluation results
        train_filepath = os.path.join(log_dir, "train.csv")
        eval_filepath = os.path.join(log_dir, "eval.csv")

        # Setup the seed value.
        random.seed(seed)
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

        if self.verbose >= 2:
            print('Using agent with the following configuration:')
            print(str(self.__dict__.items()))

        # Initialize class variables.
        steps_incr = 0
        self.episode_reward = 0
        self.episode_step = 0
        self.episodes = 0
        self.total_steps = 0
        self.epoch = 0
        self.episode_rewards_history = deque(maxlen=100)

        with self.sess.as_default(), self.graph.as_default():
            # Prepare everything.
            self.obs = self.env.reset()
            # Add the fingerprint term, if needed.
            self.obs = self._add_fingerprint(
                self.obs, self.total_steps, total_timesteps)
            start_time = time.time()

            # Reset epoch-specific variables. FIXME: hacky
            self.epoch_episodes = 0
            self.epoch_actions = []
            self.epoch_q1s = []
            self.epoch_q2s = []
            self.epoch_actor_losses = []
            self.epoch_critic_losses = []
            self.epoch_episode_rewards = []
            self.epoch_episode_steps = []
            # Perform rollouts.
            print("Collecting pre-samples...")
            self._collect_samples(total_timesteps,
                                  run_steps=start_timesteps,
                                  random_actions=True)
            print("Done!")
            self.episode_reward = 0
            self.episode_step = 0
            self.episodes = 0
            self.total_steps = 0
            self.epoch = 0
            self.episode_rewards_history = deque(maxlen=100)

            while True:
                # Reset epoch-specific variables.
                self.epoch_episodes = 0
                self.epoch_actions = []
                self.epoch_q1s = []
                self.epoch_q2s = []
                self.epoch_actor_losses = []
                self.epoch_critic_losses = []
                self.epoch_episode_rewards = []
                self.epoch_episode_steps = []

                for _ in range(log_interval):
                    # If the requirement number of time steps has been met,
                    # terminate training.
                    if self.total_steps >= total_timesteps:
                        return

                    # Perform rollouts.
                    self._collect_samples(total_timesteps)

                    # Train.
                    self._train()

                # Log statistics.
                self._log_training(train_filepath, start_time)

                # Evaluate.
                if self.eval_env is not None and \
                        (self.total_steps - steps_incr) >= eval_interval:
                    steps_incr += eval_interval

                    # Run the evaluation operations over the evaluation env(s).
                    # Note that multiple evaluation envs can be provided.
                    if isinstance(self.eval_env, list):
                        eval_rewards = []
                        eval_successes = []
                        eval_info = []
                        for env in self.eval_env:
                            rew, suc, inf = \
                                self._evaluate(total_timesteps, env)
                            eval_rewards.append(rew)
                            eval_successes.append(suc)
                            eval_info.append(inf)
                    else:
                        eval_rewards, eval_successes, eval_info = \
                            self._evaluate(total_timesteps, self.eval_env)

                    # Log the evaluation statistics.
                    self._log_eval(
                        eval_filepath,
                        start_time,
                        eval_rewards,
                        eval_successes,
                        eval_info,
                    )

                # Run and store summary.
                if writer is not None:
                    td_map = self.policy_tf.get_td_map()
                    # Check if td_map is empty.
                    if td_map:
                        td_map.update({
                            self.rew_ph: np.mean(self.epoch_episode_rewards),
                            self.rew_history_ph: np.mean(
                                self.episode_rewards_history),
                        })
                        summary = self.sess.run(self.summary, td_map)
                        writer.add_summary(summary, self.total_steps)

                # Save a checkpoint of the model.
                self.save(os.path.join(log_dir, "checkpoints/itr"))

                # Update the epoch count.
                self.epoch += 1

    def save(self, save_path):
        """Save the parameters of a tensorflow model.

        Parameters
        ----------
        save_path : str
            Prefix of filenames created for the checkpoint
        """
        self.saver.save(self.sess, save_path, global_step=self.total_steps)

    def load(self, load_path):
        """Load model parameters from a checkpoint.

        Parameters
        ----------
        load_path : str
            location of the checkpoint
        """
        self.saver.restore(self.sess, load_path)

    def _collect_samples(self,
                         total_timesteps,
                         run_steps=None,
                         random_actions=False):
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
        random_actions : bool
            if set to True, actions are sampled randomly from the action space
            instead of being computed by the policy. This is used for
            exploration purposes.
        """
        for _ in range(run_steps or self.nb_rollout_steps):
            # Predict next action. Use random actions when initializing the
            # replay buffer.
            action, (q1_value, q2_value) = self._policy(
                self.obs,
                apply_noise=True,
                random_actions=random_actions,
                compute_q=True,
                context=[getattr(self.env, "current_context", None)])
            assert action.shape == self.env.action_space.shape

            # Execute next action.
            new_obs, reward, done, info = self.env.step(action)

            # Visualize the current step.
            if self.render:
                self.env.render()

            # Add the fingerprint term, if needed. When collecting the initial
            # random actions, we assume the fingerprint does not change from
            # its initial value.
            new_obs = self._add_fingerprint(
                new_obs,
                0 if random_actions else self.total_steps,
                total_timesteps)

            # Store a transition in the replay buffer. The terminal flag is
            # chosen to match the TD3 implementation (see Appendix 1 of their
            # paper).
            self._store_transition(
                obs0=self.obs,
                action=action,
                reward=reward,
                obs1=new_obs,
                terminal1=done and self.episode_step < self.horizon - 1
            )

            # Book-keeping.
            self.total_steps += 1
            self.episode_reward += reward
            self.episode_step += 1
            self.epoch_actions.append(action)
            self.epoch_q1s.append(q1_value)
            self.epoch_q2s.append(q2_value)

            # Update the current observation.
            self.obs = new_obs.copy()

            if done:
                # Episode done.
                self.epoch_episode_rewards.append(self.episode_reward)
                self.episode_rewards_history.append(self.episode_reward)
                self.epoch_episode_steps.append(self.episode_step)
                self.episode_reward = 0.
                self.episode_step = 0
                self.epoch_episodes += 1
                self.episodes += 1

                # Reset the environment.
                self.obs = self.env.reset()

                # Add the fingerprint term, if needed.
                self.obs = self._add_fingerprint(
                    self.obs, self.total_steps, total_timesteps)

    def _train(self):
        """Perform the training operation.

        Through this method, the actor and critic networks are updated within
        the policy, and the summary information is logged to tensorboard.
        """
        for t_train in range(self.nb_train_steps):
            if self.policy == GoalDirectedPolicy:
                # specifies whether to update the meta actor and critic
                # policies based on the meta and actor update frequencies
                kwargs = {
                    "update_meta":
                        (self.total_steps + t_train)
                        % self.meta_update_freq == 0,
                    "update_meta_actor":
                        (self.total_steps + t_train)
                        % (self.meta_update_freq * self.actor_update_freq) == 0
                }
            else:
                kwargs = {}

            # specifies whether to update the actor policy, base on the actor
            # update frequency
            update = (self.total_steps + t_train) % self.actor_update_freq == 0

            # Run a step of training from batch.
            critic_loss, actor_loss = self.policy_tf.update(
                update_actor=update, **kwargs)

            # Add actor and critic loss information for logging purposes.
            self.epoch_critic_losses.append(critic_loss)
            self.epoch_actor_losses.append(actor_loss)

    def _evaluate(self, total_timesteps, env):
        """Perform the evaluation operation.

        This method runs the evaluation environment for a number of episodes
        and returns the cumulative rewards and successes from each environment.

        Parameters
        ----------
        total_timesteps : int
            the total number of samples to train on
        env : gym.Env
            the evaluation environment that the policy is meant to be tested on

        Returns
        -------
        list of float
            the list of cumulative rewards from every episode in the evaluation
            phase
        list of bool
            a list of boolean terms representing if each episode ended in
            success or not. If the list is empty, then the environment did not
            output successes or failures, and the success rate will be set to
            zero.
        dict
            additional information that is meant to be logged
        """
        num_steps = deepcopy(self.total_steps)
        eval_episode_rewards = []
        eval_episode_successes = []
        ret_info = {'initial': [], 'final': [], 'average': []}

        if self.verbose >= 1:
            for _ in range(3):
                print("-------------------")
            print("Running evaluation for {} episodes:".format(
                self.nb_eval_episodes))

        for i in range(self.nb_eval_episodes):
            # Reset the environment.
            eval_obs = env.reset()

            # Add the fingerprint term, if needed.
            eval_obs = self._add_fingerprint(
                eval_obs, self.total_steps, total_timesteps)

            # Reset rollout-specific variables.
            eval_episode_reward = 0.
            eval_episode_step = 0

            rets = np.array([])
            while True:
                eval_action, _ = self._policy(
                    eval_obs,
                    apply_noise=False,
                    random_actions=False,
                    compute_q=False,
                    context=[getattr(env, "current_context", None)])

                obs, eval_r, done, info = env.step(eval_action)

                if self.render_eval:
                    env.render()

                # Add the distance to this list for logging purposes (applies
                # only to the Ant* environments).
                if hasattr(env, "current_context"):
                    context_obs = getattr(env, "current_context")
                    reward_fn = getattr(env, "contextual_reward")
                    rets = np.append(
                        rets, reward_fn(eval_obs, context_obs, obs))

                # Update the previous step observation.
                eval_obs = obs.copy()

                # Add the fingerprint term, if needed.
                eval_obs = self._add_fingerprint(
                    eval_obs, self.total_steps, total_timesteps)

                # Increment the reward and step count.
                num_steps += 1
                eval_episode_reward += eval_r
                eval_episode_step += 1

                if done:
                    eval_episode_rewards.append(eval_episode_reward)
                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        eval_episode_successes.append(float(maybe_is_success))

                    if self.verbose >= 1:
                        if rets.shape[0] > 0:
                            print("%d/%d: initial: %.3f, final: %.3f, average:"
                                  " %.3f, success: %d"
                                  % (i + 1, self.nb_eval_episodes, rets[0],
                                     rets[-1], float(rets.mean()),
                                     int(info.get('is_success'))))
                        else:
                            print("%d/%d" % (i + 1, self.nb_eval_episodes))

                    if hasattr(env, "current_context"):
                        ret_info['initial'].append(rets[0])
                        ret_info['final'].append(rets[-1])
                        ret_info['average'].append(float(rets.mean()))

                    # Exit the loop.
                    break

        if self.verbose >= 1:
            print("Done.")
            print("Average return: {}".format(np.mean(eval_episode_rewards)))
            if len(eval_episode_successes) > 0:
                print("Success rate: {}".format(
                    np.mean(eval_episode_successes)))
            for _ in range(3):
                print("-------------------")
            print("")

        # get the average of the reward information
        ret_info['initial'] = np.mean(ret_info['initial'])
        ret_info['final'] = np.mean(ret_info['final'])
        ret_info['average'] = np.mean(ret_info['average'])

        return eval_episode_rewards, eval_episode_successes, ret_info

    def _add_fingerprint(self, obs, steps, total_steps):
        """Add a fingerprint element to the observation.

        This should be done when setting "use_fingerprints" in policy_kwargs to
        True. The new observation looks as follows:

                  ---------------------------------------------------
        new_obs = || obs || 5 * frac_steps || 5 * (1 - frac_steps) ||
                  ---------------------------------------------------

        where frac_steps is the fraction of the total requested number of
        training steps that have been performed. Note that the "5" term is a
        fixed hyperparameter, and can be changed based on its effect on
        training performance.

        If "use_fingerprints" is set to False in policy_kwargs, or simply not
        specified, this method returns the current observation without the
        fingerprint term.

        Parameters
        ----------
        obs : array_like
            the current observation without the fingerprint element
        steps : int
            the total number of steps that have been performed

        Returns
        -------
        array_like
            the observation with the fingerprint element
        """
        # if the fingerprint element should not be added, simply return the
        # current observation.
        if not self.policy_kwargs.get("use_fingerprints", False):
            return obs

        # compute the fingerprint term
        frac_steps = float(steps) / float(total_steps)
        fp = [5 * frac_steps, 5 * (1 - frac_steps)]

        # append the fingerprint term to the current observation
        new_obs = np.concatenate((obs, fp), axis=0)

        return new_obs

    def _log_training(self, file_path, start_time):
        """Log training statistics.

        Parameters
        ----------
        file_path : str
            the list of cumulative rewards from every episode in the evaluation
            phase
        start_time : float
            the time when training began. This is used to print the total
            training time.
        """
        # Log statistics.
        duration = time.time() - start_time
        stats = self.policy_tf.get_stats()
        combined_stats = stats.copy()
        combined_stats['rollout/return'] = np.mean(self.epoch_episode_rewards)
        combined_stats['rollout/return_history'] = np.mean(
            self.episode_rewards_history)
        combined_stats['rollout/episode_steps'] = np.mean(
            self.epoch_episode_steps)
        combined_stats['rollout/actions_mean'] = np.mean(self.epoch_actions)
        combined_stats['rollout/Q1_mean'] = np.mean(self.epoch_q1s)
        combined_stats['rollout/Q2_mean'] = np.mean(self.epoch_q2s)
        combined_stats['train/loss_actor'] = np.mean(self.epoch_actor_losses)
        combined_stats['train/loss_critic'] = np.mean(self.epoch_critic_losses)
        combined_stats['total/duration'] = duration
        combined_stats['total/steps_per_second'] = self.total_steps / duration
        combined_stats['total/episodes'] = self.episodes
        combined_stats['rollout/episodes'] = self.epoch_episodes
        combined_stats['rollout/actions_std'] = np.std(self.epoch_actions)

        # Total statistics.
        combined_stats['total/epochs'] = self.epoch + 1
        combined_stats['total/steps'] = self.total_steps

        # Save combined_stats in a csv file.
        if file_path is not None:
            exists = os.path.exists(file_path)
            with open(file_path, 'a') as f:
                w = csv.DictWriter(f, fieldnames=combined_stats.keys())
                if not exists:
                    w.writeheader()
                w.writerow(combined_stats)

        # Print statistics.
        print("-" * 57)
        for key in sorted(combined_stats.keys()):
            val = combined_stats[key]
            print("| {:<25} | {:<25} |".format(key, val))
        print("-" * 57)
        print('')

    def _log_eval(self,
                  file_path,
                  start_time,
                  eval_episode_rewards,
                  eval_episode_successes,
                  eval_info):
        """Log evaluation statistics.

        Parameters
        ----------
        file_path : str
            path to the evaluation csv file
        start_time : float
            the time when training began. This is used to print the total
            training time.
        eval_episode_rewards : array_like
            the list of cumulative rewards from every episode in the evaluation
            phase
        eval_episode_successes : list of bool
            a list of boolean terms representing if each episode ended in
            success or not. If the list is empty, then the environment did not
            output successes or failures, and the success rate will be set to
            zero.
        eval_info : dict
            additional information that is meant to be logged
        """
        duration = time.time() - start_time

        if isinstance(eval_info, dict):
            eval_episode_rewards = [eval_episode_rewards]
            eval_episode_successes = [eval_episode_successes]
            eval_info = [eval_info]

        for i, (ep_rewards, ep_success, info) in enumerate(
                zip(eval_episode_rewards, eval_episode_successes, eval_info)):
            if len(eval_episode_successes) > 0:
                success_rate = np.mean(ep_success)
            else:
                success_rate = 0  # no success rate to log

            evaluation_stats = {
                "duration": duration,
                "total_step": self.total_steps,
                "success_rate": success_rate,
                "average_return": np.mean(ep_rewards)
            }
            # Add additional evaluation information.
            evaluation_stats.update(info)

            if file_path is not None:
                # Add an evaluation number to the csv file in case of multiple
                # evaluation environments.
                eval_fp = file_path[:-4] + "_{}.csv".format(i)
                exists = os.path.exists(eval_fp)

                # Save evaluation statistics in a csv file.
                with open(eval_fp, "a") as f:
                    w = csv.DictWriter(f, fieldnames=evaluation_stats.keys())
                    if not exists:
                        w.writeheader()
                    w.writerow(evaluation_stats)
