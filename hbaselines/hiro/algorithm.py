"""Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.

See: https://arxiv.org/pdf/1802.09477.pdf

A large portion of this code is adapted from the following repository:
https://github.com/hill-a/stable-baselines
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
from mpi4py import MPI

from hbaselines.hiro.tf_util import make_session
from hbaselines.hiro.policy import GoalDirectedPolicy
from hbaselines.common.train import ensure_dir
try:
    from flow.utils.registry import make_create_env
except (ImportError, ModuleNotFoundError):
    pass
from hbaselines.envs.efficient_hrl.envs import AntMaze, AntFall, AntPush
from hbaselines.envs.hac.envs import UR5, Pendulum


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
    sims_per_step : int
        number of sumo simulation steps performed in any given rollout step. RL
        agents perform the same action for the duration of these steps.
    gamma : float
        the discount rate
    eval_env : gym.Env or str
        the environment to evaluate from (if registered in Gym, can be str)
    nb_train_steps : int
        the number of training steps
    nb_rollout_steps : int
        the number of rollout steps
    nb_eval_episodes : int
        the number of evaluation episodes
    tau : float
        the soft update coefficient (keep old values, between 0 and 1)
    batch_size : int
        the size of the batch for learning the policy
    critic_l2_reg : float
        l2 regularizer coefficient
    actor_lr : float
        the actor learning rate
    critic_lr : float
        the critic learning rate
    clip_norm : float
        clip the gradients (disabled if None)
    reward_scale : float
        the value the reward should be scaled by
    render : bool
        enable rendering of the training environment
    render_eval : bool
        enable rendering of the evaluation environment
    buffer_size : int
        the max number of transitions to store
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    action_space : gym.spaces.*
        the action space of the training environment
    observation_space : gym.spaces.*
        the observation space of the training environment
    meta_period : int
        manger action period. Only applies to GoalDirectedPolicy
    relative_goals : bool
        specifies whether the goal issued by the Manager is meant to be a
        relative or absolute goal, i.e. specific state or change in state.
        Only applies to GoalDirectedPolicy
    off_policy_corrections : bool
        whether to use off-policy corrections during the update procedure.
        See: https://arxiv.org/abs/1805.08296. Only applies to
        GoalDirectedPolicy
    use_fingerprints : bool
        specifies whether to add a time-dependent fingerprint to the
        observations. Only applies to GoalDirectedPolicy
    fingerprint_range : (list of float, list of float)
        the low and high values for each fingerprint element, if they are being
        used
    fingerprint_dim : tuple of int
        the shape of the fingerprint elements, if they are being used
    centralized_value_functions : bool
        specifies whether to use centralized value functions for the
        Manager and Worker critic functions. Only applies to
        GoalDirectedPolicy
    connected_gradients : bool
        whether to connect the graph between the manager and worker.
        Defaults to False. Only applies to GoalDirectedPolicy
    graph : tf.Graph
        the current tensorflow graph
    policy_tf : hbaselines.hiro.policy.ActorCriticPolicy
        the policy object
    sess : tf.Session
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
    epoch_qs : list of float
        a list of the Q values that were calculated during the most recent
        training iteration
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
        evaluations occured. Used for logging purposes.
    eval_success_ph : tf.compat.v1.placeholder
        placeholder for the average evaluation success rate from the last time
        evaluations occured. Used for logging purposes.
    """

    def __init__(self,
                 policy,
                 env,
                 num_cpus=1,
                 sims_per_step=1,
                 gamma=0.99,
                 eval_env=None,
                 nb_train_steps=1,
                 nb_rollout_steps=1,
                 nb_eval_episodes=50,
                 tau=0.001,
                 batch_size=100,
                 critic_l2_reg=0.,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 clip_norm=None,
                 reward_scale=1.,
                 render=False,
                 render_eval=False,
                 memory_limit=None,
                 buffer_size=50000,
                 verbose=0,
                 meta_period=10,
                 relative_goals=False,
                 off_policy_corrections=False,
                 use_fingerprints=False,
                 centralized_value_functions=False,
                 connected_gradients=False,
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
        sims_per_step : int, optional
            number of sumo simulation steps performed in any given rollout
            step. RL agents perform the same action for the duration of these
            steps. Defaults to 5.
        gamma : float
            the discount rate
        eval_env : gym.Env or str
            the environment to evaluate from (if registered in Gym, can be str)
        nb_train_steps : int
            the number of training steps
        nb_rollout_steps : int
            the number of rollout steps
        nb_eval_episodes : int
            the number of evaluation episodes
        tau : float
            the soft update coefficient (keep old values, between 0 and 1)
        batch_size : int
            the size of the batch for learning the policy
        critic_l2_reg : float
            l2 regularizer coefficient
        actor_lr : float
            the actor learning rate
        critic_lr : float
            the critic learning rate
        clip_norm : float
            clip the gradients (disabled if None)
        reward_scale : float
            the value the reward should be scaled by
        render : bool
            enable rendering of the training environment
        render_eval : bool
            enable rendering of the evaluation environment
        buffer_size : int
            the max number of transitions to store
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        meta_period : int, optional
            manger action period. Only applies to GoalDirectedPolicy
        relative_goals : bool, optional
            specifies whether the goal issued by the Manager is meant to be a
            relative or absolute goal, i.e. specific state or change in state.
            Only applies to GoalDirectedPolicy
        off_policy_corrections : bool, optional
            whether to use off-policy corrections during the update procedure.
            See: https://arxiv.org/abs/1805.08296. Only applies to
            GoalDirectedPolicy
        use_fingerprints : bool, optional
            specifies whether to add a time-dependent fingerprint to the
            observations. Only applies to GoalDirectedPolicy
        centralized_value_functions : bool, optional
            specifies whether to use centralized value functions for the
            Manager and Worker critic functions. Only applies to
            GoalDirectedPolicy
        connected_gradients : bool, optional
            whether to connect the graph between the manager and worker. Only
            applies to GoalDirectedPolicy
        _init_setup_model : bool
            Whether or not to build the network at the creation of the instance
        """
        self.policy = policy
        self.env_name = deepcopy(env)
        self.env = self._create_env(env, evaluate=False)
        self.num_cpus = num_cpus
        self.sims_per_step = sims_per_step
        self.gamma = gamma
        self.eval_env = self._create_env(eval_env, evaluate=True)
        self.nb_train_steps = nb_train_steps
        self.nb_rollout_steps = nb_rollout_steps
        self.nb_eval_episodes = nb_eval_episodes
        self.tau = tau
        self.batch_size = batch_size
        self.critic_l2_reg = critic_l2_reg
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.reward_scale = reward_scale
        self.render = render
        self.render_eval = render_eval
        self.memory_limit = memory_limit
        self.buffer_size = buffer_size
        self.verbose = verbose
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.context_space = getattr(self.env, "context_space", None)
        self.meta_period = meta_period
        self.relative_goals = relative_goals
        self.off_policy_corrections = off_policy_corrections
        self.use_fingerprints = use_fingerprints
        self.fingerprint_range = ([0], [5])  # FIXME: parameter
        self.fingerprint_dim = (len(self.fingerprint_range[0]),)
        self.centralized_value_functions = centralized_value_functions
        self.connected_gradients = connected_gradients

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
        self.epoch_qs = None
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
        if self.use_fingerprints:
            low = np.concatenate(
                (self.observation_space.low, self.fingerprint_range[0]))
            high = np.concatenate(
                (self.observation_space.high, self.fingerprint_range[1]))
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
        gym.Env
            a gym-compatible environment
        """
        if env == "AntMaze":
            if evaluate:
                env = AntMaze(use_contexts=True, context_range=[16, 0])
                # env = AntMaze(use_contexts=True, context_range=[16, 16])
                # env = AntMaze(use_contexts=True, context_range=[0, 16])
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

            # Collect specific parameters only if using GoalDirectedPolicy.
            additional_params = {}
            if self.policy == GoalDirectedPolicy:
                additional_params.update({
                    "meta_period": self.meta_period,
                    "relative_goals": self.relative_goals,
                    "off_policy_corrections": self.off_policy_corrections,
                    "use_fingerprints": self.use_fingerprints,
                    "fingerprint_range": self.fingerprint_range,
                    "centralized_value_functions":
                        self.centralized_value_functions,
                    "connected_gradients": self.connected_gradients,
                    "env_name": self.env_name
                })

            # Create the policy.
            self.policy_tf = self.policy(
                self.sess,
                self.observation_space,
                self.action_space,
                self.context_space,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                actor_lr=self.actor_lr,
                critic_lr=self.critic_lr,
                clip_norm=self.clip_norm,
                critic_l2_reg=self.critic_l2_reg,
                verbose=self.verbose,
                tau=self.tau,
                gamma=self.gamma,
                **additional_params
            )

            # for tensorboard logging
            with tf.variable_scope("Train"):
                self.rew_ph = tf.compat.v1.placeholder(tf.float32)
                self.rew_history_ph = tf.compat.v1.placeholder(tf.float32)
            with tf.variable_scope("Evaluate"):
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

    def _policy(self, obs, apply_noise=True, compute_q=True, **kwargs):
        """Get the actions and critic output, from a given observation.

        Parameters
        ----------
        obs : list of float or list of int
            the observation
        apply_noise : bool
            enable the noise
        compute_q : bool
            compute the critic output

        Returns
        -------
        list of float
            the action value
        float
            the critic value
        """
        obs = np.array(obs).reshape((-1,) + self.observation_space.shape)

        action = self.policy_tf.get_action(
            obs, apply_noise,
            total_steps=self.total_steps,
            time=kwargs["episode_step"],
            context_obs=kwargs["context"])
        action = action.flatten()

        q_value = self.policy_tf.value(obs, context_obs=kwargs["context"]) \
            if compute_q else None

        return action, q_value

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
                                        context_obs1=context,
                                        time=self.episode_step)

    def learn(self,
              total_timesteps,
              log_dir=None,
              seed=None,
              log_interval=100,
              eval_interval=5e4,
              exp_num=None,
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
        exp_num : int, optional
            an additional experiment number term used by the runner scripts
            when running multiple experiments simultaneously. If set to None,
            the train, evaluate, and tensorboard results are stored in log_dir
            immediately
        start_timesteps : int, optional
            number of timesteps that the policy is run before training to
            initialize the replay buffer with samples
        """
        if exp_num is not None:
            log_dir = os.path.join(log_dir, "trial_{}".format(exp_num))

        # Make sure that the log directory exists, and if not, make it.
        ensure_dir(log_dir)

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
            if self.use_fingerprints:
                fp = [self.total_steps / total_timesteps * 5]
                self.obs = np.concatenate((self.obs, fp), axis=0)
            start_time = time.time()

            # Reset epoch-specific variables. FIXME: hacky
            self.epoch_episodes = 0
            self.epoch_actions = []
            self.epoch_qs = []
            self.epoch_actor_losses = []
            self.epoch_critic_losses = []
            self.epoch_episode_rewards = []
            self.epoch_episode_steps = []
            # Perform rollouts.
            print("Collecting pre-samples...")
            self._collect_samples(total_timesteps, run_steps=start_timesteps)
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
                self.epoch_qs = []
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
                    eval_rewards, eval_successes = self._evaluate(
                        total_timesteps)
                    self._log_eval(
                        eval_filepath,
                        start_time,
                        eval_rewards,
                        eval_successes
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
                self.save(os.path.join(log_dir, "itr"))

                # Update the epoch count.
                self.epoch += 1

    def save(self, save_path):
        """Save the parameters of a tensorflow model.

        save_path : str
            Prefix of filenames created for the checkpoint
        """
        self.saver.save(self.sess, save_path, global_step=self.total_steps)

    def load(self, load_path):
        """Load model parameters from a checkpoint.

        save_path : str
            location of the checkpoint
        """
        self.saver.restore(self.sess, load_path)

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
        rank = MPI.COMM_WORLD.Get_rank()

        new_obs, done = [], False
        for _ in range(run_steps or self.nb_rollout_steps):
            # Predict next action.
            action, q_value = self._policy(
                self.obs,
                apply_noise=True,
                compute_q=True,
                context=[getattr(self.env, "current_context", None)],
                episode_step=self.episode_step)
            assert action.shape == self.env.action_space.shape

            reward = 0
            for _ in range(self.sims_per_step):
                # Execute next action.
                new_obs, new_reward, done, info = self.env.step(action)
                reward += new_reward

                # Visualize the current step.
                if rank == 0 and self.render:
                    self.env.render()

            # Add the fingerprint term, if needed.
            if self.use_fingerprints:
                fp = [self.total_steps / total_timesteps * 5]
                new_obs = np.concatenate((new_obs, fp), axis=0)

            # Add the contextual reward to the environment reward.
            if hasattr(self.env, "current_context"):
                reward += getattr(self.env, "contextual_reward")(
                    self.obs, getattr(self.env, "current_context"), new_obs)

            # Store a transition in the replay buffer.
            self._store_transition(self.obs, action, reward, new_obs, done)

            # Book-keeping.
            self.total_steps += 1
            self.episode_reward += reward
            self.episode_step += 1
            self.epoch_actions.append(action)
            self.epoch_qs.append(q_value)

            # Update the current observation.
            self.obs = new_obs

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
                if self.use_fingerprints:
                    fp = [self.total_steps / total_timesteps * 5]
                    self.obs = np.concatenate((self.obs, fp), axis=0)

    def _train(self):
        """Perform the training operation.

        Through this method, the actor and critic networks are updated within
        the policy, and the summary information is logged to tensorboard.
        """
        for t_train in range(self.nb_train_steps):
            # Run a step of training from batch.
            critic_loss, actor_loss = self.policy_tf.update()

            # Add actor and critic loss information for logging purposes.
            self.epoch_critic_losses.append(critic_loss)
            self.epoch_actor_losses.append(actor_loss)

    def _evaluate(self, total_timesteps):
        """Perform the evaluation operation.

        This method runs the evaluation environment for a number of episodes
        and returns the cumulative rewards and successes from each environment.

        Parameters
        ----------
        total_timesteps : int
            the total number of samples to train on

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
        """
        num_steps = deepcopy(self.total_steps)
        eval_episode_rewards = []
        eval_episode_successes = []

        if self.verbose >= 1:
            for _ in range(3):
                print("-------------------")
            print("Running evaluation for {} episodes:".format(
                self.nb_eval_episodes))

        for i in range(self.nb_eval_episodes):
            # Reset the environment.
            eval_obs = self.eval_env.reset()

            # Add the fingerprint term, if needed.
            if self.use_fingerprints:
                fp = [num_steps / total_timesteps * 5]
                eval_obs = np.concatenate((eval_obs, fp), axis=0)

            # Reset rollout-specific variables.
            eval_episode_reward = 0.
            eval_episode_step = 0

            obs, done, info = [], False, {}
            while True:
                eval_action, _ = self._policy(
                    eval_obs,
                    apply_noise=False,
                    compute_q=False,
                    context=[getattr(self.eval_env, "current_context", None)],
                    episode_step=eval_episode_step)

                eval_r = 0
                for _ in range(self.sims_per_step):
                    obs, new_r, done, info = self.eval_env.step(eval_action)
                    eval_r += new_r

                    if self.render_eval:
                        self.eval_env.render()

                # Add the contextual reward to the environment reward.
                if hasattr(self.eval_env, "current_context"):
                    context_obs = getattr(self.eval_env, "current_context")
                    eval_r += getattr(self.eval_env, "contextual_reward")(
                        eval_obs, context_obs, obs)

                # Update the previous step observation.
                eval_obs = obs.copy()

                # Add the fingerprint term, if needed.
                if self.use_fingerprints:
                    fp = [num_steps / total_timesteps * 5]
                    eval_obs = np.concatenate((eval_obs, fp), axis=0)

                # Increment the reward and step count.
                num_steps += 1
                eval_episode_reward += eval_r
                eval_episode_step += 1

                if done:
                    eval_episode_rewards.append(eval_episode_reward)
                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        eval_episode_successes.append(float(maybe_is_success))

                    # Exit the loop.
                    break

            if self.verbose >= 1:
                print("{}/{}...".format(i+1, self.nb_eval_episodes))

        if self.verbose >= 1:
            print("Done.")
            print("Average return: {}".format(np.mean(eval_episode_rewards)))
            if len(eval_episode_successes) > 0:
                print("Success rate: {}".format(
                    np.mean(eval_episode_successes)))
            for _ in range(3):
                print("-------------------")
            print("")

        return eval_episode_rewards, eval_episode_successes

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
        mpi_size = MPI.COMM_WORLD.Get_size()

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
        combined_stats['rollout/Q_mean'] = np.mean(self.epoch_qs)
        combined_stats['train/loss_actor'] = np.mean(self.epoch_actor_losses)
        combined_stats['train/loss_critic'] = np.mean(self.epoch_critic_losses)
        combined_stats['total/duration'] = duration
        combined_stats['total/steps_per_second'] = self.total_steps / duration
        combined_stats['total/episodes'] = self.episodes
        combined_stats['rollout/episodes'] = self.epoch_episodes
        combined_stats['rollout/actions_std'] = np.std(self.epoch_actions)

        combined_stats_sums = MPI.COMM_WORLD.allreduce(
            np.array([as_scalar(x) for x in combined_stats.values()]))
        combined_stats = {k: v / mpi_size for (k, v) in
                          zip(combined_stats.keys(), combined_stats_sums)}

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
                  eval_episode_successes):
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
        """
        duration = time.time() - start_time

        if len(eval_episode_successes) > 0:
            success_rate = np.mean(eval_episode_successes)
        else:
            success_rate = 0  # no success rate to log

        evaluation_stats = {
            "duration": duration,
            "total_step": self.total_steps,
            "success_rate": success_rate,
            "average_return": np.mean(eval_episode_rewards)
        }

        # Save evaluation statistics in a csv file.
        if file_path is not None:
            exists = os.path.exists(file_path)
            with open(file_path, "a") as f:
                w = csv.DictWriter(f, fieldnames=evaluation_stats.keys())
                if not exists:
                    w.writeheader()
                w.writerow(evaluation_stats)
