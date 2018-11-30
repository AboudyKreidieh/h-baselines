"""Deep Deterministic Policy Gradient (DDPG) algorithm.

See: https://arxiv.org/pdf/1509.02971.pdf

The majority of this code is adapted from the following repository:
https://github.com/hill-a/stable-baselines
"""
import os
import time
from collections import deque
import pickle
import csv
import os.path
from copy import deepcopy

import gym
import numpy as np
import tensorflow as tf
from mpi4py import MPI

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, \
    TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines.a2c.utils import find_trainable_variables, \
    total_episode_reward_logger
from stable_baselines.ddpg.memory import Memory

from hbaselines.utils.exp_replay import RecurrentMemory
from hbaselines.utils.stats import reduce_std

# TODO: read https://arxiv.org/pdf/1602.07714.pdf


def get_target_updates(_vars, target_vars, tau, verbose=0):
    """Get target update operations.

    Parameters
    ----------
    _vars : list of tf.Tensor
        the initial variables
    target_vars : list of tf.Tensor
        the target variables
    tau : float
        the soft update coefficient (keep old values, between 0 and 1)
    verbose : int
        the verbosity level:

        * 0 none,
        * 1 training information,
        * 2 tensorflow debug

    Returns
    -------
    tf.Operation
        initial update
    tf.Operation
        soft update
    """
    if verbose >= 2:
        logger.info('Setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(_vars) == len(target_vars)
    for var, target_var in zip(_vars, target_vars):
        if verbose >= 2:
            logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(
            tf.assign(target_var, (1 - tau) * target_var + tau * var))
    assert len(init_updates) == len(_vars)
    assert len(soft_updates) == len(_vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbed_actor_updates(actor,
                                perturbed_actor,
                                param_noise_stddev,
                                verbose=0):
    """Get the actor update, with noise.

    Parameters
    ----------
    actor : str
        the actor
    perturbed_actor : str
        the perturbed actor
    param_noise_stddev : float
        the std of the parameter noise
    verbose : int
        the verbosity level:

        * 0 none
        * 1 training information
        * 2 tensorflow debug

    Returns
    -------
    tf.Operation
        the update function
    """
    assert len(tf_util.get_globals_vars(actor)) == \
        len(tf_util.get_globals_vars(perturbed_actor))
    assert len([var for var in tf_util.get_trainable_vars(actor)
                if 'LayerNorm' not in var.name]) == \
        len([var for var in tf_util.get_trainable_vars(perturbed_actor)
             if 'LayerNorm' not in var.name])

    updates = []
    for var, perturbed_var in zip(tf_util.get_globals_vars(actor),
                                  tf_util.get_globals_vars(perturbed_actor)):
        if var in [v for v in tf_util.get_trainable_vars(actor)
                   if 'LayerNorm' not in v.name]:
            if verbose >= 2:
                logger.info('  {} <- {} + noise'.format(perturbed_var.name,
                                                        var.name))
            updates.append(tf.assign(perturbed_var,
                                     var + tf.random_normal(
                                         shape=tf.shape(var),
                                         mean=0,
                                         stddev=param_noise_stddev)))
        else:
            if verbose >= 2:
                logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(tf_util.get_globals_vars(actor))
    return tf.group(*updates)


class DDPG(OffPolicyRLModel):
    """Deep Deterministic Policy Gradient (DDPG) model.

    DDPG: https://arxiv.org/pdf/1509.02971.pdf

    Parameters
    ----------
    policy : DDPGPolicy type or str
        The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    env : gym.Env or str
        The environment to learn from (if registered in Gym, can be str)
    recurrent : bool
        specifies whether recurrent policies are being used
    gamma : float
        the discount rate
    memory_policy : Memory type
        the replay buffer (if None, default to baselines.ddpg.memory.Memory)
    nb_train_steps : int
        the number of training steps
    nb_rollout_steps : int
        the number of rollout steps
    param_noise : AdaptiveParamNoiseSpec
        the parameter noise type (can be None)
    action_noise : ActionNoise
        the action noise type (can be None)
    param_noise_adaption_interval : int
        apply param noise every N steps
    tau : float
        the soft update coefficient (keep old values, between 0 and 1)
    normalize_returns : bool
        should the critic output be normalized
    normalize_observations : bool
        should the observation be normalized
    batch_size : int
        the size of the batch for learning the policy
    observation_range : tuple
        the bounding values for the observation
    return_range : tuple
        the bounding values for the critic output
    critic_l2_reg : float
        l2 regularizer coefficient
    actor_lr : float
        the actor learning rate
    critic_lr : float
        the critic learning rate
    clip_norm: float
        clip the gradients (disabled if None)
    reward_scale : float
        the value the reward should be scaled by
    render : bool
        enable rendering of the environment
    memory_limit : int
        the max number of transitions to store
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    tensorboard_log : str
        the log location for tensorboard (if None, no logging)
    _init_setup_model : bool
        Whether or not to build the network at the creation of the instance
    """

    def __init__(self,
                 policy,
                 env,
                 recurrent=False,
                 gamma=0.99,
                 memory_policy=None,
                 nb_train_steps=50,
                 nb_rollout_steps=100,
                 param_noise=None,
                 action_noise=None,
                 normalize_observations=False,
                 tau=0.001,
                 batch_size=128,
                 param_noise_adaption_interval=50,
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
                 _init_setup_model=True):

        # TODO: replay_buffer refactoring
        super(DDPG, self).__init__(policy=policy,
                                   env=env,
                                   replay_buffer=None,
                                   verbose=verbose,
                                   policy_base=DDPGPolicy,
                                   requires_vec_env=False)

        # Parameters
        self.recurrent = recurrent
        self.gamma = gamma
        self.tau = tau
        if recurrent:
            self.memory_policy = memory_policy or RecurrentMemory
        else:
            self.memory_policy = memory_policy or Memory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.return_range = return_range
        self.observation_range = observation_range
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.critic_l2_reg = critic_l2_reg
        self.render = render
        self.param_noise_adaption_interval = param_noise_adaption_interval
        self.nb_train_steps = nb_train_steps
        self.nb_rollout_steps = nb_rollout_steps
        self.memory_limit = memory_limit
        self.tensorboard_log = tensorboard_log

        # init
        self.graph = None
        self.stats_sample = None
        self.memory = None
        self.policy_tf = None
        self.target_init_updates = None
        self.target_soft_updates = None
        self.sess = None
        self.stats_ops = None
        self.stats_names = None
        self.perturbed_actor_tf = None
        self.perturb_policy_ops = None
        self.perturb_adaptive_policy_ops = None
        self.adaptive_policy_distance = None
        self.obs_rms = None
        self.ret_rms = None
        self.target_policy = None
        self.actor_tf = None
        self.target_q = None
        self.obs_train = None
        self.state_init = None
        self.obs_noise = None
        self.action_noise_ph = None
        self.obs_adapt_noise = None
        self.action_adapt_noise = None
        self.terminals1 = None
        self.rewards = None
        self.param_noise_stddev = None
        self.param_noise_actor = None
        self.adaptive_param_noise_actor = None
        self.params = None
        self.episode_reward = None
        self.tb_seen_steps = None

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        with SetVerbosity(self.verbose):
            # determine whether the action space is continuous
            assert isinstance(self.action_space, gym.spaces.Box), \
                "Error: DDPG cannot output a {} action space, only spaces." \
                "Box is supported.".format(self.action_space)

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.single_threaded_session(graph=self.graph)

                self.memory = self.memory_policy(
                    limit=self.memory_limit,
                    action_shape=self.action_space.shape,
                    observation_shape=self.observation_space.shape)

                with tf.variable_scope("input", reuse=False):
                    # Observation normalization.
                    if self.normalize_observations:
                        with tf.variable_scope('obs_rms'):
                            self.obs_rms = RunningMeanStd(
                                shape=self.observation_space.shape)
                    else:
                        self.obs_rms = None

                    # Return normalization.
                    if self.normalize_returns:
                        with tf.variable_scope('ret_rms'):
                            self.ret_rms = RunningMeanStd()
                    else:
                        self.ret_rms = None

                    # Create the policy networks.
                    self.policy_tf = self.policy(
                        sess=self.sess,
                        ob_space=self.observation_space,
                        ac_space=self.action_space,
                        obs_rms=self.obs_rms,
                        ret_rms=self.ret_rms,
                        return_range=self.return_range,
                        observation_range=self.observation_range
                    )
                    self.obs_train = self.policy_tf.obs_ph  # TODO: delete

                    # Create target networks.
                    self.target_policy = self.policy(
                        sess=self.sess,
                        ob_space=self.observation_space,
                        ac_space=self.action_space,
                        obs_rms=self.obs_rms,
                        ret_rms=self.ret_rms,
                        return_range=self.return_range,
                        observation_range=self.observation_range
                    )

                    if self.param_noise is not None:
                        # Configure perturbed actor.
                        self.param_noise_actor = self.policy(
                            sess=self.sess,
                            ob_space=self.observation_space,
                            ac_space=self.action_space,
                            obs_rms=self.obs_rms,
                            ret_rms=self.ret_rms,
                            return_range=self.return_range,
                            observation_range=self.observation_range
                        )
                        self.obs_noise = self.param_noise_actor.obs_ph  # TODO: delete
                        self.action_noise_ph = self.param_noise_actor.action_ph  # TODO: delete

                        # Configure separate copy for stddev adoption.
                        self.adaptive_param_noise_actor = self.policy(
                            sess=self.sess,
                            ob_space=self.observation_space,
                            ac_space=self.action_space,
                            obs_rms=self.obs_rms,
                            ret_rms=self.ret_rms,
                            return_range=self.return_range,
                            observation_range=self.observation_range
                        )
                        self.obs_adapt_noise = \
                            self.adaptive_param_noise_actor.obs_ph  # TODO: delete
                        self.action_adapt_noise = \
                            self.adaptive_param_noise_actor.action_ph  # TODO: delete

                    # Inputs.
                    self.terminals1 = tf.placeholder(
                        tf.float32, shape=(None, 1), name='terminals1')
                    self.rewards = tf.placeholder(
                        tf.float32, shape=(None, 1), name='rewards')
                    self.param_noise_stddev = tf.placeholder(
                        tf.float32, shape=(), name='param_noise_stddev')

                # Create networks and core TF parts that are shared across
                # setup parts.
                with tf.variable_scope("model", reuse=False):
                    self.actor_tf = self.policy_tf.make_actor()
                    _, _ = self.policy_tf.make_critic(use_actor=True)
                    if self.recurrent:
                        self.state_init = self.policy_tf.state_init

                # Noise setup
                if self.param_noise is not None:
                    self._setup_param_noise()

                with tf.variable_scope("target", reuse=False):
                    _ = self.target_policy.make_actor()
                    _, q_obs1 = self.target_policy.make_critic(
                        use_actor=True)
                    # this is the input to the internal states of the actor
                    if self.recurrent:
                        self.target_h_c = self.target_policy.states_ph  # TODO: delete
                        self.target_batch_size = self.target_policy.batch_size  # TODO: delete
                        self.target_train_length = self.target_policy.train_length  # TODO: delete

                with tf.variable_scope("loss", reuse=False):
                    self.target_q = self.rewards + (1 - self.terminals1) * \
                        self.gamma * q_obs1

                    # Set up parts.
                    self._setup_stats()
                    self._setup_target_network_updates()

                with tf.variable_scope("Adam_mpi", reuse=False):
                    self._setup_actor_optimizer()
                    self._setup_critic_optimizer()

                self.params = find_trainable_variables("model")

                with self.sess.as_default():
                    self._initialize(self.sess)

    def _setup_target_network_updates(self):
        """Set the target update operations."""
        init_updates, soft_updates = get_target_updates(
            tf_util.get_trainable_vars('model/'),
            tf_util.get_trainable_vars('target/'), self.tau,
            self.verbose)
        self.target_init_updates = init_updates
        self.target_soft_updates = soft_updates

    def _setup_param_noise(self):
        """Set the parameter noise operations."""
        assert self.param_noise is not None

        with tf.variable_scope("noise", reuse=False):
            self.perturbed_actor_tf = self.param_noise_actor.make_actor()

        with tf.variable_scope("noise_adapt", reuse=False):
            adaptive_actor_tf = self.adaptive_param_noise_actor.make_actor()

        with tf.variable_scope("noise_update_func", reuse=False):
            if self.verbose >= 2:
                logger.info('setting up param noise')
            self.perturb_policy_ops = get_perturbed_actor_updates(
                'model/pi/', 'noise/pi/', self.param_noise_stddev,
                verbose=self.verbose)

            self.perturb_adaptive_policy_ops = get_perturbed_actor_updates(
                'model/pi/', 'noise_adapt/pi/',
                self.param_noise_stddev, verbose=self.verbose)
            self.adaptive_policy_distance = tf.sqrt(
                tf.reduce_mean(tf.square(self.actor_tf - adaptive_actor_tf)))

    # TODO: update to include HRL dichotomy
    def _setup_actor_optimizer(self):
        """Setup the optimizer for the actor."""
        self.policy_tf.setup_actor_optimizer(
            clip_norm=self.clip_norm,
            verbose=self.verbose
        )

    # TODO: update to include HRL dichotomy (?)
    def _setup_critic_optimizer(self):
        """Setup the optimizer for the critic."""
        self.policy_tf.setup_critic_optimizer(
            critic_l2_reg=self.critic_l2_reg,
            clip_norm=self.clip_norm,
            verbose=self.verbose
        )

    def _setup_stats(self):
        """Setup the running means and std of the model inputs and outputs."""
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(
                self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.policy_tf.critic)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.policy_tf.critic)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.policy_tf.critic_with_actor)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.policy_tf.critic_with_actor)]
        names += ['reference_actor_Q_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['reference_action_std']

        if self.param_noise:
            ops += [tf.reduce_mean(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_mean']
            ops += [reduce_std(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_std']

        self.stats_ops = ops
        self.stats_names = names

    def _policy(self, obs, state, apply_noise=True, compute_q=True):
        """Get the actions and critic output, from a given observation.

        Parameters
        ----------
        obs : list of float or list of int
            the observation
        state : list of float or list of int
            internal state (for recurrent neural networks)
        apply_noise : bool
            enable the noise
        compute_q : bool
            compute the critic output

        Returns
        -------
        float or list of float
            the action
        float or list of float
            the next internal state of the actor (for RNNs)
        float or list of float
            the critic value
        """
        obs = np.array(obs).reshape((-1,) + self.observation_space.shape)

        if self.param_noise is not None and apply_noise:
            action = self.param_noise_actor.step(obs=obs, state=state)
        else:
            action = self.policy_tf.step(obs=obs, state=state)

        if self.recurrent:
            action, state1 = action
        else:
            state1 = None

        if compute_q:
            q_value = self.policy_tf.value(
                obs=obs, action=None, state=state, use_actor=True)
        else:
            q_value = None

        action = action.flatten()
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise
        action = np.clip(action, -1, 1)

        return action, state1, q_value

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
        reward *= self.reward_scale
        self.memory.append(obs0, action, reward, obs1, terminal1)
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs0]))

    def _train_step(self):
        """Run a step of training from batch.

        Returns
        -------
        float
            critic loss
        float
            actor loss
        """
        # do not start until there are at least two entries in the memory
        # buffer (needed for recurrent networks)
        if self.memory.nb_entries <= 1:
            return None, None

        # Get a batch
        batch = self.memory.sample(batch_size=self.batch_size)

        feed_dict = {
            self.target_policy.obs_ph: batch['obs1'],
            self.rewards: batch['rewards'],
            self.terminals1: batch['terminals1'].astype('float32')
        }
        if self.recurrent:
            feed_dict[self.target_train_length] = 8
            feed_dict[self.target_batch_size] = self.batch_size
            feed_dict[self.target_h_c] = (
                np.zeros([self.batch_size, self.policy_tf.layers[0]]),
                np.zeros([self.batch_size, self.policy_tf.layers[0]]))

        target_q = self.sess.run(self.target_q, feed_dict=feed_dict)

        return self.policy_tf.train_actor_critic(batch=batch,
                                                 target_q=target_q,
                                                 actor_lr=self.actor_lr,
                                                 critic_lr=self.critic_lr)

    def _initialize(self, sess):
        """Initialize the model parameters and optimizers

        Parameters
        ----------
        sess : tf.Session
            the current TensorFlow session
        """
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.policy_tf.actor_optimizer.sync()
        self.policy_tf.critic_optimizer.sync()
        self.sess.run(self.target_init_updates)

    def _update_target_net(self):
        """Run target soft update operation."""
        self.sess.run(self.target_soft_updates)

    def _get_stats(self):
        """Get the mean and standard dev of the model's inputs and outputs.

        Returns
        -------
        dict
            the means and stds
        """
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set
            # of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)

        feed_dict = {}

        for placeholder in [self.policy_tf.action_ph,
                            self.target_policy.action_ph,
                            self.action_adapt_noise,
                            self.action_noise_ph]:
            if placeholder is not None:
                feed_dict[placeholder] = self.stats_sample['actions']

        for placeholder in [self.obs_train,
                            self.target_policy.obs_ph,
                            self.obs_adapt_noise,
                            self.obs_noise]:
            if placeholder is not None:
                feed_dict[placeholder] = self.stats_sample['obs0']

        if self.recurrent:
            feed_dict[self.policy_tf.states_ph] = (
                np.zeros([self.batch_size, self.policy_tf.layers[0]]),
                np.zeros([self.batch_size, self.policy_tf.layers[0]]))
            feed_dict[self.policy_tf.batch_size] = self.batch_size
            feed_dict[self.policy_tf.train_length] = 8

        values = self.sess.run(self.stats_ops, feed_dict=feed_dict)

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def _adapt_param_noise(self):
        """Calculate the adaptation for the parameter noise.

        Returns
        -------
        float
            the mean distance for the parameter noise
        """
        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the
        # next "real" perturbation.
        batch = self.memory.sample(batch_size=self.batch_size)
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs_adapt_noise: batch['obs0'], self.obs_train: batch['obs0'],
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        mean_distance = MPI.COMM_WORLD.allreduce(
            distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        self.param_noise.adapt(mean_distance)
        return mean_distance

    def _reset(self):
        """Reset internal state after an episode is complete."""
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })

    def learn(self,
              total_timesteps,
              file_path=None,
              callback=None,
              seed=None,
              log_interval=100,
              tb_log_name="DDPG"):
        """Train an RL model.

        Parameters
        ----------
        total_timesteps : int
            The total number of samples to train on
        file_path : str, optional
            location of the save file
        seed : int
            The initial seed for training, if None: keep current seed
        callback : function (dict, dict)
            function called at every steps with state of the algorithm.
            It takes the local and global variables.
        log_interval : int
            The number of timesteps before logging.
        tb_log_name : str
            the name of the run for tensorboard log

        Returns
        -------
        BaseRLModel
            the trained model
        """
        with SetVerbosity(self.verbose), TensorboardWriter(
                self.graph, self.tensorboard_log, tb_log_name) as writer:
            self._setup_learn(seed)

            # a list for tensorboard logging, to prevent logging with the same
            # step number, if it already occurred
            self.tb_seen_steps = []

            rank = MPI.COMM_WORLD.Get_rank()
            # we assume symmetric actions.
            assert np.all(np.abs(self.env.action_space.low) ==
                          self.env.action_space.high)
            if self.verbose >= 2:
                logger.log('Using agent with the following configuration:')
                logger.log(str(self.__dict__.items()))

            episode_rewards_history = deque(maxlen=100)
            self.episode_reward = np.zeros((1,))
            with self.sess.as_default(), self.graph.as_default():
                # Prepare everything.
                self._reset()
                obs = self.env.reset()
                episodes = 0
                step = 0
                total_steps = 0

                start_time = time.time()

                epoch_episode_rewards = []
                epoch_episode_steps = []
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                epoch_actions = []
                epoch_qs = []
                episode_reward = 0
                episode_step = 0
                epoch_episodes = 0
                epoch = 0
                # internal state (for recurrent actors)
                state = deepcopy(self.state_init)
                while True:
                    for _ in range(log_interval):
                        # Perform rollouts.
                        for _ in range(self.nb_rollout_steps):
                            if total_steps >= total_timesteps:
                                return self

                            # Predict next action.
                            action, state, q_value = self._policy(
                                obs, state, apply_noise=True, compute_q=True)
                            assert action.shape == self.env.action_space.shape

                            # Execute next action.
                            if rank == 0 and self.render:
                                self.env.render()
                            new_obs, reward, done, _ = self.env.step(
                                action * np.abs(self.action_space.low))

                            if writer is not None:
                                ep_rew = np.array([reward]).reshape((1, -1))
                                ep_done = np.array([done]).reshape((1, -1))
                                self.episode_reward = \
                                    total_episode_reward_logger(
                                        self.episode_reward, ep_rew, ep_done,
                                        writer, total_steps)
                            step += 1
                            total_steps += 1
                            if rank == 0 and self.render:
                                self.env.render()
                            episode_reward += reward
                            episode_step += 1

                            # Book-keeping.
                            epoch_actions.append(action)
                            epoch_qs.append(q_value)
                            self._store_transition(
                                obs, action, reward, new_obs, done)
                            obs = new_obs
                            if callback is not None:
                                callback(locals(), globals())

                            if done:
                                # Episode done.
                                epoch_episode_rewards.append(episode_reward)
                                episode_rewards_history.append(episode_reward)
                                epoch_episode_steps.append(episode_step)
                                episode_reward = 0.
                                episode_step = 0
                                epoch_episodes += 1
                                episodes += 1
                                # internal state (for recurrent actors)
                                state = deepcopy(self.state_init)

                                self._reset()
                                if not isinstance(self.env, VecEnv):
                                    obs = self.env.reset()

                        # Train.
                        epoch_actor_losses = []
                        epoch_critic_losses = []
                        epoch_adaptive_distances = []
                        for t_train in range(self.nb_train_steps):
                            # Adapt param noise, if necessary.
                            if self.memory.nb_entries >= self.batch_size and \
                                    t_train % \
                                    self.param_noise_adaption_interval == 0:
                                distance = self._adapt_param_noise()
                                epoch_adaptive_distances.append(distance)

                            # weird equation to deal with the fact the
                            # nb_train_steps will be different to
                            # nb_rollout_steps
                            step = (int(t_train * (self.nb_rollout_steps /
                                                   self.nb_train_steps)) +
                                    total_steps - self.nb_rollout_steps)

                            critic_loss, actor_loss = self._train_step()

                            if critic_loss is not None:
                                epoch_critic_losses.append(critic_loss)
                                epoch_actor_losses.append(actor_loss)
                                self._update_target_net()

                    mpi_size = MPI.COMM_WORLD.Get_size()

                    # Log statistics.
                    # XXX shouldn't call np.mean on variable length lists
                    duration = time.time() - start_time
                    stats = self._get_stats()
                    combined_stats = stats.copy()
                    combined_stats['rollout/return'] = np.mean(
                        epoch_episode_rewards)
                    combined_stats['rollout/return_history'] = np.mean(
                        episode_rewards_history)
                    combined_stats['rollout/episode_steps'] = np.mean(
                        epoch_episode_steps)
                    combined_stats['rollout/actions_mean'] = np.mean(
                        epoch_actions)
                    combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
                    combined_stats['train/loss_actor'] = np.mean(
                        epoch_actor_losses)
                    combined_stats['train/loss_critic'] = np.mean(
                        epoch_critic_losses)
                    if len(epoch_adaptive_distances) != 0:
                        combined_stats['train/param_noise_distance'] = np.mean(
                            epoch_adaptive_distances)
                    combined_stats['total/duration'] = duration
                    combined_stats['total/steps_per_second'] = \
                        float(step) / float(duration)
                    combined_stats['total/episodes'] = episodes
                    combined_stats['rollout/episodes'] = epoch_episodes
                    combined_stats['rollout/actions_std'] = np.std(
                        epoch_actions)

                    def as_scalar(scalar):
                        """Check and return the input if it is a scalar.

                        If it is not scale, raise a ValueError.

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
                            raise ValueError(
                                'expected scalar, got %s' % scalar)

                    combined_stats_sums = MPI.COMM_WORLD.allreduce(
                        np.array([as_scalar(x)
                                  for x in combined_stats.values()]))
                    combined_stats = {k: v / mpi_size for (k, v) in zip(
                        combined_stats.keys(), combined_stats_sums)}

                    # Total statistics.
                    combined_stats['total/epochs'] = epoch + 1
                    combined_stats['total/steps'] = step

                    # save combined_stats in a csv file
                    if file_path is not None:
                        exists = os.path.exists(file_path)
                        with open(file_path, 'a') as f:
                            w = csv.DictWriter(
                                f, fieldnames=combined_stats.keys())
                            if not exists:
                                w.writeheader()
                            w.writerow(combined_stats)

                    for key in sorted(combined_stats.keys()):
                        logger.record_tabular(key, combined_stats[key])
                    logger.dump_tabular()
                    logger.info('')
                    logdir = logger.get_dir()
                    if rank == 0 and logdir:
                        if hasattr(self.env, 'get_state'):
                            with open(os.path.join(logdir, 'env_state.pkl'),
                                      'wb') as file_handler:
                                pickle.dump(self.env.get_state(), file_handler)

    # TODO: delete
    def predict(self, observation, state=None, mask=None, deterministic=True):
        pass

    # TODO: delete
    def action_probability(self, observation, state=None, mask=None):
        pass

    def save(self, save_path):
        data = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "param_noise_adaption_interval":
                self.param_noise_adaption_interval,
            "nb_train_steps": self.nb_train_steps,
            "nb_rollout_steps": self.nb_rollout_steps,
            "verbose": self.verbose,
            "param_noise": self.param_noise,
            "action_noise": self.action_noise,
            "gamma": self.gamma,
            "tau": self.tau,
            "normalize_returns": self.normalize_returns,
            "normalize_observations": self.normalize_observations,
            "batch_size": self.batch_size,
            "observation_range": self.observation_range,
            "return_range": self.return_range,
            "critic_l2_reg": self.critic_l2_reg,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "clip_norm": self.clip_norm,
            "reward_scale": self.reward_scale,
            "memory_limit": self.memory_limit,
            "policy": self.policy,
            "memory_policy": self.memory_policy,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, params = cls._load_from_file(load_path)

        model = cls(None, env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()

        restores = []
        for param, loaded_p in zip(model.params, params):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        return model
