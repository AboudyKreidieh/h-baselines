"""Deep Deterministic Policy Gradient (DDPG) algorithm.

See: https://arxiv.org/pdf/1509.02971.pdf

A large portion of this code is adapted from the following repository:
https://github.com/hill-a/stable-baselines
"""
from functools import reduce
import os
import time
from collections import deque
import csv
import pickle
import warnings

from gym.spaces import Box
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from mpi4py import MPI

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.deepq.replay_buffer import ReplayBuffer


def normalize(tensor, stats):
    """Normalize a tensor using a running mean and std.

    Parameters
    ----------
    tensor : tf.Tensor
        the input tensor
    stats : RunningMeanStd
        the running mean and std of the input to normalize

    Returns
    -------
    tf.Tensor
        the normalized tensor
    """
    if stats is None:
        return tensor
    return (tensor - stats.mean) / stats.std


def denormalize(tensor, stats):
    """Denormalize a tensor using a running mean and std.

    Parameters
    ----------
    tensor : tf.Tensor
        the normalized tensor
    stats : RunningMeanStd
        the running mean and std of the input to normalize

    Returns
    -------
    tf.Tensor
        the restored tensor
    """
    if stats is None:
        return tensor
    return tensor * stats.std + stats.mean


def reduce_std(tensor, axis=None, keepdims=False):
    """Get the standard deviation of a Tensor.

    Parameters
    ----------
    tensor : tf.Tensor
        the input tensor
    axis : int or list of int
        the axis to itterate the std over
    keepdims : bool
        keep the other dimensions the same

    Returns
    -------
    tf.Tensor
        the std of the tensor
    """
    return tf.sqrt(reduce_var(tensor, axis=axis, keepdims=keepdims))


def reduce_var(tensor, axis=None, keepdims=False):
    """Get the variance of a Tensor.

    Parameters
    ----------
    tensor : tf.Tensor
        the input tensor
    axis : int or list of int
        the axis to itterate the variance over
    keepdims : bool
        keep the other dimensions the same

    Returns
    -------
    tf.Tensor
        the variance of the tensor
    """
    tensor_mean = tf.reduce_mean(tensor, axis=axis, keepdims=True)
    devs_squared = tf.square(tensor - tensor_mean)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


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
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug

    Returns
    -------
    tf.Operation
        initial update
    tf.Operation
        soft update
    """
    if verbose >= 2:
        logger.info('setting up target updates ...')

    soft_updates = []
    init_updates = []
    assert len(_vars) == len(target_vars)

    for var, target_var in zip(_vars, target_vars):
        if verbose >= 2:
            logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(
            tf.assign(target_var, (1. - tau) * target_var + tau * var))

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
        the pertubed actor
    param_noise_stddev : float
        the std of the parameter noise
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug

    Returns
    -------
    tf.Operation
        the update function
    """

    assert len(tf_util.get_globals_vars(actor)) == \
        len(tf_util.get_globals_vars(perturbed_actor))
    assert \
        len([var for var in tf_util.get_trainable_vars(actor)
             if 'LayerNorm' not in var.name]) == \
        len([var for var in tf_util.get_trainable_vars(perturbed_actor)
             if 'LayerNorm' not in var.name])

    updates = []
    for var, perturbed_var in zip(tf_util.get_globals_vars(actor),
                                  tf_util.get_globals_vars(perturbed_actor)):
        if var in [var for var in tf_util.get_trainable_vars(actor)
                   if 'LayerNorm' not in var.name]:
            if verbose >= 2:
                logger.info('  {} <- {} + noise'.format(
                    perturbed_var.name, var.name))
            updates.append(
                tf.assign(perturbed_var, var + tf.random_normal(
                    tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            if verbose >= 2:
                logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(tf_util.get_globals_vars(actor))
    return tf.group(*updates)


class DDPG(OffPolicyRLModel):
    """Deep Deterministic Policy Gradient (DDPG) model.

    See: https://arxiv.org/pdf/1509.02971.pdf

    Parameters
    ----------
    policy : DDPGPolicy type or str
        The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    env : gym.Env or str
        The environment to learn from (if registered in Gym, can be str)
    gamma : float
        the discount rate
    nb_train_steps : int
        the number of training steps
    nb_rollout_steps : int
        the number of rollout steps
    action_noise : ActionNoise
        the action noise type (can be None)
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
                 gamma=0.99,
                 eval_env=None,
                 nb_train_steps=50,
                 nb_rollout_steps=100,
                 nb_eval_steps=100,
                 param_noise=None,
                 action_noise=None,
                 normalize_observations=False,
                 tau=0.001,
                 batch_size=128,
                 param_noise_adaption_interval=50,
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
                 memory_limit=None,
                 buffer_size=50000,
                 random_exploration=0.0,
                 verbose=0,
                 tensorboard_log=None,
                 _init_setup_model=True,
                 policy_kwargs=None):
        # Parameters.
        self.gamma = gamma
        self.tau = tau

        super(DDPG, self).__init__(
            policy=policy, env=env, replay_buffer=None, verbose=verbose,
            policy_base=DDPGPolicy, requires_vec_env=False,
            policy_kwargs=policy_kwargs)

        if memory_limit is not None:
            warnings.warn(
                "memory_limit will be removed in a future version (v3.x.x) "
                "use buffer_size instead", DeprecationWarning)
            buffer_size = memory_limit

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
        self.eval_env = eval_env
        self.render = render
        self.render_eval = render_eval
        self.nb_eval_steps = nb_eval_steps
        self.param_noise_adaption_interval = param_noise_adaption_interval
        self.nb_train_steps = nb_train_steps
        self.nb_rollout_steps = nb_rollout_steps
        self.memory_limit = memory_limit
        self.buffer_size = buffer_size
        self.tensorboard_log = tensorboard_log
        self.random_exploration = random_exploration

        # init
        self.graph = None
        self.stats_sample = None
        self.replay_buffer = None
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
        self.old_std = None
        self.old_mean = None
        self.renormalize_q_outputs_op = None
        self.obs_rms = None
        self.ret_rms = None
        self.target_q = None
        self.obs_train = None
        self.action_train_ph = None
        self.obs_target = None
        self.action_target = None
        self.obs_noise = None
        self.action_noise_ph = None
        self.obs_adapt_noise = None
        self.action_adapt_noise = None
        self.terminals1 = None
        self.rewards = None
        self.actions = None
        self.param_noise_stddev = None
        self.param_noise_actor = None
        self.adaptive_param_noise_actor = None
        self.params = None
        self.summary = None
        self.episode_reward = None
        self.tb_seen_steps = None

        # TODO: move to actor class
        self.actor_loss = None
        self.actor_grads = None
        self.actor_optimizer = None
        self.target_policy = None
        self.actor_tf = None

        # TODO: move to critic class
        self.critic_loss = None
        self.critic_grads = None
        self.critic_optimizer = None
        self.normalized_critic_tf = None
        self.critic_tf = None
        self.normalized_critic_with_actor_tf = None
        self.critic_with_actor_tf = None
        self.critic_target = None

        self.target_params = None
        self.obs_rms_params = None
        self.ret_rms_params = None

        # TODO: my hacks
        if "feature_extraction" not in self.policy_kwargs:
            self.policy_kwargs["feature_extraction"] = None
        self.policy_kwargs["feature_extraction"] = "mlp"
        self.obs = None
        self.eval_obs = None
        self.episode_reward = 0.
        self.episode_step = 0
        self.episodes = 0
        self.step = 0
        self.total_steps = 0
        self.epoch_episode_rewards = []
        self.epoch_episode_steps = []
        self.epoch_actor_losses = []
        self.epoch_critic_losses = []
        self.epoch_adaptive_distances = []
        self.eval_episode_rewards = []
        self.eval_qs = []
        self.epoch_actions = []
        self.epoch_qs = []
        self.epoch_episodes = 0
        self.epoch = 0
        self.eval_episode_rewards_history = deque(maxlen=100)
        self.episode_rewards_history = deque(maxlen=100)
        self.episode_reward = np.zeros((1,))
        self.episode_successes = []

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        deterministic_action = self.actor_tf * np.abs(self.action_space.low)

        return policy.obs_ph, self.actions, deterministic_action

    def setup_model(self):

        with SetVerbosity(self.verbose):
            # determine whether the action space is continuous
            assert isinstance(self.action_space, Box), \
                "Error: DDPG cannot output a {} action space, only " \
                "spaces.Box is supported.".format(self.action_space)
            # print(self.policy)
            # assert issubclass(self.policy, DDPGPolicy), \
            #     "Error: the input policy for the DDPG model must be " \
            #     "an instance of DDPGPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.single_threaded_session(graph=self.graph)

                self.replay_buffer = ReplayBuffer(self.buffer_size)

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

                    self.policy_tf = self.policy(
                        self.sess,
                        self.observation_space,
                        self.action_space,
                        1, 1, None, **self.policy_kwargs)

                    # Create target networks.
                    self.target_policy = self.policy(
                        self.sess,
                        self.observation_space,
                        self.action_space,
                        1, 1, None, **self.policy_kwargs)

                    self.obs_target = self.target_policy.obs_ph
                    self.action_target = self.target_policy.action_ph

                    normalized_obs0 = tf.clip_by_value(
                        normalize(self.policy_tf.processed_obs, self.obs_rms),
                        self.observation_range[0], self.observation_range[1])
                    normalized_obs1 = tf.clip_by_value(
                        normalize(self.target_policy.processed_obs,
                                  self.obs_rms),
                        self.observation_range[0], self.observation_range[1])

                    if self.param_noise is not None:
                        # Configure perturbed actor.
                        self.param_noise_actor = self.policy(
                            self.sess,
                            self.observation_space,
                            self.action_space,
                            1, 1, None, **self.policy_kwargs)
                        self.obs_noise = self.param_noise_actor.obs_ph
                        self.action_noise_ph = self.param_noise_actor.action_ph

                        # Configure separate copy for stddev adoption.
                        self.adaptive_param_noise_actor = self.policy(
                            self.sess,
                            self.observation_space,
                            self.action_space,
                            1, 1, None, **self.policy_kwargs)
                        self.obs_adapt_noise = \
                            self.adaptive_param_noise_actor.obs_ph
                        self.action_adapt_noise = \
                            self.adaptive_param_noise_actor.action_ph

                    # Inputs.
                    self.obs_train = self.policy_tf.obs_ph
                    self.action_train_ph = self.policy_tf.action_ph
                    self.terminals1 = tf.placeholder(
                        tf.float32,
                        shape=(None, 1),
                        name='terminals1')
                    self.rewards = tf.placeholder(
                        tf.float32,
                        shape=(None, 1),
                        name='rewards')
                    self.actions = tf.placeholder(
                        tf.float32,
                        shape=(None,) + self.action_space.shape,
                        name='actions')
                    self.critic_target = tf.placeholder(
                        tf.float32,
                        shape=(None, 1),
                        name='critic_target')
                    self.param_noise_stddev = tf.placeholder(
                        tf.float32,
                        shape=(),
                        name='param_noise_stddev')

                # Create networks and core TF parts that are shared across
                # setup parts.
                with tf.variable_scope("model", reuse=False):
                    self.actor_tf = self.policy_tf.make_actor(normalized_obs0)
                    self.normalized_critic_tf = self.policy_tf.make_critic(
                        normalized_obs0,
                        self.actions)
                    self.normalized_critic_with_actor_tf = \
                        self.policy_tf.make_critic(normalized_obs0,
                                                   self.actor_tf,
                                                   reuse=True)

                # Noise setup
                if self.param_noise is not None:
                    self._setup_param_noise(normalized_obs0)

                with tf.variable_scope("target", reuse=False):
                    critic_target = self.target_policy.make_critic(
                        normalized_obs1,
                        self.target_policy.make_actor(normalized_obs1))

                with tf.variable_scope("loss", reuse=False):
                    self.critic_tf = denormalize(
                        tf.clip_by_value(
                            self.normalized_critic_tf,
                            self.return_range[0],
                            self.return_range[1]),
                        self.ret_rms)

                    self.critic_with_actor_tf = denormalize(
                        tf.clip_by_value(
                            self.normalized_critic_with_actor_tf,
                            self.return_range[0],
                            self.return_range[1]),
                        self.ret_rms)

                    q_obs1 = denormalize(critic_target, self.ret_rms)
                    self.target_q = self.rewards + \
                        (1. - self.terminals1) * self.gamma * q_obs1

                    tf.summary.scalar('critic_target',
                                      tf.reduce_mean(self.critic_target))

                    self._setup_stats()
                    self._setup_target_network_updates()

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('rewards', tf.reduce_mean(self.rewards))
                    tf.summary.scalar('param_noise_stddev',
                                      tf.reduce_mean(self.param_noise_stddev))

                with tf.variable_scope("Adam_mpi", reuse=False):
                    self._setup_actor_optimizer()
                    self._setup_critic_optimizer()
                    tf.summary.scalar('actor_loss', self.actor_loss)
                    tf.summary.scalar('critic_loss', self.critic_loss)

                self.params = \
                    tf_util.get_trainable_vars("model") + \
                    tf_util.get_trainable_vars('noise/') + \
                    tf_util.get_trainable_vars('noise_adapt/')

                self.target_params = tf_util.get_trainable_vars("target")
                self.obs_rms_params = [var for var in tf.global_variables()
                                       if "obs_rms" in var.name]
                self.ret_rms_params = [var for var in tf.global_variables()
                                       if "ret_rms" in var.name]

                with self.sess.as_default():
                    self._initialize(self.sess)

                self.summary = tf.summary.merge_all()

    def _setup_target_network_updates(self):
        """Set the target update operations."""
        init_updates, soft_updates = get_target_updates(
            tf_util.get_trainable_vars('model/'),
            tf_util.get_trainable_vars('target/'),
            self.tau, self.verbose)
        self.target_init_updates = init_updates
        self.target_soft_updates = soft_updates

    def _setup_param_noise(self, normalized_obs0):
        """Set the parameter noise operations.

        :param normalized_obs0: (TensorFlow Tensor) the normalized observation
        """
        assert self.param_noise is not None

        with tf.variable_scope("noise", reuse=False):
            self.perturbed_actor_tf = self.param_noise_actor.make_actor(
                normalized_obs0)

        with tf.variable_scope("noise_adapt", reuse=False):
            adaptive_actor_tf = self.adaptive_param_noise_actor.make_actor(
                normalized_obs0)

        with tf.variable_scope("noise_update_func", reuse=False):
            if self.verbose >= 2:
                logger.info('setting up param noise')
            self.perturb_policy_ops = get_perturbed_actor_updates(
                'model/pi/',
                'noise/pi/',
                self.param_noise_stddev,
                verbose=self.verbose)

            self.perturb_adaptive_policy_ops = get_perturbed_actor_updates(
                'model/pi/',
                'noise_adapt/pi/',
                self.param_noise_stddev,
                verbose=self.verbose)
            self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(
                tf.square(self.actor_tf - adaptive_actor_tf)))

    def _setup_actor_optimizer(self):
        """Setup the optimizer for the actor."""
        if self.verbose >= 2:
            logger.info('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list()
                        for var in tf_util.get_trainable_vars('model/pi/')]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape)
                               for shape in actor_shapes])
        if self.verbose >= 2:
            logger.info('  actor shapes: {}'.format(actor_shapes))
            logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = tf_util.flatgrad(
            self.actor_loss,
            tf_util.get_trainable_vars('model/pi/'),
            clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(
            var_list=tf_util.get_trainable_vars('model/pi/'),
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def _setup_critic_optimizer(self):
        """Setup the optimizer for the critic."""
        if self.verbose >= 2:
            logger.info('setting up critic optimizer')

        normalized_critic_target_tf = tf.clip_by_value(
            normalize(self.critic_target, self.ret_rms),
            self.return_range[0],
            self.return_range[1])

        self.critic_loss = tf.reduce_mean(tf.square(
            self.normalized_critic_tf - normalized_critic_target_tf))

        if self.critic_l2_reg > 0.:
            critic_reg_vars = [
                var for var in tf_util.get_trainable_vars('model/qf/')
                if 'bias' not in var.name
                and 'qf_output' not in var.name
                and 'b' not in var.name
            ]

            if self.verbose >= 2:
                for var in critic_reg_vars:
                    logger.info('  regularizing: {}'.format(var.name))
                logger.info('  applying l2 regularization with {}'.format(
                    self.critic_l2_reg))

            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg

        critic_shapes = [var.get_shape().as_list()
                         for var in tf_util.get_trainable_vars('model/qf/')]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                for shape in critic_shapes])

        if self.verbose >= 2:
            logger.info('  critic shapes: {}'.format(critic_shapes))
            logger.info('  critic params: {}'.format(critic_nb_params))

        self.critic_grads = tf_util.flatgrad(
            self.critic_loss,
            tf_util.get_trainable_vars('model/qf/'),
            clip_norm=self.clip_norm)

        self.critic_optimizer = MpiAdam(
            var_list=tf_util.get_trainable_vars('model/qf/'),
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def _setup_stats(self):
        """Setup the running means and std of the model inputs and outputs."""
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean),
                    tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
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

    def _policy(self, obs, apply_noise=True, compute_q=True):
        """
        Get the actions and critic output, from a given observation

        :param obs: ([float] or [int]) the observation
        :param apply_noise: (bool) enable the noise
        :param compute_q: (bool) compute the critic output
        :return: ([float], float) the action and critic value
        """
        obs = np.array(obs).reshape((-1,) + self.observation_space.shape)
        feed_dict = {self.obs_train: obs}
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
            feed_dict[self.obs_noise] = obs
        else:
            actor_tf = self.actor_tf

        if compute_q:
            action, q_value = self.sess.run(
                [actor_tf, self.critic_with_actor_tf],
                feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q_value = None

        action = action.flatten()
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise
        action = np.clip(action, -1, 1)

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
        reward *= self.reward_scale
        self.replay_buffer.add(obs0, action, reward, obs1, float(terminal1))
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs0]))

    def _train_step(self, step, writer, log=False):
        """Run a step of training from batch.

        Parameters
        ----------
        step : int
            the current step iteration
        writer : tf.Summary.writer
            the writer for tensorboard
        log : bool
            whether or not to log to metadata

        Returns
        -------
        float
            critic loss
        float
            actor loss
        """
        # Get a batch
        obs0, actions, rewards, obs1, terminals1 = self.replay_buffer.sample(
            batch_size=self.batch_size)

        # Reshape to match previous behavior and placeholder shape
        rewards = rewards.reshape(-1, 1)
        terminals1 = terminals1.reshape(-1, 1)

        target_q = self.sess.run(self.target_q, feed_dict={
            self.obs_target: obs1,
            self.rewards: rewards,
            self.terminals1: terminals1
        })

        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads,
               self.critic_loss]
        td_map = {
            self.obs_train: obs0,
            self.actions: actions,
            self.action_train_ph: actions,
            self.rewards: rewards,
            self.critic_target: target_q,
            self.param_noise_stddev:
                0 if self.param_noise is None
                else self.param_noise.current_stddev
        }
        if writer is not None:
            # run loss backprop with summary if the step_id was not already
            # logged (can happen with the right parameters as the step value is
            # only an estimate)
            summary, actor_grads, actor_loss, critic_grads, critic_loss = \
                self.sess.run([self.summary] + ops, td_map)
            writer.add_summary(summary, step)
        else:
            actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(
                ops, td_map)

        self.actor_optimizer.update(
            actor_grads, learning_rate=self.actor_lr)
        self.critic_optimizer.update(
            critic_grads, learning_rate=self.critic_lr)

        return critic_loss, actor_loss

    def _initialize(self, sess):
        """Initialize the model parameters and optimizers.

        Parameters
        ----------
        sess : tf.Session
            the current TensorFlow session
        """
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()
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
            obs0, actions, rewards, obs1, terminals1 = \
                self.replay_buffer.sample(batch_size=self.batch_size)
            self.stats_sample = {
                'obs0': obs0,
                'actions': actions,
                'rewards': rewards,
                'obs1': obs1,
                'terminals1': terminals1
            }

        feed_dict = {
            self.actions: self.stats_sample['actions']
        }

        for placeholder in [self.action_train_ph, self.action_target,
                            self.action_adapt_noise, self.action_noise_ph]:
            if placeholder is not None:
                feed_dict[placeholder] = self.stats_sample['actions']

        for placeholder in [self.obs_train, self.obs_target,
                            self.obs_adapt_noise, self.obs_noise]:
            if placeholder is not None:
                feed_dict[placeholder] = self.stats_sample['obs0']

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
        obs0, *_ = self.replay_buffer.sample(batch_size=self.batch_size)
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs_adapt_noise: obs0, self.obs_train: obs0,
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) \
            / MPI.COMM_WORLD.Get_size()
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
              tb_log_name="DDPG",
              reset_num_timesteps=True,
              replay_wrapper=None):
        """TODO

        :param total_timesteps:
        :param file_path:
        :param callback:
        :param seed:
        :param log_interval:
        :param tb_log_name:
        :param reset_num_timesteps:
        :param replay_wrapper:
        :return:
        """
        # new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        # Create a tensorboard object for logging.
        # save_path = os.path.join()
        # writer = tf.summary.FileWriter(save_path, graph=self.graph)
        writer = None

        with SetVerbosity(self.verbose):
            self._setup_learn(seed)

            # a list for tensorboard logging, to prevent logging with the same
            # step number, if it already occurred
            self.tb_seen_steps = []

            # we assume symmetric actions.  # FIXME
            assert np.all(np.abs(self.env.action_space.low)
                          == self.env.action_space.high)
            if self.verbose >= 2:
                logger.log('Using agent with the following configuration:')
                logger.log(str(self.__dict__.items()))

            # Reset class variables.
            self.episode_reward = 0.
            self.episode_step = 0
            self.episodes = 0
            self.step = 0
            self.total_steps = 0
            self.epoch_episode_rewards = []
            self.epoch_episode_steps = []
            self.epoch_actor_losses = []
            self.epoch_critic_losses = []
            self.epoch_adaptive_distances = []
            self.eval_episode_rewards = []
            self.eval_qs = []
            self.epoch_actions = []
            self.epoch_qs = []
            self.epoch_episodes = 0
            self.epoch = 0
            self.eval_episode_rewards_history = deque(maxlen=100)
            self.episode_rewards_history = deque(maxlen=100)
            self.episode_reward = np.zeros((1,))
            self.episode_successes = []

            with self.sess.as_default(), self.graph.as_default():
                # Prepare everything.
                self._reset()
                self.obs = self.env.reset()
                if self.eval_env is not None:
                    self.eval_obs = self.eval_env.reset()
                start_time = time.time()

                while True:
                    for _ in range(log_interval):
                        # If the requirement number of time steps has been met,
                        # terminate training.
                        if self.total_steps >= total_timesteps:
                            return self

                        # Perform rollouts.
                        self._collect_samples(writer, callback)

                        # Train.
                        self._train(writer)

                        # Evaluate.
                        eval_episode_rewards = []
                        eval_qs = []
                        if self.eval_env is not None:
                            eval_episode_rewards, eval_qs = self._evaluate()
                            self.eval_episode_rewards_history.extend(
                                eval_episode_rewards)

                    # Log statistics.
                    self._log_stats(
                        file_path,
                        start_time,
                        eval_episode_rewards,
                        eval_qs,
                    )

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(
            observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions, _, = self._policy(
            observation, apply_noise=not deterministic, compute_q=False)

        # reshape to the correct action shape
        actions = actions.reshape((-1,) + self.action_space.shape)
        # scale the output for the prediction
        actions = actions * np.abs(self.action_space.low)

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def action_probability(self, observation, state=None, mask=None,
                           actions=None):
        if actions is not None:
            raise ValueError("Error: DDPG does not have action probabilities.")

        # here there are no action probabilities, as DDPG does not use a
        # probability distribution
        warnings.warn("Warning: action probability is meaningless for DDPG. "
                      "Returning None")

        return None

    def get_parameter_list(self):
        return (self.params +
                self.target_params +
                self.obs_rms_params +
                self.ret_rms_params)

    def save(self, save_path):
        data = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "nb_eval_steps": self.nb_eval_steps,
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
            "buffer_size": self.buffer_size,
            "random_exploration": self.random_exploration,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path,
                           data=data,
                           params=params_to_save)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, params = cls._load_from_file(load_path)

        if 'policy_kwargs' in kwargs \
                and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError(
                "The specified policy kwargs do not equal the stored policy "
                "kwargs. Stored kwargs: {}, specified kwargs: {}".
                format(data['policy_kwargs'], kwargs['policy_kwargs']))

        model = cls(None, env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()
        # Patch for version < v2.6.0, duplicated keys where saved
        if len(params) > len(model.get_parameter_list()):
            n_params = len(model.params)
            n_target_params = len(model.target_params)
            n_normalisation_params = len(model.obs_rms_params) + len(
                model.ret_rms_params)
            # Check that the issue is the one from
            # https://github.com/hill-a/stable-baselines/issues/363
            assert len(params) == 2 * (n_params + n_target_params) \
                + n_normalisation_params,\
                "The number of parameter saved differs from the number of " \
                "parameters that should be loaded: {}!={}".format(
                    len(params), len(model.get_parameter_list()))

            # Remove duplicates
            params_ = params[:n_params + n_target_params]
            if n_normalisation_params > 0:
                params_ += params[-n_normalisation_params:]
            params = params_
        model.load_parameters(params)

        return model

    def _collect_samples(self, writer, callback):
        """

        :return:
        """
        rank = MPI.COMM_WORLD.Get_rank()

        for _ in range(self.nb_rollout_steps):
            # Predict next action.
            action, q_value = self._policy(
                self.obs, apply_noise=True, compute_q=True)
            assert action.shape == self.env.action_space.shape

            # Execute next action.
            if rank == 0 and self.render:
                self.env.render()

            # Randomly sample actions from a uniform distribution with a
            # probability self.random_exploration (used in HER + DDPG)
            if np.random.rand() < self.random_exploration:
                rescaled_action = action = self.action_space.sample()
            else:
                rescaled_action = action * np.abs(self.action_space.low)

            new_obs, reward, done, info = self.env.step(rescaled_action)

            if writer is not None:
                ep_rew = np.array([reward]).reshape((1, -1))
                ep_done = np.array([done]).reshape((1, -1))
                self.episode_reward = total_episode_reward_logger(
                    self.episode_reward, ep_rew, ep_done, writer,
                    self.num_timesteps)

            self.step += 1
            self.total_steps += 1
            self.num_timesteps += 1
            if rank == 0 and self.render:
                self.env.render()
            self.episode_reward += reward
            self.episode_step += 1

            # Book-keeping.
            self.epoch_actions.append(action)
            self.epoch_qs.append(q_value)
            self._store_transition(self.obs, action, reward, new_obs, done)

            self.obs = new_obs
            if callback is not None:
                # Only stop training if return value is False, not when it is
                # None. This is for backwards compatibility with callbacks that
                # have no return statement.
                if callback(locals(), globals()) is False:
                    return self

            if done:
                # Episode done.
                self.epoch_episode_rewards.append(self.episode_reward)
                self.episode_rewards_history.append(self.episode_reward)
                self.epoch_episode_steps.append(self.episode_step)
                self.episode_reward = 0.
                self.episode_step = 0
                self.epoch_episodes += 1
                self.episodes += 1

                maybe_is_success = info.get('is_success')
                if maybe_is_success is not None:
                    self.episode_successes.append(float(maybe_is_success))

                self._reset()
                self.obs = self.env.reset()

    def _train(self, writer):
        self.epoch_actor_losses = []
        self.epoch_critic_losses = []
        self.epoch_adaptive_distances = []
        for t_train in range(self.nb_train_steps):
            # Not enough samples in the replay buffer.
            if not self.replay_buffer.can_sample(self.batch_size):
                break

            # Adapt param noise, if necessary.
            if len(self.replay_buffer) >= self.batch_size and \
                    t_train % self.param_noise_adaption_interval == 0:
                distance = self._adapt_param_noise()
                self.epoch_adaptive_distances.append(distance)

            # weird equation to deal with the fact the nb_train steps will be
            # different to nb_rollout_steps
            step = (
                int(t_train * (self.nb_rollout_steps / self.nb_train_steps)) +
                self.num_timesteps - self.nb_rollout_steps)

            critic_loss, actor_loss = self._train_step(
                step, writer, log=t_train == 0)

            # add actor and critic loss information for logging purposes
            self.epoch_critic_losses.append(critic_loss)
            self.epoch_actor_losses.append(actor_loss)

            # update the target networks
            self._update_target_net()

    def _evaluate(self):
        """

        :return:
        """
        eval_episode_rewards = []
        eval_qs = []
        eval_episode_reward = 0.
        for _ in range(self.nb_eval_steps):
            eval_action, eval_q = self._policy(
                self.eval_obs,
                apply_noise=False,
                compute_q=True)

            self.eval_obs, eval_r, eval_done, _ = self.eval_env.step(
                eval_action * np.abs(self.action_space.low))

            if self.render_eval:
                self.eval_env.render()

            eval_episode_reward += eval_r
            eval_qs.append(eval_q)

            if eval_done:
                self.eval_obs = self.eval_env.reset()
                eval_episode_rewards.append(eval_episode_reward)
                eval_episode_reward = 0.

        return eval_episode_rewards, eval_qs

    def _log_stats(self,
                   file_path,
                   start_time,
                   eval_episode_rewards,
                   eval_qs):
        """TODO

        :param start_time:
        :param eval_episode_rewards:
        :param eval_qs:
        :return:
        """
        rank = MPI.COMM_WORLD.Get_rank()
        mpi_size = MPI.COMM_WORLD.Get_size()

        # Log statistics.
        duration = time.time() - start_time
        stats = self._get_stats()
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
        if len(self.epoch_adaptive_distances) != 0:
            combined_stats['train/param_noise_distance'] = np.mean(
                self.epoch_adaptive_distances)
        combined_stats['total/duration'] = duration
        combined_stats['total/steps_per_second'] = float(self.step) \
            / float(duration)
        combined_stats['total/episodes'] = self.episodes
        combined_stats['rollout/episodes'] = self.epoch_episodes
        combined_stats['rollout/actions_std'] = np.std(self.epoch_actions)

        # Evaluation statistics.
        if self.eval_env is not None:
            combined_stats['eval/return'] = np.mean(eval_episode_rewards)
            combined_stats['eval/return_history'] = np.mean(
                self.eval_episode_rewards_history)
            combined_stats['eval/Q'] = np.mean(eval_qs)
            combined_stats['eval/episodes'] = len(eval_episode_rewards)

        combined_stats_sums = MPI.COMM_WORLD.allreduce(
            np.array([as_scalar(x)
                      for x in combined_stats.values()]))
        combined_stats = {
            k: v / mpi_size for (k, v) in
            zip(combined_stats.keys(), combined_stats_sums)
        }

        # Total statistics.
        combined_stats['total/epochs'] = self.epoch + 1
        combined_stats['total/steps'] = self.step

        # Save combined_stats in a csv file.
        if file_path is not None:
            exists = os.path.exists(file_path)
            with open(file_path, 'a') as f:
                w = csv.DictWriter(
                    f, fieldnames=combined_stats.keys())
                if not exists:
                    w.writeheader()

                w.writerow(combined_stats)

        # Print statistics.
        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])
        if len(self.episode_successes) > 0:
            logger.logkv("success rate",
                         np.mean(self.episode_successes[-100:]))
        logger.dump_tabular()
        logger.info('')
        logdir = logger.get_dir()
        if rank == 0 and logdir:
            if hasattr(self.env, 'get_state'):
                with open(os.path.join(logdir, 'env_state.pkl'), 'wb') \
                        as file_handler:
                    pickle.dump(self.env.get_state(), file_handler)

            if self.eval_env and hasattr(self.eval_env, 'get_state'):
                with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') \
                        as file_handler:
                    pickle.dump(self.eval_env.get_state(), file_handler)
