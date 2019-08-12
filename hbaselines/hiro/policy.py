"""TD3-compatible policies."""
import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from numpy.random import normal
from functools import reduce
import logging
from gym.spaces import Box
import random

from hbaselines.hiro.tf_util import normalize, denormalize, flatgrad
from hbaselines.hiro.tf_util import get_trainable_vars, get_target_updates
from hbaselines.hiro.tf_util import reduce_std
from hbaselines.hiro.replay_buffer import ReplayBuffer, HierReplayBuffer
from hbaselines.common.reward_fns import negative_distance
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.mpi_running_mean_std import RunningMeanStd


class ActorCriticPolicy(object):
    """Base Actor Critic Policy.

    Attributes
    ----------
    sess : tf.Session
        the current TensorFlow session
    ob_space : gym.space.*
        the observation space of the environment
    ac_space : gym.space.*
        the action space of the environment
    co_space : gym.space.*
        the context space of the environment
    """

    def __init__(self, sess, ob_space, ac_space, co_space):
        """Instantiate the base policy object.

        Parameters
        ----------
        sess : tf.Session
            the current TensorFlow session
        ob_space : gym.space.*
            the observation space of the environment
        ac_space : gym.space.*
            the action space of the environment
        co_space : gym.space.*
            the context space of the environment
        """
        self.sess = sess
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.co_space = co_space

    def initialize(self):
        """Initialize the policy.

        This is used at the beginning of training by the algorithm, after the
        model parameters have been initialized.
        """
        raise NotImplementedError

    def update(self):
        """Perform a gradient update step.

        Returns
        -------
        float
            critic loss
        float
            actor loss
        """
        raise NotImplementedError

    def get_action(self, obs, apply_noise=False, **kwargs):
        """Call the actor methods to compute policy actions.

        Parameters
        ----------
        obs : array_like
            the observation
        apply_noise : bool, optional
            whether to add Gaussian noise to the output of the actor. Defaults
            to False

        Returns
        -------
        array_like
            computed action by the policy
        """
        raise NotImplementedError

    def value(self, obs, action=None, with_actor=True, **kwargs):
        """Call the critic methods to compute the value.

        Parameters
        ----------
        obs : array_like
            the observation
        action : array_like, optional
            the actions performed in the given observation
        with_actor : bool, optional
            specifies whether to use the actor when computing the values. In
            this case, the actions are computed directly from the actor, and
            the input actions are not used.

        Returns
        -------
        array_like
            computed value by the critic
        """
        raise NotImplementedError

    def store_transition(self, obs0, action, reward, obs1, done, **kwargs):
        """Store a transition in the replay buffer.

        Parameters
        ----------
        obs0 : array_like
            the last observation
        action : array_like
            the action
        reward : float
            the reward
        obs1 : array_like
            the current observation
        done : float
            is the episode done
        """
        raise NotImplementedError

    def get_stats(self):
        """Return the model statistics.

        This data wil be stored in the training csv file.

        Returns
        -------
        dict
            model statistic
        """
        raise NotImplementedError

    def get_td_map(self):
        """Return dict map for the summary (to be run in the algorithm)."""
        raise NotImplementedError


class FeedForwardPolicy(ActorCriticPolicy):
    """Feed-forward neural network actor-critic policy.

    Attributes
    ----------
    sess : tf.Session
        the current TensorFlow session
    ob_space : gym.space.*
        the observation space of the environment
    ac_space : gym.space.*
        the action space of the environment
    co_space : gym.space.*
        the context space of the environment
    buffer_size : int
        the max number of transitions to store
    batch_size : int
        SGD batch size
    actor_lr : float
        actor learning rate
    critic_lr : float
        critic learning rate
    clip_norm : float
        clip the gradients (disabled if None)
    critic_l2_reg : float
        l2 regularizer coefficient
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    reuse : bool
        if the policy is reusable or not
    layers : list of int
        the size of the Neural network for the policy
    tau : float
        target update rate
    gamma : float
        discount factor
    noise : float
        scaling term to the range of the action space, that is subsequently
        used as the standard deviation of Gaussian noise added to the action if
        `apply_noise` is set to True in `get_action`
    layer_norm : bool
        enable layer normalisation
    normalize_observations : bool
        should the observation be normalized
    normalize_returns : bool
        should the critic output be normalized
    return_range : (float, float)
        the bounding values for the critic output
    activ : tf.nn.*
        the activation function to use in the neural network
    replay_buffer : hbaselines.hiro.replay_buffer.ReplayBuffer
        the replay buffer
    critic_target : tf.placeholder
        a placeholder for the current-step estimate of the target Q values
    terminals1 : tf.placeholder
        placeholder for the next step terminals
    rew_ph : tf.placeholder
        placeholder for the rewards
    action_ph : tf.placeholder
        placeholder for the actions
    obs_ph : tf.placeholder
        placeholder for the observations
    obs1_ph : tf.placeholder
        placeholder for the next step observations
    obs_rms : stable_baselines.common.mpi_running_mean_std.RunningMeanStd
        an object that computes the running mean and standard deviations for
        the observations
    ret_rms : stable_baselines.common.mpi_running_mean_std.RunningMeanStd
        an object that computes the running mean and standard deviations for
        the rewards
    actor_tf : tf.Variable
        the output from the actor network
    normalized_critic_tf : tf.Variable
        normalized output from the critic
    normalized_critic_with_actor_tf : tf.Variable
        normalized output from the critic with the action provided directly by
        the actor policy
    critic_tf : tf.Variable
        de-normalized output from the critic
    critic_with_actor_tf : tf.Variable
        de-normalized output from the critic with the action provided directly
        by the actor policy
    target_q : tf.Variable
        the Q-value as estimated by the current reward and next step estimate
        by the target Q-value
    target_init_updates : tf.Operation
        an operation that sets the values of the trainable parameters of the
        target actor/critic to match those actual actor/critic
    target_soft_updates : tf.Operation
        soft target update function
    actor_loss : tf.Operation
        the operation that returns the loss of the actor
    actor_grads : tf.Operation
        the operation that returns the gradients of the trainable parameters of
        the actor
    actor_optimizer : tf.Operation
        the operation that updates the trainable parameters of the actor
    critic_loss : tf.Operation
        the operation that returns the loss of the critic
    critic_grads : tf.Operation
        the operation that returns the gradients of the trainable parameters of
        the critic
    critic_optimizer : tf.Operation
        the operation that updates the trainable parameters of the critic
    stats_sample : dict
        a batch of samples to compute model means and stds from
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 co_space,
                 buffer_size,
                 batch_size,
                 actor_lr,
                 critic_lr,
                 clip_norm,
                 critic_l2_reg,
                 verbose,
                 tau,
                 gamma,
                 normalize_observations,
                 normalize_returns,
                 return_range,
                 noise=0.05,
                 layer_norm=False,
                 reuse=False,
                 layers=None,
                 act_fun=tf.nn.relu,
                 scope=None):
        """Instantiate the feed-forward neural network policy.

        Parameters
        ----------
        sess : tf.Session
            the current TensorFlow session
        ob_space : gym.space.*
            the observation space of the environment
        ac_space : gym.space.*
            the action space of the environment
        co_space : gym.space.*
            the context space of the environment
        buffer_size : int
            the max number of transitions to store
        batch_size : int
            SGD batch size
        actor_lr : float
            actor learning rate
        critic_lr : float
            critic learning rate
        clip_norm : float
            clip the gradients (disabled if None)
        critic_l2_reg : float
            l2 regularizer coefficient
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        tau : float
            target update rate
        gamma : float
            discount factor
        normalize_observations : bool
            should the observation be normalized
        normalize_returns : bool
            should the critic output be normalized
        return_range : (float, float)
            the bounding values for the critic output
        noise : float, optional
            scaling term to the range of the action space, that is subsequently
            used as the standard deviation of Gaussian noise added to the
            action if `apply_noise` is set to True in `get_action`. Defaults to
            0.05, i.e. 5% of action range.
        layer_norm : bool
            enable layer normalisation
        reuse : bool
            if the policy is reusable or not
        layers : list of int or None
            the size of the Neural network for the policy (if None, default to
            [64, 64])
        act_fun : tf.nn.*
            the activation function to use in the neural network
        scope : str
            an upper-level scope term. Used by policies that call this one.

        Raises
        ------
        AssertionError
            if the layers is not a list of at least size 1
        """
        super(FeedForwardPolicy, self).__init__(sess,
                                                ob_space, ac_space, co_space)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.critic_l2_reg = critic_l2_reg
        self.verbose = verbose
        self.reuse = reuse
        self.layers = layers or [300, 300]
        self.tau = tau
        self.gamma = gamma
        self.noise = noise
        self.layer_norm = layer_norm
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.return_range = return_range
        self.activ = act_fun
        assert len(self.layers) >= 1, \
            "Error: must have at least one hidden layer for the policy."

        # =================================================================== #
        # Step 1: Create a replay buffer object.                              #
        # =================================================================== #

        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # =================================================================== #
        # Step 2: Create input variables.                                     #
        # =================================================================== #

        # Compute the shape of the input observation space, which may include
        # the contextual term.
        ob_dim = ob_space.shape
        if co_space is not None:
            ob_dim = tuple(map(sum, zip(ob_dim, co_space.shape)))

        with tf.variable_scope("input", reuse=False):
            self.critic_target = tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='critic_target')
            self.terminals1 = tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='terminals1')
            self.rew_ph = tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='rewards')
            self.action_ph = tf.placeholder(
                tf.float32,
                shape=(None,) + ac_space.shape,
                name='actions')
            self.obs_ph = tf.placeholder(
                tf.float32,
                shape=(None,) + ob_dim,
                name='observations')
            self.obs1_ph = tf.placeholder(
                tf.float32,
                shape=(None,) + ob_dim,
                name='observations')

        # logging of rewards to tensorboard
        with tf.variable_scope("input_info", reuse=False):
            tf.summary.scalar('rewards', tf.reduce_mean(self.rew_ph))

        # =================================================================== #
        # Step 3: Additional (optional) normalizing terms.                    #
        # =================================================================== #

        # Observation normalization.
        if normalize_observations:
            with tf.variable_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=ob_dim)
        else:
            self.obs_rms = None

        obs_high = self.ob_space.high
        obs_low = self.ob_space.low
        if co_space is not None:
            obs_high = np.append(obs_high, self.co_space.high)
            obs_low = np.append(obs_low, self.co_space.low)

        # Clip the observations by their min/max values.
        normalized_obs0 = tf.clip_by_value(
            normalize(self.obs_ph, self.obs_rms), obs_low, obs_high)
        normalized_obs1 = tf.clip_by_value(
            normalize(self.obs1_ph, self.obs_rms), obs_low, obs_high)

        # Return normalization.
        if normalize_returns:
            with tf.variable_scope('ret_rms'):
                self.ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None

        # =================================================================== #
        # Step 4: Create actor and critic variables.                          #
        # =================================================================== #

        # Create networks and core TF parts that are shared across setup parts.
        with tf.variable_scope("model", reuse=False):
            self.actor_tf = self.make_actor(normalized_obs0)
            self.normalized_critic_tf = [
                self.make_critic(normalized_obs0, self.action_ph,
                                 scope="qf_{}".format(i))
                for i in range(2)
            ]
            self.normalized_critic_with_actor_tf = [
                self.make_critic(normalized_obs0, self.actor_tf, reuse=True,
                                 scope="qf_{}".format(i))
                for i in range(2)
            ]

        with tf.variable_scope("target", reuse=False):
            actor_target = self.make_actor(normalized_obs1)
            critic_target = [
                self.make_critic(normalized_obs1, actor_target,
                                 scope="qf_{}".format(i))
                for i in range(2)
            ]

        with tf.variable_scope("loss", reuse=False):
            self.critic_tf = [
                denormalize(tf.clip_by_value(
                    critic, return_range[0], return_range[1]), self.ret_rms)
                for critic in self.normalized_critic_tf
            ]

            self.critic_with_actor_tf = [
                denormalize(tf.clip_by_value(
                    critic, return_range[0], return_range[1]), self.ret_rms)
                for critic in self.normalized_critic_with_actor_tf
            ]

            q_obs1 = tf.reduce_min(
                [denormalize(critic_target[0], self.ret_rms),
                 denormalize(critic_target[1], self.ret_rms)],
                axis=0
            )
            self.target_q = self.rew_ph + (1-self.terminals1) * gamma * q_obs1

            tf.summary.scalar('critic_target', tf.reduce_mean(self.target_q))

        # Create the target update operations.
        model_scope = 'model/'
        target_scope = 'target/'
        if scope is not None:
            model_scope = scope + '/' + model_scope
            target_scope = scope + '/' + target_scope
        init_updates, soft_updates = get_target_updates(
            get_trainable_vars(model_scope),
            get_trainable_vars(target_scope),
            tau, verbose)
        self.target_init_updates = init_updates
        self.target_soft_updates = soft_updates

        # =================================================================== #
        # Step 5: Setup the optimizers for the actor and critic.              #
        # =================================================================== #

        with tf.variable_scope("Adam_mpi", reuse=False):
            self._setup_actor_optimizer(scope=scope)
            self._setup_critic_optimizer(scope=scope)
            tf.summary.scalar('actor_loss', self.actor_loss)
            tf.summary.scalar('critic_loss', self.critic_loss)

        # =================================================================== #
        # Step 6: Setup the operations for computing model statistics.        #
        # =================================================================== #

        self.stats_sample = None

        # Setup the running means and standard deviations of the model inputs
        # and outputs.
        self.stats_ops, self.stats_names = self._setup_stats(scope or "Model")

    def _setup_actor_optimizer(self, scope):
        """Create the actor loss, gradient, and optimizer."""
        if self.verbose >= 2:
            logging.info('setting up actor optimizer')

        scope_name = 'model/pi/'
        if scope is not None:
            scope_name = scope + '/' + scope_name

        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf[0])
        if self.verbose >= 2:
            actor_shapes = [var.get_shape().as_list()
                            for var in get_trainable_vars(scope_name)]
            actor_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                   for shape in actor_shapes])
            logging.info('  actor shapes: {}'.format(actor_shapes))
            logging.info('  actor params: {}'.format(actor_nb_params))

        self.actor_grads = flatgrad(
            self.actor_loss,
            get_trainable_vars(scope_name),
            clip_norm=self.clip_norm)

        self.actor_optimizer = MpiAdam(
            var_list=get_trainable_vars(scope_name),
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def _setup_critic_optimizer(self, scope):
        """Create the critic loss, gradient, and optimizer."""
        if self.verbose >= 2:
            logging.info('setting up critic optimizer')

        normalized_critic_target_tf = tf.clip_by_value(
            normalize(self.target_q, self.ret_rms),
            self.return_range[0],
            self.return_range[1])

        self.critic_loss = sum(
            tf.losses.huber_loss(
                self.normalized_critic_tf[i], normalized_critic_target_tf)
            for i in range(2)
        )

        if self.critic_l2_reg > 0.:
            critic_reg_vars = []
            for i in range(2):
                scope_name = 'model/qf_{}/'.format(i)
                if scope is not None:
                    scope_name = scope + '/' + scope_name

                critic_reg_vars += [
                    var for var in get_trainable_vars(scope_name)
                    if 'bias' not in var.name
                    and 'qf_output' not in var.name
                    and 'b' not in var.name
                ]

            if self.verbose >= 2:
                for var in critic_reg_vars:
                    logging.info('  regularizing: {}'.format(var.name))
                logging.info('  applying l2 regularization with {}'.format(
                    self.critic_l2_reg))

            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg

        self.critic_grads = []
        self.critic_optimizer = []

        for i in range(2):
            scope_name = 'model/qf_{}/'.format(i)
            if scope is not None:
                scope_name = scope + '/' + scope_name

            if self.verbose >= 2:
                critic_shapes = [var.get_shape().as_list()
                                 for var in get_trainable_vars(scope_name)]
                critic_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                        for shape in critic_shapes])
                logging.info('  critic shapes: {}'.format(critic_shapes))
                logging.info('  critic params: {}'.format(critic_nb_params))

            self.critic_grads.append(
                flatgrad(self.critic_loss,
                         get_trainable_vars(scope_name),
                         clip_norm=self.clip_norm)
            )

            self.critic_optimizer.append(
                MpiAdam(var_list=get_trainable_vars(scope_name),
                        beta1=0.9, beta2=0.999, epsilon=1e-08)
            )

    def make_actor(self, obs, reuse=False, scope="pi"):
        """Create an actor tensor.

        Parameters
        ----------
        obs : tf.placeholder
            the input observation placeholder
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor

        Returns
        -------
        tf.Variable
            the output from the actor
        """
        with tf.variable_scope(scope, reuse=reuse):
            # flatten the input placeholder
            pi_h = tf.layers.flatten(obs)

            # create the hidden layers
            for i, layer_size in enumerate(self.layers):
                pi_h = tf.layers.dense(pi_h, layer_size, name='fc' + str(i))
                if self.layer_norm:
                    pi_h = tf.contrib.layers.layer_norm(
                        pi_h, center=True, scale=True)
                pi_h = self.activ(pi_h)

            # create the output layer
            if any(np.isinf(self.ac_space.high)) \
                    or any(np.isinf(self.ac_space.low)):
                # no nonlinearity to the output if the action space is not
                # bounded
                policy = tf.layers.dense(
                    pi_h,
                    self.ac_space.shape[0],
                    name='output',
                    kernel_initializer=tf.random_uniform_initializer(
                        minval=-3e-3, maxval=3e-3))
            else:
                # tanh nonlinearity with an added offset and scale is the
                # action space is bounded
                policy = tf.nn.tanh(tf.layers.dense(
                    pi_h,
                    self.ac_space.shape[0],
                    name='output',
                    kernel_initializer=tf.random_uniform_initializer(
                        minval=-3e-3, maxval=3e-3)))

                # scaling terms to the output from the actor
                action_means = tf.expand_dims(
                    tf.constant((self.ac_space.high + self.ac_space.low) / 2.,
                                dtype=tf.float32),
                    0
                )
                action_magnitudes = tf.expand_dims(
                    tf.constant((self.ac_space.high - self.ac_space.low) / 2.,
                                dtype=tf.float32),
                    0
                )

                policy = tf.add(action_means,
                                tf.multiply(action_magnitudes, policy))

        return policy

    def make_critic(self, obs, action, reuse=False, scope="qf"):
        """Create a critic tensor.

        Parameters
        ----------
        obs : tf.placeholder
            the input observation placeholder
        action : tf.placeholder
            the input action placeholder
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor

        Returns
        -------
        tf.Variable
            the output from the critic
        """
        with tf.variable_scope(scope, reuse=reuse):
            # flatten the input placeholder
            qf_h = tf.layers.flatten(obs)
            qf_h = tf.concat([qf_h, action], axis=-1)

            # create the hidden layers
            for i, layer_size in enumerate(self.layers):
                qf_h = tf.layers.dense(qf_h, layer_size, name='fc' + str(i))
                if self.layer_norm:
                    qf_h = tf.contrib.layers.layer_norm(
                        qf_h, center=True, scale=True)
                qf_h = self.activ(qf_h)

            # create the output layer
            qvalue_fn = tf.layers.dense(
                qf_h,
                1,
                name='qf_output',
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3))

        return qvalue_fn

    def update(self):
        """See parent class."""
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample(self.batch_size):
            return 0, 0

        # Get a batch
        obs0, actions, rewards, obs1, terminals1 = self.replay_buffer.sample(
            batch_size=self.batch_size)

        return self.update_from_batch(obs0, actions, rewards, obs1, terminals1)

    def update_from_batch(self, obs0, actions, rewards, obs1, terminals1):
        """Perform gradient update step given a batch of data.

        Parameters
        ----------
        obs0 : np.ndarray
            batch of observations
        actions : numpy float
            batch of actions executed given obs_batch
        rewards : numpy float
            rewards received as results of executing act_batch
        obs1 : np.ndarray
            next set of observations seen after executing act_batch
        terminals1 : numpy bool
            done_mask[i] = 1 if executing act_batch[i] resulted in the end of
            an episode and 0 otherwise.

        Returns
        -------
        float
            critic loss
        float
            actor loss
        """
        # Reshape to match previous behavior and placeholder shape.
        rewards = rewards.reshape(-1, 1)
        terminals1 = terminals1.reshape(-1, 1)

        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads[0],
               self.critic_grads[1], self.critic_loss]
        td_map = {
            self.obs_ph: obs0,
            self.action_ph: actions,
            self.rew_ph: rewards,
            self.obs1_ph: obs1,
            self.terminals1: terminals1
        }

        actor_grads, actor_loss, grads_0, grads_1, critic_loss = self.sess.run(
            ops, td_map)

        self.actor_optimizer.update(actor_grads, learning_rate=self.actor_lr)
        self.critic_optimizer[0].update(grads_0, learning_rate=self.critic_lr)
        self.critic_optimizer[1].update(grads_1, learning_rate=self.critic_lr)

        # Run target soft update operation.
        self.sess.run(self.target_soft_updates)

        return critic_loss, actor_loss

    def get_action(self, obs, apply_noise=False, **kwargs):
        """See parent class."""
        # Add the contextual observation, if applicable.
        context_obs = kwargs.get("context_obs")
        if context_obs[0] is not None:
            obs = np.concatenate((obs, context_obs), axis=1)

        action = self.sess.run(self.actor_tf, {self.obs_ph: obs})
        if apply_noise:
            noise = self.noise * (self.ac_space.high - self.ac_space.low) / 2
            action += normal(loc=0, scale=noise, size=action.shape)
        action = np.clip(action, self.ac_space.low, self.ac_space.high)

        return action

    def value(self, obs, action=None, with_actor=True, **kwargs):
        """See parent class."""
        # Add the contextual observation, if applicable.
        context_obs = kwargs.get("context_obs")
        if context_obs[0] is not None:
            obs = np.concatenate((obs, context_obs), axis=1)

        if with_actor:
            return self.sess.run(
                self.critic_with_actor_tf,
                feed_dict={self.obs_ph: obs})
        else:
            return self.sess.run(
                self.critic_tf,
                feed_dict={self.obs_ph: obs, self.action_ph: action})

    def store_transition(self, obs0, action, reward, obs1, done, **kwargs):
        """See parent class."""
        # Add the contextual observation, if applicable.
        if kwargs.get("context_obs0") is not None:
            obs0 = np.concatenate(
                (obs0, kwargs["context_obs0"].flatten()), axis=0)
        if kwargs.get("context_obs1") is not None:
            obs1 = np.concatenate(
                (obs1, kwargs["context_obs1"].flatten()), axis=0)

        self.replay_buffer.add(obs0, action, reward, obs1, float(done))
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs0]))

    def initialize(self):
        """See parent class.

        This method syncs the actor and critic optimizers across CPUs, and
        initializes the target parameters to match the model parameters.
        """
        self.actor_optimizer.sync()
        for i in range(2):
            self.critic_optimizer[i].sync()
        self.sess.run(self.target_init_updates)

    def _setup_stats(self, base="Model"):
        """Create the running means and std of the model inputs and outputs.

        This method also adds the same running means and stds as scalars to
        tensorboard for additional storage.
        """
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['{}/ret_rms_mean'.format(base),
                      '{}/ret_rms_std'.format(base)]

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean),
                    tf.reduce_mean(self.obs_rms.std)]
            names += ['{}/obs_rms_mean'.format(base),
                      '{}/obs_rms_std'.format(base)]

        ops += [tf.reduce_mean(self.critic_tf[0])]
        names += ['{}/reference_Q1_mean'.format(base)]
        ops += [reduce_std(self.critic_tf[0])]
        names += ['{}/reference_Q1_std'.format(base)]

        ops += [tf.reduce_mean(self.critic_tf[1])]
        names += ['{}/reference_Q2_mean'.format(base)]
        ops += [reduce_std(self.critic_tf[1])]
        names += ['{}/reference_Q2_std'.format(base)]

        ops += [tf.reduce_mean(self.critic_with_actor_tf[0])]
        names += ['{}/reference_actor_Q1_mean'.format(base)]
        ops += [reduce_std(self.critic_with_actor_tf[0])]
        names += ['{}/reference_actor_Q1_std'.format(base)]

        ops += [tf.reduce_mean(self.critic_with_actor_tf[1])]
        names += ['{}/reference_actor_Q2_mean'.format(base)]
        ops += [reduce_std(self.critic_with_actor_tf[1])]
        names += ['{}/reference_actor_Q2_std'.format(base)]

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['{}/reference_action_mean'.format(base)]
        ops += [reduce_std(self.actor_tf)]
        names += ['{}/reference_action_std'.format(base)]

        # Add all names and ops to the tensorboard summary.
        for op, name in zip(ops, names):
            tf.summary.scalar(name, op)

        return ops, names

    def get_stats(self):
        """See parent class.

        Get the mean and standard dev of the model's inputs and outputs.

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
            self.action_ph: self.stats_sample['actions']
        }

        for placeholder in [self.action_ph]:
            if placeholder is not None:
                feed_dict[placeholder] = self.stats_sample['actions']

        for placeholder in [self.obs_ph, self.obs1_ph]:
            if placeholder is not None:
                feed_dict[placeholder] = self.stats_sample['obs0']

        values = self.sess.run(self.stats_ops, feed_dict=feed_dict)

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        return stats

    def get_td_map(self):
        """See parent class."""
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample(self.batch_size):
            return {}

        # Get a batch.
        obs0, actions, rewards, obs1, terminals1 = self.replay_buffer.sample(
            batch_size=self.batch_size)

        return self.get_td_map_from_batch(
            obs0, actions, rewards, obs1, terminals1)

    def get_td_map_from_batch(self, obs0, actions, rewards, obs1, terminals1):
        """Convert a batch to a td_map."""
        # Reshape to match previous behavior and placeholder shape.
        rewards = rewards.reshape(-1, 1)
        terminals1 = terminals1.reshape(-1, 1)

        td_map = {
            self.obs_ph: obs0,
            self.action_ph: actions,
            self.rew_ph: rewards,
            self.obs1_ph: obs1,
            self.terminals1: terminals1
        }

        return td_map


class GoalDirectedPolicy(ActorCriticPolicy):
    """Goal-directed hierarchical reinforcement learning model.

    This policy is an implementation of the two-level hierarchy presented
    in [1], which itself is similar to the feudal networks formulation [2, 3].
    This network consists of a high-level, or Manager, pi_{\theta_H} that
    computes and outputs goals g_t ~ pi_{\theta_H}(s_t, h) every meta_period
    time steps, and a low-level policy pi_{\theta_L} that takes as inputs the
    current state and the assigned goals and attempts to perform an action
    a_t ~ pi_{\theta_L}(s_t,g_t) that satisfies these goals.

    The Manager is rewarded based on the original environment reward function:
    r_H = r(s,a;h).

    The Target term, h, parameterizes the reward assigned to the Manager in
    order to allow the policy to generalize to several goals within a task, a
    technique that was first proposed by [4].

    Finally, the Worker is motivated to follow the goals set by the Manager via
    an intrinsic reward based on the distance between the current observation
    and the goal observation: r_L (s_t, g_t, s_{t+1}) = ||s_t + g_t - s_{t+1}||

    Bibliography:

    [1] Nachum, Ofir, et al. "Data-efficient hierarchical reinforcement
        learning." Advances in Neural Information Processing Systems. 2018.
    [2] Dayan, Peter, and Geoffrey E. Hinton. "Feudal reinforcement learning."
        Advances in neural information processing systems. 1993.
    [3] Vezhnevets, Alexander Sasha, et al. "Feudal networks for hierarchical
        reinforcement learning." Proceedings of the 34th International
        Conference on Machine Learning-Volume 70. JMLR. org, 2017.
    [4] Schaul, Tom, et al. "Universal value function approximators."
        International Conference on Machine Learning. 2015.

    Attributes
    ----------
    manager : hbaselines.hiro.policy.FeedForwardPolicy
        the manager policy
    meta_period : int
        manger action period
    relative_goals : bool
        specifies whether the goal issued by the Manager is meant to be a
        relative or absolute goal, i.e. specific state or change in state
    off_policy_corrections : bool
        whether to use off-policy corrections during the update procedure. See:
        https://arxiv.org/abs/1805.08296.
    use_fingerprints : bool
        specifies whether to add a time-dependent fingerprint to the
        observations
    fingerprint_range : (list of float, list of float)
        the low and high values for each fingerprint element, if they are being
        used
    fingerprint_dim : tuple of int
        the shape of the fingerprint elements, if they are being used
    centralized_value_functions : bool
        specifies whether to use centralized value functions for the Manager
        and Worker critic functions
    connected_gradients : bool
        whether to connect the graph between the manager and worker
    prev_meta_obs : array_like
        previous observation by the Manager
    meta_action : array_like
        current action by the Manager
    meta_reward : float
        current meta reward, counting as the cumulative environment reward
        during the meta period
    batch_size : int
        SGD batch size
    worker : hbaselines.hiro.policy.FeedForwardPolicy
        the worker policy
    worker_reward : function
        reward function for the worker
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 co_space,
                 buffer_size,
                 batch_size,
                 actor_lr,
                 critic_lr,
                 clip_norm,
                 critic_l2_reg,
                 verbose,
                 tau,
                 gamma,
                 normalize_observations,
                 normalize_returns,
                 return_range,
                 noise=0.05,
                 layer_norm=False,
                 reuse=False,
                 layers=None,
                 act_fun=tf.nn.relu,
                 meta_period=10,
                 relative_goals=False,
                 off_policy_corrections=False,
                 use_fingerprints=False,
                 fingerprint_range=([0], [5]),
                 centralized_value_functions=False,
                 connected_gradients=False):
        """Instantiate the goal-directed hierarchical policy.

        Parameters
        ----------
        sess : tf.Session
            the current TensorFlow session
        ob_space : gym.space.*
            the observation space of the environment
        ac_space : gym.space.*
            the action space of the environment
        co_space : gym.space.*
            the context space of the environment
        buffer_size : int
            the max number of transitions to store
        batch_size : int
            SGD batch size
        actor_lr : float
            actor learning rate
        critic_lr : float
            critic learning rate
        clip_norm : float
            clip the gradients (disabled if None)
        critic_l2_reg : float
            l2 regularizer coefficient
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        tau : float
            target update rate
        gamma : float
            discount factor
        normalize_observations : bool
            should the observation be normalized
        normalize_returns : bool
            should the critic output be normalized
        return_range : (float, float)
            the bounding values for the critic output
        noise : float, optional
            scaling term to the range of the action space, that is subsequently
            used as the standard deviation of Gaussian noise added to the
            action if `apply_noise` is set to True in `get_action`. Defaults to
            0.05, i.e. 5% of action range.
        layer_norm : bool
            enable layer normalisation
        reuse : bool
            if the policy is reusable or not
        layers : list of int or None
            the size of the Neural network for the policy (if None, default to
            [64, 64])
        act_fun : tf.nn.*
            the activation function to use in the neural network
        meta_period : int, optional
            manger action period. Defaults to 10.
        relative_goals : bool, optional
            specifies whether the goal issued by the Manager is meant to be a
            relative or absolute goal, i.e. specific state or change in state
        off_policy_corrections : bool, optional
            whether to use off-policy corrections during the update procedure.
            See: https://arxiv.org/abs/1805.08296. Defaults to False.
        use_fingerprints : bool, optional
            specifies whether to add a time-dependent fingerprint to the
            observations
        fingerprint_range : (list of float, list of float), optional
            the low and high values for each fingerprint element, if they are
            being used
        centralized_value_functions : bool, optional
            specifies whether to use centralized value functions for the
            Manager and Worker critic functions
        connected_gradients : bool, optional
            whether to connect the graph between the manager and worker.
            Defaults to False.
        """
        super(GoalDirectedPolicy, self).__init__(sess,
                                                 ob_space, ac_space, co_space)

        self.meta_period = meta_period
        self.relative_goals = relative_goals
        self.off_policy_corrections = off_policy_corrections
        self.use_fingerprints = use_fingerprints
        self.fingerprint_range = fingerprint_range
        self.fingerprint_dim = (len(self.fingerprint_range[0]),)
        self.centralized_value_functions = centralized_value_functions
        self.connected_gradients = connected_gradients
        self.fingerprint_dim = (1,)
        self.fingerprint_range = ([0], [5])

        self.replay_buffer = HierReplayBuffer(int(buffer_size/meta_period))

        # =================================================================== #
        # Part 1. Setup the Manager                                           #
        # =================================================================== #

        # Compute the action space for the Manager. If the fingerprint terms
        # are being appended onto the observations, this should be removed from
        # the action space.
        # if self.use_fingerprints:
        #     low = np.array(ob_space.low)[:-self.fingerprint_dim[0]]
        #     high = ob_space.high[:-self.fingerprint_dim[0]]
        #     manager_ac_space = Box(low=low, high=high)
        # else:
        #     manager_ac_space = ob_space
        # FIXME: only for ant
        manager_ac_space = Box(
            low=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3, -0.5,
                          -0.3, -0.5, -0.3, -0.5, -0.3]),
            high=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5,
                           0.3, 0.5, 0.3])
        )

        # Create the Manager policy.
        with tf.variable_scope("Manager"):
            self.manager = FeedForwardPolicy(
                sess=sess,
                ob_space=ob_space,
                ac_space=manager_ac_space,
                co_space=co_space,
                buffer_size=buffer_size,
                batch_size=batch_size,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                clip_norm=clip_norm,
                critic_l2_reg=critic_l2_reg,
                verbose=verbose,
                tau=tau,
                gamma=gamma,
                normalize_observations=normalize_observations,
                normalize_returns=normalize_returns,
                return_range=return_range,
                layer_norm=layer_norm,
                reuse=reuse,
                layers=layers,
                act_fun=act_fun,
                scope="Manager",
                noise=noise,
            )

        # previous observation by the Manager
        self.prev_meta_obs = None

        # current action by the Manager
        self.meta_action = None

        # current meta reward, counting as the cumulative environment reward
        # during the meta period
        self.meta_reward = None

        # The following is redundant but necessary if the changes to the update
        # function are to be in the GoalDirected policy and not FeedForward.
        self.batch_size = batch_size

        # Use this to store a list of observations that stretch as long as the
        # dilated horizon chosen for the Manager. These observations correspond
        # to the s(t) in the HIRO paper.
        self._observations = []

        # Use this to store the list of environmental actions that the worker
        # takes. These actions correspond to the a(t) in the HIRO paper.
        self._worker_actions = []

        # rewards provided by the policy to the worker
        self._worker_rewards = []

        # done masks at every time step for the worker
        self._dones = []

        # =================================================================== #
        # Part 2. Setup the Worker                                            #
        # =================================================================== #

        # Create the Worker policy.
        with tf.variable_scope("Worker"):
            self.worker = FeedForwardPolicy(
                sess,
                ob_space=ob_space,
                ac_space=ac_space,
                co_space=manager_ac_space,
                buffer_size=buffer_size,
                batch_size=batch_size,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                clip_norm=clip_norm,
                critic_l2_reg=critic_l2_reg,
                verbose=verbose,
                tau=tau,
                gamma=gamma,
                normalize_observations=normalize_observations,
                normalize_returns=normalize_returns,
                return_range=return_range,
                layer_norm=layer_norm,
                reuse=reuse,
                layers=layers,
                act_fun=act_fun,
                scope="Worker",
                noise=noise,
            )

        # remove the last element to compute the reward FIXME
        # if self.use_fingerprints:
        #     state_indices = list(np.arange(
        #         0, self.manager.ob_space.shape[0] - self.fingerprint_dim[0]))
        # else:
        #     state_indices = None
        state_indices = list(np.arange(0, self.manager.ac_space.shape[0]))

        # reward function for the worker
        def worker_reward(states, goals, next_states):
            return negative_distance(
                states=states,
                state_indices=state_indices,
                goals=goals,
                next_states=next_states,
                relative_context=relative_goals,
                offset=0.0
            )
        self.worker_reward = worker_reward

    def initialize(self):
        """See parent class.

        This method calls the initialization methods of the manager and worker.
        """
        self.manager.initialize()
        self.worker.initialize()
        self.meta_reward = 0

    def update(self):
        """See parent class."""
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample(self.batch_size):
            return (0, 0), (0, 0)

        # Get a batch.
        samples = self.replay_buffer.sample(batch_size=self.batch_size)

        # Collect the relevant components of each sample.
        meta_obs0, meta_obs1, meta_act, meta_rew, meta_done, worker_obs0, \
            worker_obs1, worker_act, worker_rew, worker_done = \
            self._process_samples(samples)

        # Update the Manager policy.
        m_critic_loss, m_actor_loss = self.manager.update_from_batch(
            obs0=meta_obs0,
            actions=meta_act,
            rewards=meta_rew,
            obs1=meta_obs1,
            terminals1=meta_done
        )

        # Update the Worker policy.
        w_critic_loss, w_actor_loss = self.worker.update_from_batch(
            obs0=worker_obs0,
            actions=worker_act,
            rewards=worker_rew,
            obs1=worker_obs1,
            terminals1=worker_done
        )

        return (m_critic_loss, w_critic_loss), (m_actor_loss, w_actor_loss)

    @staticmethod
    def _process_samples(samples):
        """Convert the samples into a form that is usable for an update.

        Parameters
        ----------
        samples : list of tuple
            each element of the tuples consists of:

            * list of (numpy.ndarray, numpy.ndarray): the previous and next
              manager observations for each meta period
            * list of numpy.ndarray: the meta action (goal) for each meta
              period
            * list of float: the meta reward for each meta period
            * list of list of numpy.ndarray: all observations for the worker
              for each meta period
            * list of list of numpy.ndarray: all actions for the worker for
              each meta period
            * list of list of float: all rewards for the worker for each meta
              period
            * list of list of float: all done masks for the worker for each
              meta period. The last done mask corresponds to the done mask of
              the manager

        Returns
        -------
        numpy.ndarray
            (batch_size, meta_obs) matrix of meta observations
        numpy.ndarray
            (batch_size, meta_obs) matrix of next meta-period meta observations
        numpy.ndarray
            (batch_size, meta_ac) matrix of meta actions
        numpy.ndarray
            (batch_size,) vector of meta rewards
        numpy.ndarray
            (batch_size,) vector of meta done masks
        numpy.ndarray
            (batch_size, worker_obs) matrix of worker observations
        numpy.ndarray
            (batch_size, worker_obs) matrix of next step worker observations
        numpy.ndarray
            (batch_size, worker_ac) matrix of worker actions
        numpy.ndarray
            (batch_size,) vector of worker rewards
        numpy.ndarray
            (batch_size,) vector of worker done masks
        """
        meta_obs0_all = []
        meta_obs1_all = []
        meta_act_all = []
        meta_rew_all = []
        meta_done_all = []
        worker_obs0_all = []
        worker_obs1_all = []
        worker_act_all = []
        worker_rew_all = []
        worker_done_all = []

        for sample in samples:
            # Extract the elements of the sample.
            meta_obs, meta_action, meta_reward, worker_obses, worker_actions, \
                worker_rewards, worker_dones = sample

            # Separate the current and next step meta observations.
            meta_obs0, meta_obs1 = meta_obs

            # The meta done value corresponds to the last done value.
            meta_done = worker_dones[-1]

            # Sample one obs0/obs1/action/reward from the list of per-meta-
            # period variables.
            indx_val = random.randint(0, len(worker_obses)-2)
            worker_obs0 = worker_obses[indx_val]
            worker_obs1 = worker_obses[indx_val + 1]
            worker_action = worker_actions[indx_val]
            worker_reward = worker_rewards[indx_val]
            worker_done = worker_dones[indx_val]

            # Add the new sample to the list of returned samples.
            meta_obs0_all.append(np.array(meta_obs0, copy=False))
            meta_obs1_all.append(np.array(meta_obs1, copy=False))
            meta_act_all.append(np.array(meta_action, copy=False))
            meta_rew_all.append(np.array(meta_reward, copy=False))
            meta_done_all.append(np.array(meta_done, copy=False))
            worker_obs0_all.append(np.array(worker_obs0, copy=False))
            worker_obs1_all.append(np.array(worker_obs1, copy=False))
            worker_act_all.append(np.array(worker_action, copy=False))
            worker_rew_all.append(np.array(worker_reward, copy=False))
            worker_done_all.append(np.array(worker_done, copy=False))

        return np.array(meta_obs0_all), \
            np.array(meta_obs1_all), \
            np.array(meta_act_all), \
            np.array(meta_rew_all), \
            np.array(meta_done_all), \
            np.array(worker_obs0_all), \
            np.array(worker_obs1_all), \
            np.array(worker_act_all), \
            np.array(worker_rew_all), \
            np.array(worker_done_all)

    def get_action(self, obs, apply_noise=False, **kwargs):
        """See parent class."""
        # Update the meta action, if the time period requires is.
        if kwargs["time"] % self.meta_period == 0:
            self.meta_action = self.manager.get_action(
                obs, apply_noise, **kwargs)

        # Return the worker action.
        return self.worker.get_action(
            obs, apply_noise, context_obs=self.meta_action)

    def value(self, obs, action=None, with_actor=True, **kwargs):
        """See parent class."""
        return 0  # FIXME

    def store_transition(self, obs0, action, reward, obs1, done, **kwargs):
        """See parent class."""
        # Compute the worker reward and append it to the list of rewards.
        self._worker_rewards.append(
            self.worker_reward(obs0, self.meta_action.flatten(), obs1)
        )

        # Add the environmental observations and done masks, and the worker
        # actions to their respective lists.
        self._worker_actions.append(action)
        self._observations.append(
            np.concatenate((obs0, self.meta_action.flatten()), axis=0))
        self._dones.append(done)

        # Increment the meta reward with the most recent reward.
        self.meta_reward += reward

        # Modify the previous meta observation whenever the action has changed.
        if kwargs["time"] % self.meta_period == 0:
            if kwargs.get("context_obs0") is not None:
                self.prev_meta_obs = np.concatenate(
                    (obs0, kwargs["context_obs0"].flatten()), axis=0)
            else:
                self.prev_meta_obs = np.copy(obs0)

        # Add a sample to the replay buffer.
        if (kwargs["time"] + 1) % self.meta_period == 0 or done:
            # Add the last observation if about to reset.
            if done:
                self._observations.append(
                    np.concatenate((obs1, self.meta_action.flatten()), axis=0))

            # Add the contextual observation, if applicable.
            if kwargs.get("context_obs1") is not None:
                meta_obs1 = np.concatenate(
                    (obs1, kwargs["context_obs1"].flatten()), axis=0)
            else:
                meta_obs1 = np.copy(obs1)

            # If this is the first time step, do not add the transition to the
            # meta replay buffer (it is not complete yet).
            if kwargs["time"] != 0:
                # Store a sample in the Manager policy.
                self.replay_buffer.add(
                    obs_t=self._observations,
                    goal_t=self.meta_action.flatten(),
                    action_t=self._worker_actions,
                    reward_t=self._worker_rewards,
                    done=self._dones,
                    meta_obs_t=(self.prev_meta_obs, meta_obs1),
                    meta_reward_t=self.meta_reward,
                )

                # Reset the meta reward.
                self.meta_reward = 0

                # Clear the worker rewards and actions, and the environmental
                # observation.
                self._observations = []
                self._worker_actions = []
                self._worker_rewards = []
                self._dones = []

    def _sample_best_meta_action(self,
                                 state_reps,
                                 next_state_reprs,
                                 prev_meta_actions,
                                 low_states,
                                 low_actions,
                                 low_state_reprs,
                                 k=8):
        """Return meta-actions that approximately maximize low-level log-probs.

        Parameters
        ----------
        state_reps : array_like
            current Manager state observation
        next_state_reprs : array_like
            next Manager state observation
        prev_meta_actions : array_like
            previous meta Manager action
        low_states : array_like
            current Worker state observation
        low_actions : array_like
            current Worker environmental action
        low_state_reprs : array_like
            current Worker state observation
        k : int, optional
            number of goals returned, excluding the initial goal and the mean
            value

        Returns
        -------
        array_like
            most likely meta-actions
        """
        # Collect several samples of potentially optimal goals.
        sampled_actions = self._sample(
            state_reps, next_state_reprs, k, prev_meta_actions)

        sampled_log_probs = tf.reshape(self._log_probs(
            tf.tile(low_states, [k, 1, 1]),
            tf.tile(low_actions, [k, 1, 1]),
            tf.tile(low_state_reprs, [k, 1, 1]),
            [tf.reshape(sampled_actions, [-1, sampled_actions.shape[-1]])]),
            [k, low_states.shape[0], low_states.shape[1], -1])

        fitness = tf.reduce_sum(sampled_log_probs, [2, 3])
        best_actions = tf.argmax(fitness, 0)
        best_goals = tf.gather_nd(
            sampled_actions,
            tf.stack([
                best_actions,
                tf.range(prev_meta_actions.shape[0], dtype=tf.int64)], -1))

        return best_goals

    def _log_probs(self, manager_obs, worker_obs, actions, goals):
        """Calculate the log probability of the next goal by the Manager.

        Parameters
        ----------
        manager_obs : array_like
            (batch_size, m_obs_dim) matrix of manager observations
        worker_obs : array_like
            (batch_size, w_obs_dim, meta_period) matrix of worker observations
        actions : array_like
            (batch_size, ac_dim, meta_period-1) list of low-level actions
        goals : array_like
            (batch_size, goal_dim, num_samples) matrix of sampled goals

        Returns
        -------
        array_like
            (batch_size, num_samples) error associated with every state /
            action / goal pair

        Helps
        -----
        * _sample_best_meta_action(self):
        """
        # Action a policy would perform given a specific observation / goal.
        pred_actions = self.worker.get_action(worker_obs, context_obs=goals)

        # Normalize the error based on the range of applicable goals.
        goal_space = self.manager.ac_space
        spec_range = goal_space.high - goal_space.low
        scale = np.tile(np.square(spec_range), (manager_obs.shape[0], 1))

        # Compute error as the distance between expected and actual actions.
        normalized_error = np.mean(
            np.square(np.divide(actions - pred_actions, scale)), axis=1)

        return -normalized_error

    def _sample(self, states, next_states, num_samples, orig_goals, sc=0.5):
        """Sample different goals.

        These goals are acquired from a random Gaussian distribution centered
        at s_{t+c} - s_t.

        Parameters
        ----------
        states : array_like
            (batch_size, obs_dim) matrix of current time step observation
        next_states : array_like
            (batch_size, obs_dim) matrix of next time step observation
        num_samples : int
            number of samples
        orig_goals : array_like
            (batch_size, goal_dim) matrix of original goal specified by Manager
        sc : float
            scaling factor for the normal distribution.

        Returns
        -------
        array_like
            (batch_size, goal_dim, num_samples) matrix of sampled goals

        Helps
        -----
        * _sample_best_meta_action(self)
        """
        batch_size, goal_dim = orig_goals.shape
        goal_space = self.manager.ac_space
        spec_range = goal_space.high - goal_space.low

        # Compute the mean and std for the Gaussian distribution to sample from
        loc = np.tile((next_states - states)[:, :goal_dim].flatten(),
                      (num_samples-2, 1))
        scale = np.tile(sc * spec_range / 2, (num_samples-2, batch_size))

        # Sample the requested number of goals from the Gaussian distribution.
        samples = loc + np.random.normal(
            size=(num_samples - 2, goal_dim * batch_size)) * scale

        # Add the original goal and the average of the original and final state
        # to the sampled goals.
        samples = np.vstack(
            [samples,
             (next_states - states)[:, :goal_dim].flatten(),
             orig_goals.flatten()],
        )

        # Clip the values based on the Manager action space range.
        minimum = np.tile(goal_space.low, (num_samples, batch_size))
        maximum = np.tile(goal_space.high, (num_samples, batch_size))
        samples = np.minimum(np.maximum(samples, minimum), maximum)

        # Reshape to (batch_size, goal_dim, num_samples).
        samples = samples.T.reshape((batch_size, goal_dim, num_samples))

        return samples

    def get_stats(self):
        """See parent class."""
        stats = {}
        # FIXME
        # stats.update(self.manager.get_stats())
        # stats.update(self.worker.get_stats())
        return stats

    def get_td_map(self):
        """See parent class."""
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample(self.batch_size):
            return {}

        # Get a batch.
        samples = self.replay_buffer.sample(batch_size=self.batch_size)

        # Collect the relevant components of each sample.
        meta_obs0, meta_obs1, meta_act, meta_rew, meta_done, worker_obs0, \
            worker_obs1, worker_act, worker_rew, worker_done = \
            self._process_samples(samples)

        td_map = {}
        td_map.update(self.manager.get_td_map_from_batch(
            meta_obs0, meta_act, meta_rew, meta_obs1, meta_done))
        td_map.update(self.worker.get_td_map_from_batch(
            worker_obs0, worker_act, worker_rew, worker_obs1, worker_done))

        return td_map
