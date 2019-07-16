import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from functools import reduce
from copy import deepcopy
import logging

import hbaselines.hiro.tf_util as tf_util
from hbaselines.hiro.replay_buffer import ReplayBuffer
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
        dict
            feed_dict map for the summary (to be run in the algorithm)
        """
        raise NotImplementedError

    def get_action(self, obs, **kwargs):
        """Call the actor methods to compute policy actions.

        Parameters
        ----------
        obs : array_like
            the observation

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
    layer_norm : bool
        enable layer normalisation
    normalize_observations : bool
        should the observation be normalized
    observation_range : (float, float)
        the bounding values for the observation
    normalize_returns : bool
        should the critic output be normalized
    return_range : (float, float)
        the bounding values for the critic output
    activ : tf.nn.*
        the activation function to use in the neural network
    replay_buffer : hbaselines.hiro.replay_buffer.ReplayBuffer
        the replay buffer
    critic_target : tf.placeholder
        TODO
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
    obs_rms : TODO
        an object that computes the running mean and standard deviations for
        the observations
    ret_rms : TODO
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
        TODO
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
                 observation_range,
                 normalize_returns,
                 return_range,
                 layer_norm=False,
                 reuse=False,
                 layers=None,
                 act_fun=tf.nn.relu,
                 scope=None):
        """Instantiate the feed-forward neural network policy.

        TODO: describe the scope and the summary.

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
        observation_range : (float, float)
            the bounding values for the observation
        normalize_returns : bool
            should the critic output be normalized
        return_range : (float, float)
            the bounding values for the critic output
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
        self.layer_norm = layer_norm
        self.normalize_observations = normalize_observations
        self.observation_range = observation_range
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
        if co_space is None:
            ob_dim = ob_space.shape
        else:
            ob_dim = tuple(map(sum, zip(ob_space.shape, co_space.shape)))

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

        normalized_obs0 = tf.clip_by_value(
            tf_util.normalize(self.obs_ph, self.obs_rms),
            observation_range[0], observation_range[1])
        normalized_obs1 = tf.clip_by_value(
            tf_util.normalize(self.obs1_ph, self.obs_rms),
            observation_range[0], observation_range[1])

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
            self.actor_tf = self._make_actor(normalized_obs0)
            self.normalized_critic_tf = self._make_critic(
                normalized_obs0,
                self.action_ph)
            self.normalized_critic_with_actor_tf = self._make_critic(
                normalized_obs0, self.actor_tf, reuse=True)

        with tf.variable_scope("target", reuse=False):
            actor_target = self._make_actor(normalized_obs1)
            critic_target = self._make_critic(normalized_obs1, actor_target)

        with tf.variable_scope("loss", reuse=False):
            self.critic_tf = tf_util.denormalize(
                tf.clip_by_value(
                    self.normalized_critic_tf,
                    return_range[0],
                    return_range[1]),
                self.ret_rms)

            self.critic_with_actor_tf = tf_util.denormalize(
                tf.clip_by_value(
                    self.normalized_critic_with_actor_tf,
                    return_range[0],
                    return_range[1]),
                self.ret_rms)

            q_obs1 = tf_util.denormalize(critic_target, self.ret_rms)
            self.target_q = self.rew_ph + (1-self.terminals1) * gamma * q_obs1

            tf.summary.scalar('critic_target',
                              tf.reduce_mean(self.critic_target))

        # TODO: do I need indent?
        # Create the target update operations.
        model_scope = 'model/'
        target_scope = 'target/'
        if scope is not None:
            model_scope = scope + '/' + model_scope
            target_scope = scope + '/' + target_scope
        init_updates, soft_updates = tf_util.get_target_updates(
            tf_util.get_trainable_vars(model_scope),
            tf_util.get_trainable_vars(target_scope),
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
        self.stats_ops, self.stats_names = self._setup_stats()

    def _setup_actor_optimizer(self, scope):
        """Create the actor loss, gradient, and optimizer."""
        if self.verbose >= 2:
            logging.info('setting up actor optimizer')

        scope_name = 'model/pi/'
        if scope is not None:
            scope_name = scope + '/' + scope_name

        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list()
                        for var in tf_util.get_trainable_vars(scope_name)]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape)
                               for shape in actor_shapes])
        if self.verbose >= 2:
            logging.info('  actor shapes: {}'.format(actor_shapes))
            logging.info('  actor params: {}'.format(actor_nb_params))

        self.actor_grads = tf_util.flatgrad(
            self.actor_loss,
            tf_util.get_trainable_vars(scope_name),
            clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(
            var_list=tf_util.get_trainable_vars(scope_name),
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def _setup_critic_optimizer(self, scope):
        """Create the critic loss, gradient, and optimizer."""
        if self.verbose >= 2:
            logging.info('setting up critic optimizer')

        normalized_critic_target_tf = tf.clip_by_value(
            tf_util.normalize(self.critic_target, self.ret_rms),
            self.return_range[0],
            self.return_range[1])

        self.critic_loss = tf.reduce_mean(tf.square(
            self.normalized_critic_tf - normalized_critic_target_tf))

        scope_name = 'model/qf/'
        if scope is not None:
            scope_name = scope + '/' + scope_name

        if self.critic_l2_reg > 0.:
            critic_reg_vars = [
                var for var in tf_util.get_trainable_vars(scope_name)
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

        critic_shapes = [var.get_shape().as_list()
                         for var in tf_util.get_trainable_vars(scope_name)]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                for shape in critic_shapes])

        if self.verbose >= 2:
            logging.info('  critic shapes: {}'.format(critic_shapes))
            logging.info('  critic params: {}'.format(critic_nb_params))

        self.critic_grads = tf_util.flatgrad(
            self.critic_loss,
            tf_util.get_trainable_vars(scope_name),
            clip_norm=self.clip_norm)

        self.critic_optimizer = MpiAdam(
            var_list=tf_util.get_trainable_vars(scope_name),
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def _make_actor(self,
                    obs=None,
                    reuse=False,
                    scope="pi"):
        """Create an actor tensor.

        Parameters
        ----------
        obs : tf.placeholder or None
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
        if obs is None:
            obs = self.obs_ph

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
            policy = tf.nn.tanh(tf.layers.dense(
                pi_h,
                self.ac_space.shape[0],
                name=scope,
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3)))

        # FIXME
        # # scaling terms to the output from the actor
        # action_means = tf.constant(
        #     (self.ac_space.high + self.ac_space.low) / 2.,
        #     dtype=tf.float32
        # )
        # action_magnitudes = tf.constant(
        #     (self.ac_space.high - self.ac_space.low) / 2.,
        #     dtype=tf.float32
        # )
        #
        # return action_means + action_magnitudes * policy

        return policy

    def _make_critic(self, obs=None, action=None, reuse=False, scope="qf"):
        """Create a critic tensor.

        Parameters
        ----------
        obs : tf.placeholder or None
            the input observation placeholder
        action : tf.placeholder or None
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
        if obs is None:
            obs = self.obs_ph
        if action is None:
            action = self.action_ph

        with tf.variable_scope(scope, reuse=reuse):
            # flatten the input placeholder
            qf_h = tf.layers.flatten(obs)

            # create the hidden layers
            for i, layer_size in enumerate(self.layers):
                qf_h = tf.layers.dense(qf_h, layer_size, name='fc' + str(i))
                if self.layer_norm:
                    qf_h = tf.contrib.layers.layer_norm(
                        qf_h, center=True, scale=True)
                qf_h = self.activ(qf_h)
                if i == 0:
                    qf_h = tf.concat([qf_h, action], axis=-1)

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
            return 0, 0, {}

        # Get a batch
        obs0, actions, rewards, obs1, terminals1 = self.replay_buffer.sample(
            batch_size=self.batch_size)

        # Reshape to match previous behavior and placeholder shape.
        rewards = rewards.reshape(-1, 1)
        terminals1 = terminals1.reshape(-1, 1)

        target_q = self.sess.run(self.target_q, feed_dict={
            self.obs1_ph: obs1,
            self.rew_ph: rewards,
            self.terminals1: terminals1
        })

        # TODO this is the belly of the gradient beast
        # TODO ---------------------------------------
        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads,
               self.critic_loss]
        td_map = {
            self.obs_ph: obs0,
            self.action_ph: actions,
            self.rew_ph: rewards,
            self.critic_target: target_q,
        }
        # TODO ---------------------------------------

        actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(
            ops, td_map)

        self.actor_optimizer.update(
            actor_grads, learning_rate=self.actor_lr)
        self.critic_optimizer.update(
            critic_grads, learning_rate=self.critic_lr)

        # Run target soft update operation.
        self.sess.run(self.target_soft_updates)

        return critic_loss, actor_loss, td_map

    def get_action(self, obs, **kwargs):
        """See parent class."""
        # Add the contextual observation, if applicable.
        context_obs = kwargs.get("context_obs")  # goals specified by Manager
        if context_obs is not None:
            obs = np.concatenate((obs, context_obs), axis=1)

        return self.sess.run(self.actor_tf, {self.obs_ph: obs})

    def value(self, obs, action=None, with_actor=True, **kwargs):
        """See parent class."""
        # Add the contextual observation, if applicable.
        context_obs = kwargs.get("context_obs")
        if context_obs is not None:
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
            obs0 = np.concatenate((obs0, kwargs["context_obs0"]), axis=0)
        if kwargs.get("context_obs1") is not None:
            obs1 = np.concatenate((obs1, kwargs["context_obs1"]), axis=0)

        self.replay_buffer.add(obs0, action, reward, obs1, float(done))
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs0]))

    def initialize(self):
        """See parent class.

        This method syncs the actor and critic optimizers across CPUs, and
        initializes the target parameters to match the model parameters.
        """
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()
        self.sess.run(self.target_init_updates)

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
        ops += [tf_util.reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [tf_util.reduce_std(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [tf_util.reduce_std(self.actor_tf)]
        names += ['reference_action_std']

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

    def get_target_q(self):
        return self.target_q

    def get_obs1_ph(self):
        return self.obs1_ph

    def get_rew_ph(self):
        return self.rew_ph

    def get_terminals1(self):
        return self.terminals1

    def get_actor_grads(self):
        return self.actor_grads

    def get_actor_loss(self):
        return self.actor_loss

    def get_critic_grads(self):
        return self.critic_grads

    def get_critic_loss(self):
        return self.critic_loss

    def get_obs_ph(self):
        return self.obs_ph

    def get_action_ph(self):
        return self.action_ph

    def get_critic_target(self):
        return self.critic_target

    def get_actor_optimizer(self):
        return self.actor_optimizer

    def get_actor_lr(self):
        return self.actor_lr

    def get_critic_optimizer(self):
        return self.critic_optimizer

    def get_critic_lr(self):
        return self.critic_lr

    def get_target_soft_updates(self):
        return self.target_soft_updates

    def get_actor_tf(self):
        return self.actor_tf


# TODO start of HIRO policy


class HIROPolicy(ActorCriticPolicy):
    """Hierarchical reinforcement learning with off-policy correction.

    See: https://arxiv.org/pdf/1805.08296.pdf

    Attributes
    ----------
    manager : hbaselines.hiro.policy.FeedForwardPolicy
        the manager policy
    meta_period : int
        manger action period
    prev_meta_obs : array_like
        previous observation by the Manager
    prev_meta_action : array_like
        action by the Manager at the previous time step
    meta_action : array_like
        current action by the Manager
    meta_reward : float
        current meta reward, counting as the cumulative environment reward
        during the meta period
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
                 observation_range,
                 normalize_returns,
                 return_range,
                 layer_norm=False,
                 reuse=False,
                 layers=None,
                 act_fun=tf.nn.relu):
        """Instantiate the HIRO policy.

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
        observation_range : (float, float)
            the bounding values for the observation
        normalize_returns : bool
            should the critic output be normalized
        return_range : (float, float)
            the bounding values for the critic output
        layer_norm : bool
            enable layer normalisation
        reuse : bool
            if the policy is reusable or not
        layers : list of int or None
            the size of the Neural network for the policy (if None, default to
            [64, 64])
        act_fun : tf.nn.*
            the activation function to use in the neural network

        Raises
        ------
        AssertionError
            if the layers is not a list of at least size 1
        """
        super(HIROPolicy, self).__init__(sess, ob_space, ac_space, co_space)

        self.replay_buffer = ReplayBuffer(buffer_size)

        # =================================================================== #
        # Part 1. Setup the Manager                                           #
        # =================================================================== #

        # Create the Manager policy.
        with tf.variable_scope("Manager"):
            self.manager = FeedForwardPolicy(
                sess=sess,
                ob_space=ob_space,
                ac_space=ob_space,  # outputs actions for each observations
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
                observation_range=observation_range,
                normalize_returns=normalize_returns,
                return_range=return_range,
                layer_norm=layer_norm,
                reuse=reuse,
                layers=layers,
                act_fun=act_fun,
                scope="Manager"
            )

        # manger action period
        self.meta_period = 10  # FIXME

        # previous observation by the Manager
        self.prev_meta_obs = None

        """
            Use this to store a list of observations
            that stretch as long as the dilated
            horizon chosen for the Manager.
            
            These observations correspond to the s(t)
            in the HIRO paper.
        """
        self._observations = []

        """
            Use this to store the list of environmental
            actions that the worker takes.
            
            These actions correspond to the a(t)
            in the HIRO paper.
        """
        self._worker_actions = []

        # action by the Manager at the previous time step
        self.prev_meta_action = None

        # current action by the Manager
        self.meta_action = None
        self.goals = []

        # current meta reward, counting as the cumulative environment reward
        # during the meta period
        self.meta_reward = None
        self.rewards = []

        """
            The following is redundant but necessary if the
            changes to the update function are to be in the
            HIRO policy and not the FeedForward.            
        """
        self.batch_size = batch_size

        # =================================================================== #
        # Part 1. Setup the Worker                                            #
        # =================================================================== #

        # Create the Worker policy.
        with tf.variable_scope("Worker"):
            self.worker = FeedForwardPolicy(
                sess,
                ob_space=ob_space,
                ac_space=ac_space,
                co_space=ob_space,
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
                observation_range=observation_range,
                normalize_returns=normalize_returns,
                return_range=return_range,
                layer_norm=layer_norm,
                reuse=reuse,
                layers=layers,
                act_fun=act_fun,
                scope="Worker"
            )

        # reward function for the worker
        def worker_reward(states, goals, next_states):
            return negative_distance(
                states=states,
                goals=goals,
                next_states=next_states,
                relative_context=False,
                diff=False,
                offset=0.0
            )[0]
        self.worker_reward = worker_reward

        manager_tf = self.manager.get_actor_lr()
        worker_obs_ph = self.worker.get_obs_ph()

        obs = np.concatenate((worker_obs_ph, manager_tf), axis=1)

        self.worker._make_actor(obs=obs)

    def initialize(self):
        """See parent class.

        This method calls the initialization methods of the manager and worker.
        """
        self.manager.initialize()
        self.worker.initialize()

    # TODO changes to update
    def update(self):
        """See parent class."""
        # -------------------

        # -------------------

        if not self.replay_buffer.can_sample(self.batch_size):
            return 0, 0, {}

        # Get a batch
        obs0, actions, rewards, obs1, terminals1 = self.replay_buffer.sample(
            batch_size=self.batch_size)

        # Reshape to match previous behavior and placeholder shape.
        rewards = rewards.reshape(-1, 1)
        terminals1 = terminals1.reshape(-1, 1)

        target_q = self.sess.run(self.manager.get_target_q(), feed_dict={
            self.manager.get_obs1_ph(): obs1,
            self.manager.get_rew_ph(): rewards,
            self.manager.get_terminals1(): terminals1
        })

        # TODO this is the belly of the gradient beast
        # TODO ---------------------------------------
        # Get all gradients and perform a synced update.
        ops = [self.worker.get_actor_grads(), self.worker.get_actor_loss(), self.manager.get_critic_grads(),
               self.manager.get_critic_loss()]
        td_map = {
            self.worker.get_obs_ph(): obs0,
            self.worker.get_action_ph(): actions,
            self.manager.get_rew_ph(): rewards,
            self.manager.get_critic_target(): target_q,
        }
        # TODO ---------------------------------------

        actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(
            ops, td_map)

        self.worker.get_actor_optimizer().update(
            actor_grads, learning_rate=self.worker.get_actor_lr())
        self.manager.get_critic_optimizer().update(
            critic_grads, learning_rate=self.manager.get_critic_lr())

        # Run target soft update operation.
        self.sess.run(self.manager.get_target_soft_updates())

        return critic_loss, actor_loss, td_map

    def get_action(self, obs, state=None, mask=None, meta_act=None, **kwargs):
        """See parent class."""
        # Update the meta action, if the time period requires is.
        if kwargs["time"] % self.meta_period == 0:
            self.meta_action = self.manager.get_action(obs)
            self.goals.append(self.meta_action)

        # Return the worker action.
        if meta_act is None:
            worker_obs = np.concatenate((obs, self.meta_action), axis=1)
        else:
            worker_obs = np.concatenate((obs, meta_act), axis=1)

        # compute worker action
        worker_action = self.worker.get_action(worker_obs)

        # notify manager of new worker action to be saved
        self._notify_manager(worker_action)

        return worker_action

    def _notify_manager(self, action, **kwargs):
        """
        Notify the Manager that the Worker produced a new action.
        These actions are saved in the Manager for future off-
        policy enhancements.

        Parameters
        ----------
        action: Any
            current action produced by Worker
        """
        if kwargs["time"] % self.meta_period == 0:
            self._worker_actions.clear()
        else:
            self._worker_actions.append(action)

    def _observation_memory(self, obs, **kwargs):
        """
        Notify the Manager that there is a new environmental
        observation. These new observations are saved in the
        Manager for future off-policy correction.

        obs: Any
            current environmental observation
        """
        if kwargs["time"] % self.meta_period == 0:
            self._observations.clear()
        else:
            self._observations.append(obs)

    def value(self, obs, action=None, with_actor=True, state=None, mask=None):
        """See parent class."""
        return 0  # FIXME

    def store_transition(self, obs0, action, reward, obs1, done, **kwargs):
        """See parent class."""
        # Add a sample to the meta transition, if required.
        if kwargs["time"] % self.meta_period == 0 or done:
            # If this is the first time step, do not add the transition to the
            # meta replay buffer (it is not complete yet).
            if kwargs["time"] != 0:
                # Store a sample in the Manager policy.
                self.replay_buffer.new_add(
                    obs_t=self._observations,
                    goal_t=self.goals,
                    action_t=self._worker_actions,
                    reward_t=self.meta_reward,
                    done_=done,
                    h_t=self.goal_xsition_model(obs0, action, obs1),
                    goal_updated=kwargs["time"] % self.meta_period == 0,
                )
            else:
                # This hasn't been assigned yet, so assign it here.
                self.prev_meta_action = deepcopy(self.meta_action)

            # Reset the meta reward and previous meta observation.
            self.meta_reward = 0
            self.prev_meta_obs = np.copy(obs0)
        else:
            # Increment the meta reward with the most recent reward.
            self.meta_reward += reward
            self.rewards.append(self.meta_reward)

        # Compute the worker reward.
        worker_reward = self.worker_reward(obs0, self.meta_action, obs1)

        # Add the worker transition to the replay buffer.
        self.worker.store_transition(
            obs0=obs0,
            context_obs0=self.prev_meta_action,
            action=action,
            reward=worker_reward,
            obs1=obs1,
            context_obs1=self.meta_action,
            done=done
        )

        # Update the prev meta action to match that of the current time step.
        self.prev_meta_action = deepcopy(self.meta_action)

    def get_stats(self):
        """See parent class."""
        return {}  # FIXME

    def goal_xsition_model(self,
                           obs_t,
                           g_t,
                           obs_tp1):
        """
        Fixed goal transition function defined by the following eqn:

        h(s_t, g_t, s_t+1) = s_t + g_t - s_t+1

        Parameters:
        -----------
        obs_t: Any
            environmental observation at time t
        g_t: Any
            Worker specified goal at time t
        obs_tp1: Any
            environmental observation at time t

        Returns
        -------
        g_tp1: Any
            Worker specified goal at time t+1
        """
        return tf.subtract(tf.add(obs_t, g_t), obs_tp1)

    def off_policy_correction(self,
                              data,
                              horizon,
                              c,
                              state_reps,
                              next_state_reprs,
                              prev_meta_actions,
                              low_states,
                              low_actions,
                              low_state_reprs,
                              tf_spec,
                              k=8):
        """
        Function in order to perform our Manager off-
        policy correction of specified goals.

        Defined by approximately solving the argmax of:

        -0.5 * summation(||a_i - pi(s_i, g_i)|| ** 2 + constant)

        over all values of time between (t, t+c-1)

        Parameters
        ----------
        data: tuple
            replay buffer data
        horizon: int
            Manager horizon
        c: int
            constant defined in equation
        """
        # current goal
        goals = [tuple(data.index(3)).index(0)]

        tmp = self._sample_best_meta_action(state_reps,
                                            next_state_reprs,
                                            prev_meta_actions,
                                            low_states,
                                            low_actions,
                                            low_state_reprs,
                                            tf_spec,
                                            k)
        # 8 candidate goals
        for elem in tmp:
            goals.append(elem)

        # goal based on sampling from a distribution centered
        # at (s_t+c - s_t)
        goals.append(self.manager.get_action(
            list(data.index(0)).index(-1)) - list(data.index(0)).index(0))  # got my goals

        tmp_actions = list(data.index(1))
        actions = []
        for a in tmp_actions:
            actions.append(a)  # got my actions

        tmp_states = list(data.index(0))
        states = []
        for s in tmp_states:
            states.append(s)  # got my states

        decision_list = []
        tmp_var = 0
        index = 0

        for time in range(horizon):
            tmp_var -= 0.5 * (
                    self.euclidean_distance(actions.index(time),
                                            self.worker.get_action(
                                                obs=states.index(time),
                                                meta_act=goals.index(time))) ** 2) + c
            decision_list.append(dict([(index, tmp_var)]))
            index, tmp_var = index+1, 0

        # now find the argmax
        decision = max(decision_list)

        return decision  # todo fix this to be a replacement in replay buffer

    def euclidean_distance(self, a, b):
        """
        Computes pairwise distances between each elements of A and each elements of B.
        Args:
          a,    [m,d] matrix
          b,    [n,d] matrix
        Returns:
          d,    [m,n] matrix of pairwise distances
        """
        with tf.variable_scope('pairwise_dist'):
            # squared norms of each row in A and B
            na = tf.reduce_sum(tf.square(a), 1)
            nb = tf.reduce_sum(tf.square(b), 1)

            # na as a row and nb as a column vectors
            na = tf.reshape(na, [-1, 1])
            nb = tf.reshape(nb, [1, -1])

            # return pairwise euclidean difference matrix
            d = tf.sqrt(tf.maximum(na - 2 * tf.matmul(a, b, False, True) + nb, 0.0))
        return d

    def _sample_best_meta_action(self,
                                 state_reps,
                                 next_state_reprs,
                                 prev_meta_actions,
                                 low_states,
                                 low_actions,
                                 low_state_reprs,
                                 tf_spec,
                                 k=8):
        """
        Return meta-actions which approximately maximize low-level log-probs.

        state_reps: Any
            current Manager state observation
        next_state_reprs: Any
            next Manager state observation
        prev_meta_actions: Any
            previous meta Manager action
        low_states: Any
            current Worker state observation
        low_actions: Any
            current Worker environmental action
        low_state_reprs: Any
            BLANK
        k: int
            number of goals returned
        """
        sampled_actions = self._sample(state_reps,
                                       next_state_reprs,
                                       k,
                                       prev_meta_actions,
                                       tf_spec)

        sampled_actions = tf.stop_gradient(sampled_actions)

        sampled_log_probs = tf.reshape(self._log_probs(
            tf.tile(low_states, [k, 1, 1]),
            tf.tile(low_actions, [k, 1, 1]),
            tf.tile(low_state_reprs, [k, 1, 1]),
            [tf.reshape(sampled_actions, [-1, sampled_actions.shape[-1]])]),
            [k, low_states.shape[0],
             low_states.shape[1], -1])

        fitness = tf.reduce_sum(sampled_log_probs, [2, 3])
        best_actions = tf.argmax(fitness, 0)
        actions = tf.gather_nd(
            sampled_actions,
            tf.stack([best_actions,
                      tf.range(prev_meta_actions.shape[0], dtype=tf.int64)], -1))
        return actions

    # TODO fix me
    def _log_probs(self,
                   states,
                   actions,  # use this as a target value for error
                   tf_spec,  # use this to define max and min
                   goals):
        """
        Utility function that helps in calculating the
        log probability of the next goal by the Manager.

        states: Any
            list of states corresponding to that of Manager
        state_reps: Any
            list of state representations corresponding to s(t)
        context: Any
            BLANK

        Returns
        -------
        next likely goal by the Manager defined by log prob

        Helps
        -----
        * _sample_best_meta_action(self):
        """
        batch_dims = [tf.shape(states)[0], tf.shape(states)[1]]

        contexts = []
        for index in range(len(states)-1):
            contexts.append(self.goal_xsition_model(states[index],
                                                    goals[index],
                                                    states[index+1]))

        flat_contexts = [tf.reshape(tf.cast(context, states.dtype),
                                    [batch_dims[0] * batch_dims[1], context.shape[-1]])
                         for context in contexts]

        flat_pred_actions = self.worker.get_action(flat_contexts)

        pred_actions = tf.reshape(flat_pred_actions,
                                  batch_dims + [flat_pred_actions.shape[-1]])

        error = tf.square(actions - pred_actions)

        spec_range = (tf_spec.maximum - tf_spec.minimum) / 2

        normalized_error = error / tf.constant(spec_range) ** 2

        return -normalized_error

    def _sample(self,
                states,
                next_states,
                num_samples,
                orig_goals,
                tf_spec,
                sc=0.5):
        """
        Sample different goals from a random Gaussian distribution
        centered at (s_t+c) - (s_t)

        states: Any
            current time step observation
        next_states: Any
            next time step observation
        num_samples: Any
            number of samples
        orig_goals: Any
            original goal specified by Manager
        tf_spec: tf.TensorSpec
            Metadata for describing the tf.Tensor objects accepted
            or returned by some TensorFlow APIs.
        sc: float
            BLANK

        Helps
        -----
        * _sample_best_meta_action(self):
        """
        goal_dim = orig_goals.shape[-1]
        spec_range = (tf_spec.maximum - tf_spec.minimum) / 2 * tf.ones([goal_dim])
        loc = tf.cast(next_states - states, tf.float32)[:, :goal_dim]
        scale = sc * tf.tile(tf.reshape(spec_range, [1, goal_dim]),
                             [tf.shape(states)[0], 1])
        dist = tf.distributions.Normal(loc, scale)
        if num_samples == 1:
            return dist.sample()
        samples = tf.concat([dist.sample(num_samples - 2),
                             tf.expand_dims(loc, 0),
                             tf.expand_dims(orig_goals, 0)], 0)
        return self._clip_to_spec(samples, tf_spec)

    def _clip_to_spec(self, value, spec):
        """
        Clips value to a given bounded tensor spec.

        Args:
          value: (tensor) value to be clipped.
          spec: (BoundedTensorSpec) spec containing min. and max. values for clipping.

        Returns:
          clipped_value: (tensor) `value` clipped to be compatible with `spec`.

        Helps:
          *_sample(self,
                states,
                next_states,
                num_samples,
                orig_goals,
                tf_spec,
                sc=0.5):
        """
        return self.clip_to_bounds(value,
                                   spec.minimum,
                                   spec.maximum)

    def clip_to_bounds(self, value, minimum, maximum):
        """Clips value to be between minimum and maximum.
        Args:
          value: (tensor) value to be clipped.
          minimum: (numpy float array) minimum value to clip to.
          maximum: (numpy float array) maximum value to clip to.
        Returns:
          clipped_value: (tensor) `value` clipped to between `minimum` and `maximum`.
        """
        value = tf.minimum(value, maximum)
        return tf.maximum(value, minimum)
