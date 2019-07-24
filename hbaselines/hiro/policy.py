"""TD3-compatible policies."""
import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from functools import reduce
from copy import deepcopy
import logging
from gym.spaces import Box

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
                 observation_range,
                 normalize_returns,
                 return_range,
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

        normalized_obs0 = tf.clip_by_value(
            normalize(self.obs_ph, self.obs_rms),
            observation_range[0], observation_range[1])
        normalized_obs1 = tf.clip_by_value(
            normalize(self.obs1_ph, self.obs_rms),
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
                normalized_obs0, self.action_ph)
            self.normalized_critic_with_actor_tf = self._make_critic(
                normalized_obs0, self.actor_tf, reuse=True)

        with tf.variable_scope("target", reuse=False):
            actor_target = self._make_actor(normalized_obs1)
            critic_target = self._make_critic(normalized_obs1, actor_target)

        with tf.variable_scope("loss", reuse=False):
            self.critic_tf = denormalize(
                tf.clip_by_value(
                    self.normalized_critic_tf,
                    return_range[0],
                    return_range[1]),
                self.ret_rms)

            self.critic_with_actor_tf = denormalize(
                tf.clip_by_value(
                    self.normalized_critic_with_actor_tf,
                    return_range[0],
                    return_range[1]),
                self.ret_rms)

            q_obs1 = denormalize(critic_target, self.ret_rms)
            self.target_q = self.rew_ph + (1-self.terminals1) * gamma * q_obs1

            tf.summary.scalar('critic_target',
                              tf.reduce_mean(self.critic_target))

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
                        for var in get_trainable_vars(scope_name)]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape)
                               for shape in actor_shapes])
        if self.verbose >= 2:
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
            normalize(self.critic_target, self.ret_rms),
            self.return_range[0],
            self.return_range[1])

        self.critic_loss = tf.reduce_mean(tf.square(
            self.normalized_critic_tf - normalized_critic_target_tf))

        scope_name = 'model/qf/'
        if scope is not None:
            scope_name = scope + '/' + scope_name

        if self.critic_l2_reg > 0.:
            critic_reg_vars = [
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

        critic_shapes = [var.get_shape().as_list()
                         for var in get_trainable_vars(scope_name)]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                for shape in critic_shapes])

        if self.verbose >= 2:
            logging.info('  critic shapes: {}'.format(critic_shapes))
            logging.info('  critic params: {}'.format(critic_nb_params))

        self.critic_grads = flatgrad(
            self.critic_loss,
            get_trainable_vars(scope_name),
            clip_norm=self.clip_norm)

        self.critic_optimizer = MpiAdam(
            var_list=get_trainable_vars(scope_name),
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def _make_actor(self, obs, reuse=False, scope="pi"):
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

    def _make_critic(self, obs, action, reuse=False, scope="qf"):
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
        dict
            feed_dict map for the summary (to be run in the algorithm)
        """
        # Reshape to match previous behavior and placeholder shape.
        rewards = rewards.reshape(-1, 1)
        terminals1 = terminals1.reshape(-1, 1)

        target_q = self.sess.run(self.target_q, feed_dict={
            self.obs1_ph: obs1,
            self.rew_ph: rewards,
            self.terminals1: terminals1
        })

        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads,
               self.critic_loss]
        td_map = {
            self.obs_ph: obs0,
            self.action_ph: actions,
            self.rew_ph: rewards,
            self.critic_target: target_q,
        }

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
        context_obs = kwargs.get("context_obs")
        if context_obs[0] is not None:
            obs = np.concatenate((obs, context_obs), axis=1)

        return self.sess.run(self.actor_tf, {self.obs_ph: obs})

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
        """Create the running means and std of the model inputs and outputs."""
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


class GoalDirectedPolicy(ActorCriticPolicy):
    """Goal-directed hierarchical reinforcement learning model.

    TODO: Description

    See: http://papers.nips.cc/paper/714-feudal-reinforcement-learning.pdf

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
    centralized_value_functions : bool
        specifies whether to use centralized value functions for the Manager
        and Worker critic functions
    connected_gradients : bool
        whether to connect the graph between the manager and worker
    fingerprint_dim : tuple of int
        the shape of the fingerprint elements, if they are being used
    fingerprint_range : (list of float, list of float)
        the low and high values for each fingerprint element, if they are being
        used
    prev_meta_obs : array_like
        previous observation by the Manager
    prev_meta_action : array_like
        action by the Manager at the previous time step
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
                 observation_range,
                 normalize_returns,
                 return_range,
                 layer_norm=False,
                 reuse=False,
                 layers=None,
                 act_fun=tf.nn.relu,
                 meta_period=10,
                 relative_goals=False,
                 off_policy_corrections=False,
                 use_fingerprints=False,
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
        self.centralized_value_functions = centralized_value_functions
        self.connected_gradients = connected_gradients
        self.fingerprint_dim = (1,)
        self.fingerprint_range = ([0], [5])

        # Compute the observation space for the Manager and Worker.
        #
        # If the fingerprint terms are being appended onto the observations,
        # this should be the original observation space plus the fingerprint
        # spaces at the end of the observation.
        if self.use_fingerprints:
            low = np.concatenate((ob_space.low, self.fingerprint_range[0]))
            high = np.concatenate((ob_space.high, self.fingerprint_range[1]))
            manager_worker_ob_space = Box(low=low, high=high)
        else:
            manager_worker_ob_space = ob_space

        self.replay_buffer = HierReplayBuffer(buffer_size)

        # =================================================================== #
        # Part 1. Setup the Manager                                           #
        # =================================================================== #

        # Create the Manager policy.
        with tf.variable_scope("Manager"):
            self.manager = FeedForwardPolicy(
                sess=sess,
                ob_space=manager_worker_ob_space,
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

        # previous observation by the Manager
        self.prev_meta_obs = None

        # action by the Manager at the previous time step
        self.prev_meta_action = None

        # current action by the Manager
        self.meta_action = None

        # current meta reward, counting as the cumulative environment reward
        # during the meta period
        self.meta_reward = None
        self.rewards = []

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
                ob_space=manager_worker_ob_space,
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

        # remove the last element to compute the reward
        if self.use_fingerprints:
            state_indices = list(np.arange(
                0, self.manager.ob_space.shape[0] - self.fingerprint_dim[0]))
        else:
            state_indices = None

        # reward function for the worker
        def worker_reward(states, goals, next_states):
            return negative_distance(
                states=states,
                state_indices=state_indices,
                goals=goals,
                next_states=next_states,
                relative_context=False,
                diff=False,
                offset=0.0
            )[0]
        self.worker_reward = worker_reward

    def initialize(self):
        """See parent class.

        This method calls the initialization methods of the manager and worker.
        """
        self.manager.initialize()
        self.worker.initialize()

    def update(self):
        """See parent class."""
        # Not enough samples in the replay buffer.
        if not self.manager.replay_buffer.can_sample(self.batch_size) or \
                not self.worker.replay_buffer.can_sample(self.batch_size):
            return 0, 0, {}

        # Get a batch.
        worker_obs0, _, worker_actions, worker_rewards, worker_done1, \
            _, _, worker_obs1 = self.worker.replay_buffer.sample(
                batch_size=self.batch_size)

        manager_obs0, _, manager_actions, manager_rewards, manager_done1, \
            _, _, manager_obs1 = self.manager.replay_buffer.sample(
                batch_size=self.batch_size)

        # Update the Manager policy.
        self.manager.update_from_batch(
            obs0=manager_obs0,
            actions=manager_actions,
            rewards=manager_rewards,
            obs1=manager_obs1,
            terminals1=manager_done1
        )

        # Update the Worker policy.
        self.worker.update_from_batch(
            obs0=worker_obs0,
            actions=worker_actions,
            rewards=worker_rewards,
            obs1=worker_obs1,
            terminals1=worker_done1
        )

        return 0, 0, {}  # FIXME

    def get_action(self, obs, state=None, mask=None, **kwargs):
        """See parent class."""
        # Update the meta action, if the time period requires is.
        if kwargs["time"] % self.meta_period == 0:
            self.meta_action = self.manager.get_action(obs)

        # Return the worker action.
        worker_obs = np.concatenate((obs, self.meta_action), axis=1)
        worker_action = self.worker.get_action(worker_obs)

        return worker_action

    def value(self, obs, action=None, with_actor=True, state=None, mask=None):
        """See parent class."""
        return 0  # FIXME

    def store_transition(self, obs0, action, reward, obs1, done, **kwargs):
        """See parent class."""
        # Compute the worker reward and append it to the list of rewards.
        self._worker_rewards.append(
            self.worker_reward(obs0, self.meta_action, obs1)
        )

        # Add the environmental observations and done masks, and the worker
        # actions to their respective lists.
        self._worker_actions.append(action)
        self._observations.append(obs0)
        self._dones.append(done)

        # Increment the meta reward with the most recent reward.
        self.meta_reward += reward
        self.rewards.append(self.meta_reward)

        # Add a sample to the replay buffer.
        if kwargs["time"] % self.meta_period == 0 or done:
            # Add the last observation if about to reset.
            if done:
                self._observations.append(obs1)

            # If this is the first time step, do not add the transition to the
            # meta replay buffer (it is not complete yet).
            if kwargs["time"] != 0:
                # Store a sample in the Manager policy.
                self.replay_buffer.add(
                    obs_t=self._observations,
                    goal_t=self.prev_meta_action,
                    action_t=self._worker_actions,
                    reward_t=self._worker_rewards,
                    done=self._dones,
                    meta_obs_t=self.prev_meta_obs,
                    meta_reward_t=self.meta_reward,
                )

                # Reset the meta reward and previous meta observation.
                self.meta_reward = 0
                self.prev_meta_obs = np.copy(obs0)

                # Clear the worker rewards and actions, and the environmental
                # observation.
                self._observations.clear()
                self._worker_actions.clear()
                self._worker_rewards.clear()
                self._dones.clear()

        # Update the prev meta action to match that of the current time step.
        self.prev_meta_action = deepcopy(self.meta_action)

    @staticmethod
    def goal_xsition_model(obs_t, g_t, obs_tp1):
        """Return a fixed goal transition.

        The goal transition is defined by the following eqn:

            h(s_t, g_t, s_t+1) = s_t + g_t - s_t+1

        Parameters
        ----------
        obs_t : array_like
            environmental observation at time t
        g_t : array_like
            Worker specified goal at time t
        obs_tp1 : array_like
            environmental observation at time t+1

        Returns
        -------
        array_like
            Worker specified goal at time t+1
        """
        return obs_t + g_t - obs_tp1

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
        # Compute absolute goals, if needed.
        if self.relative_goals:
            goals = self.goal_xsition_model(manager_obs, goals, worker_obs)

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
        return {}  # FIXME
