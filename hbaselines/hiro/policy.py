import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from functools import reduce

from hbaselines.hiro.tf_util import normalize, denormalize, flatgrad, \
    get_target_updates, get_trainable_vars
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines import logger
from stable_baselines.deepq.replay_buffer import ReplayBuffer


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
    """

    def __init__(self, sess, ob_space, ac_space):
        """Instantiate the base policy object.

        Parameters
        ----------
        sess : tf.Session
            the current TensorFlow session
        ob_space : gym.space.*
            the observation space of the environment
        ac_space : gym.space.*
            the action space of the environment
        """
        self.sess = sess
        self.ob_space = ob_space
        self.ac_space = ac_space

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

    def get_action(self, obs, state=None, mask=None):
        """Call the actor methods to compute policy actions.

        Parameters
        ----------
        obs : array_like
            the observation
        state : TODO
            TODO
        mask : TODO
            TODO

        Returns
        -------
        array_like
            computed action by the policy
        """
        raise NotImplementedError

    def value(self, obs, action=None, with_actor=True, state=None, mask=None):
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
        state : TODO
            TODO
        mask : TODO
            TODO

        Returns
        -------
        array_like
            computed value by the critic
        """
        raise NotImplementedError

    def store_transition(self, obs0, action, reward, obs1, terminal1):
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
    replay_buffer : TODO
        the replay buffer
    critic_target : tf.placeholder
        TODO
    terminals1 : tf.placeholder
        placeholder for the next step terminals
    rew_ph : tf.placeholder
        placeholder for the rewards
    action_ph : tf.placeholder
        placeholder for the actions
    obs_ph : tf.placholder
        placeholder for the observations
    obs1_ph : tf.placeholder
        placeholder for the next step observations
    obs_rms : TODO
        TODO
    ret_rms : TODO
        TODO
    actor_tf : tf.Variable
        the output from the actor network
    normalized_critic_tf : tf.Variable
        TODO
    normalized_critic_with_actor_tf : tf.Variable
        TODO
    critic_tf : tf.Variable
        TODO
    critic_with_actor_tf : tf.Variable
        TODO
    target_q : tf.Variable
        TODO
    target_init_updates : tf.Operation
        TODO
    target_soft_updates : tf.Operation
        TODO
    actor_loss : tf.Operation
        TODO
    actor_grads : tf.Operation
        TODO
    actor_optimizer : tf.Operation
        TODO
    critic_loss : tf.Operation
        TODO
    critic_grads : tf.Operation
        TODO
    critic_optimizer : tf.Operation
        TODO
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
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
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.critic_l2_reg = critic_l2_reg
        self.verbose = verbose
        self.reuse = reuse
        self.layers = layers or [64, 64]
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
                shape=(None,) + ob_space.shape,
                name='observations')
            self.obs1_ph = tf.placeholder(
                tf.float32,
                shape=(None,) + ob_space.shape,
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
                self.obs_rms = RunningMeanStd(shape=ob_space.shape)
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
                normalized_obs0,
                self.action_ph)
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

        # TODO: do I need indent?
        # Create the target update operations.
        init_updates, soft_updates = get_target_updates(
            get_trainable_vars('model/'),
            get_trainable_vars('target/'),
            tau, verbose)
        self.target_init_updates = init_updates
        self.target_soft_updates = soft_updates

        # =================================================================== #
        # Step 5: Setup the optimizers for the actor and critic.              #
        # =================================================================== #

        with tf.variable_scope("Adam_mpi", reuse=False):
            self._setup_actor_optimizer()
            self._setup_critic_optimizer()
            tf.summary.scalar('actor_loss', self.actor_loss)
            tf.summary.scalar('critic_loss', self.critic_loss)

    def _setup_actor_optimizer(self):
        """Create the actor loss, gradient, and optimizer."""
        if self.verbose >= 2:
            logger.info('setting up actor optimizer')

        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list()
                        for var in get_trainable_vars('model/pi/')]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape)
                               for shape in actor_shapes])
        if self.verbose >= 2:
            logger.info('  actor shapes: {}'.format(actor_shapes))
            logger.info('  actor params: {}'.format(actor_nb_params))

        self.actor_grads = flatgrad(
            self.actor_loss,
            get_trainable_vars('model/pi/'),
            clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(
            var_list=get_trainable_vars('model/pi/'),
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def _setup_critic_optimizer(self):
        """Create the critic loss, gradient, and optimizer."""
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
                var for var in get_trainable_vars('model/qf/')
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
                         for var in get_trainable_vars('model/qf/')]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                for shape in critic_shapes])

        if self.verbose >= 2:
            logger.info('  critic shapes: {}'.format(critic_shapes))
            logger.info('  critic params: {}'.format(critic_nb_params))

        self.critic_grads = flatgrad(
            self.critic_loss,
            get_trainable_vars('model/qf/'),
            clip_norm=self.clip_norm)

        self.critic_optimizer = MpiAdam(
            var_list=get_trainable_vars('model/qf/'),
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def _make_actor(self, obs=None, reuse=False, scope="pi"):
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
                if i == 0:  # FIXME: ????
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

    def get_action(self, obs, state=None, mask=None):
        """See parent class."""
        return self.sess.run(self.actor_tf, {self.obs_ph: obs})

    def value(self, obs, action=None, with_actor=True, state=None, mask=None):
        """See parent class."""
        if with_actor:
            return self.sess.run(
                self.critic_with_actor_tf,
                feed_dict={self.obs_ph: obs})
        else:
            return self.sess.run(
                self.critic_tf,
                feed_dict={self.obs_ph: obs, self.action_ph: action})

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        """See parent class."""
        self.replay_buffer.add(obs0, action, reward, obs1, float(terminal1))
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
