from stable_baselines.common.input import observation_input
from stable_baselines.common import tf_util
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines import logger
from functools import reduce
import numpy as np
import tensorflow as tf
from copy import deepcopy
from gym.spaces import Box
from hbaselines.models import build_fcnet, build_lstm
from hbaselines.utils.stats import normalize, denormalize


class FullyConnectedPolicy(object):
    """Policy object that implements a DDPG-like actor critic, using MLPs."""

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 actor_layers=None,
                 critic_layers=None,
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=tf.nn.tanh,
                 obs_rms=None,
                 ret_rms=None,
                 observation_range=(-np.inf, np.inf),
                 return_range=(-np.inf, np.inf)):
        """Instantiate the policy model.

        Parameters
        ----------
        sess : tf.Session
            the tensorflow session
        ob_space : gym.spaces.Box
            shape of the observation space
        ac_space : gym.spaces.Box
            shape of the action space
        actor_layers : list of int
            number of nodes in each hidden layer of the actor
        critic_layers : list of int
            number of nodes in each hidden layer of the critic
        hidden_nonlinearity : tf.nn.*
            activation nonlinearity for the hidden nodes of the model
        output_nonlinearity : tf.nn.*
            activation nonlinearity for the output of the model
        """
        if actor_layers is None:
            actor_layers = [128, 128]
        if critic_layers is None:
            critic_layers = [128, 128]

        # inputs
        self.sess = sess
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.actor_layers = actor_layers
        self.critic_layers = critic_layers
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.obs_rms = obs_rms
        self.ret_rms = ret_rms
        self.observation_range = observation_range
        self.return_range = return_range

        # create placeholders for the observations and actions
        with tf.variable_scope('input', reuse=False):
            self.obs_ph, self.processed_x = observation_input(
                ob_space, batch_size=None, scale=False)
            self.action_ph = tf.placeholder(
                dtype=ac_space.dtype, shape=(None,) + ac_space.shape,
                name='action_ph')
        self.critic_target = tf.placeholder(
            tf.float32, shape=(None, 1), name='critic_target')

        # create normalized variants of the observationss and actions
        self.normalized_obs_ph = tf.clip_by_value(
            normalize(self.processed_x, obs_rms),
            observation_range[0], observation_range[1])

        # variables that will be created by later methods
        self.policy = None
        self.critic_with_actor = None
        self.critic = None

        # normalized versions of critics
        self.normalized_critic_with_actor = None
        self.normalized_critic = None

        # flattened versions of critics
        self._critic_with_actor = None
        self._critic = None

        # optimization variables for actors and critics
        self.actor_loss = None
        self.actor_grads = None
        self.actor_optimizer = None
        self.critic_loss = None
        self.critic_grads = None
        self.critic_optimizer = None

        # some assertions
        assert isinstance(ac_space, Box), \
            'Error: the action space must be of type gym.spaces.Box'
        assert np.all(np.abs(ac_space.low) == ac_space.high), \
            'Error: the action space low and high must be symmetric'
        assert len(actor_layers) >= 1, \
            'Error: must have at least one hidden layer for the actor.'
        assert len(critic_layers) >= 1, \
            'Error: must have at least one hidden layer for the critic.'

    def make_actor(self, reuse=False, scope='pi'):
        """Create an actor tensor.

        Parameters
        ----------
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor

        Returns
        -------
        tuple of tf.Tensor
            the output tensor for the actor
        """
        self.policy = build_fcnet(
            inputs=tf.layers.flatten(self.normalized_obs_ph),
            num_outputs=self.ac_space.shape[0],
            hidden_size=self.actor_layers,
            hidden_nonlinearity=self.hidden_nonlinearity,
            output_nonlinearity=self.output_nonlinearity,
            scope=scope,
            reuse=reuse
        )

    def make_critic(self, obs=None, reuse=False, scope='qf'):
        """Create a critic tensor.

        Parameters
        ----------
        obs : tf.Tensor
            The observation placeholder (can be None for default placeholder)
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the critic

        Returns
        -------
        tf.Tensor
            the output tensor for the critic
        """
        if obs is None:
            obs = self.processed_x

        vf_h = tf.layers.flatten(obs)

        self.normalized_critic = build_fcnet(
            inputs=tf.concat([vf_h, self.action_ph], axis=-1),
            num_outputs=1,
            hidden_size=self.critic_layers,
            hidden_nonlinearity=self.hidden_nonlinearity,
            output_nonlinearity=self.output_nonlinearity,
            scope=scope,
            reuse=False,
            prefix='normalized_critic'
        )

        critic = denormalize(
            tf.clip_by_value(self.normalized_critic,
                             self.return_range[0],
                             self.return_range[1]),
            self.ret_rms)

        self.critic = critic
        self._critic = critic[:, 0]

        self.normalized_critic_with_actor = build_fcnet(
            inputs=tf.concat([vf_h, self.policy], axis=-1),
            num_outputs=1,
            hidden_size=self.critic_layers,
            hidden_nonlinearity=self.hidden_nonlinearity,
            output_nonlinearity=self.output_nonlinearity,
            scope=scope,
            reuse=True,
            prefix='normalized_critic'
        )

        critic_with_actor = denormalize(
            tf.clip_by_value(self.normalized_critic_with_actor,
                             self.return_range[0],
                             self.return_range[1]),
            self.ret_rms)

        self.critic_with_actor = critic_with_actor
        self._critic_with_actor = critic_with_actor[:, 0]

    def setup_actor_optimizer(self, clip_norm, verbose):
        """Create the actor loss, gradient, and optimizer functions.

        This structure is compatible with DDPG.

        Parameters
        ----------
        clip_norm : blank
            blank
        verbose : blank
            blank
        """
        if verbose >= 2:
            logger.info('setting up actor optimizer')
        # TODO: add mask? (see Lample & Chatlot 2016)
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor)

        actor_shapes = [var.get_shape().as_list()
                        for var in tf_util.get_trainable_vars('model/pi/')]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape)
                               for shape in actor_shapes])

        if verbose >= 2:
            logger.info('  actor shapes: {}'.format(actor_shapes))
            logger.info('  actor params: {}'.format(actor_nb_params))

        self.actor_grads = tf_util.flatgrad(
            self.actor_loss, tf_util.get_trainable_vars('model/pi/'),
            clip_norm=clip_norm)

        self.actor_optimizer = MpiAdam(
            var_list=tf_util.get_trainable_vars('model/pi/'),
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def setup_critic_optimizer(self,
                               critic_l2_reg,
                               clip_norm,
                               verbose):
        """Create the critic loss, gradient, and optimizer functions.

        This structure is compatible with DDPG.

        Parameters
        ----------FIXME
        critic_l2_reg : blank
            blank
        clip_norm : blank
            blank
        verbose : blank
            blank
        """
        if verbose >= 2:
            logger.info('setting up critic optimizer')

        normalized_critic_target_tf = tf.clip_by_value(
            normalize(self.critic_target, self.ret_rms),
            self.return_range[0],
            self.return_range[1])

        # TODO: add mask? (see Lample & Chatlot 2016)
        self.critic_loss = tf.reduce_mean(tf.square(
            self.normalized_critic - normalized_critic_target_tf))

        if critic_l2_reg > 0.:
            critic_reg_vars = [var for var
                               in tf_util.get_trainable_vars('model/qf/')
                               if 'bias' not in var.name and 'output'
                               not in var.name and 'b' not in var.name]
            if verbose >= 2:
                for var in critic_reg_vars:
                    logger.info('  regularizing: {}'.format(var.name))
                logger.info('  applying l2 regularization with {}'.
                            format(critic_l2_reg))

            critic_reg = tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l2_regularizer(critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list()
                         for var in tf_util.get_trainable_vars('model/qf/')]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                for shape in critic_shapes])
        if verbose >= 2:
            logger.info('  critic shapes: {}'.format(critic_shapes))
            logger.info('  critic params: {}'.format(critic_nb_params))

        self.critic_grads = tf_util.flatgrad(
            self.critic_loss,
            tf_util.get_trainable_vars('model/qf/'),
            clip_norm=clip_norm)

        self.critic_optimizer = MpiAdam(
            var_list=tf_util.get_trainable_vars('model/qf/'),
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def step(self, obs, state=None, mask=None, **kwargs):
        """Call the actor methods to compute policy actions.

        Parameters
        ----------FIXME
        obs : list of float or np.ndarray
            blank
        state : blank
            blank
        mask : blank
            blank

        Returns
        -------
        list of float or list of int
            computed action by the policy
        """
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def value(self,
              obs,
              action,
              state=None,
              mask=None,
              use_actor=False,
              **kwargs):
        """Compute the estimated value by the critic.

        This method may use an inputted action via placeholders, or can compute
        the action from the state given the policy.

        Parameters
        ----------FIXME
        obs : list of float or np.ndarray
            blank
        action : blank
            blank
        state : blank
            blank
        mask : blank
            blank
        use_actor : bool, defaults
            specifies whether to use the actions derived from the actor or the
            ones inputted by an external object (e.g. replay buffer)

        Returns
        -------
        list of float or list of int
            computed action by the policy
        """
        if use_actor:
            return self.sess.run(
                self._critic_with_actor, feed_dict={self.obs_ph: obs})
        else:
            return self.sess.run(
                self._critic,
                feed_dict={self.obs_ph: obs, self.action_ph: action})

    def train_actor_critic(self, batch, target_q, actor_lr, critic_lr):
        """Perform one step of training on a given minibatch.

        Parameters
        ----------FIXME
        batch : blank
            blank
        target_q : blank
            blank
        actor_lr : blank
            blank
        critic_lr : blank
            blank

        Returns
        -------
        blank
            blank
        blank
            blank
        """
        actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(
            [self.actor_grads, self.actor_loss, self.critic_grads,
             self.critic_loss],
            feed_dict={
                self.obs_ph: batch['obs0'],
                self.action_ph: batch['actions'],
                self.critic_target: target_q,
            }
        )

        self.actor_optimizer.update(actor_grads, learning_rate=actor_lr)
        self.critic_optimizer.update(critic_grads, learning_rate=critic_lr)

        return actor_loss, critic_loss


class LSTMPolicy(FullyConnectedPolicy):
    """Policy object that implements a DDPG-like actor critic, using LSTMs."""

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 actor_layers=None,
                 critic_layers=None,
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=tf.nn.tanh,
                 obs_rms=None,
                 ret_rms=None,
                 observation_range=(-np.inf, np.inf),
                 return_range=(-np.inf, np.inf)):
        """See parent class."""
        if actor_layers is None:
            actor_layers = [64]

        assert len(actor_layers) == 1, 'Only one LSTM layer is allowed.'

        super(LSTMPolicy, self).__init__(
            sess, ob_space, ac_space, actor_layers, critic_layers,
            hidden_nonlinearity, output_nonlinearity, obs_rms, ret_rms,
            observation_range, return_range)

        # input placeholder to the internal states component of the actor
        self.states_ph = None

        # output internal states component from the actor
        self.state = None

        # initial internal state of the actor
        self.state_init = None

        # placeholders for the actor recurrent network
        self.train_length = tf.placeholder(dtype=tf.int32, name='train_length')
        self.batch_size = tf.placeholder(dtype=tf.int32, name='batch_size')

    def make_actor(self, reuse=False, scope="pi"):
        """See parent class."""
        # convert batch to sequence
        obs = tf.reshape(
            tf.layers.flatten(self.normalized_obs_ph),
            [self.batch_size, self.train_length, self.ob_space.shape[0]]
        )

        # create the LSTM model
        self.policy, self.state_init, self.states_ph, self.state = build_lstm(
            inputs=obs,
            num_outputs=self.ac_space.shape[0],
            hidden_size=self.actor_layers[0],
            scope=scope,
            reuse=reuse,
            nonlinearity=tf.nn.tanh,
        )

    def step(self, obs, state=None, mask=None, **kwargs):
        """See parent class."""
        trace_length = state[0].shape[0]
        batch_size = int(obs.shape[0] / trace_length)
        return self.sess.run(
            [self.policy, self.state],
            feed_dict={
                self.obs_ph: obs,
                self.states_ph: state,
                self.train_length: trace_length,
                self.batch_size: batch_size
            }
        )

    def value(self,
              obs,
              action,
              state=None,
              mask=None,
              use_actor=False,
              **kwargs):
        """See parent class."""
        if use_actor:
            trace_length = state[0].shape[0]
            batch_size = int(obs.shape[0] / trace_length)
            return self.sess.run(
                self._critic_with_actor,
                feed_dict={
                    self.obs_ph: obs,
                    self.states_ph: state,
                    self.train_length: trace_length,
                    self.batch_size: batch_size,
                }
            )
        else:
            return self.sess.run(
                self._critic,
                feed_dict={
                    self.obs_ph: obs,
                    self.action_ph: action
                }
            )

    def train_actor_critic(self, batch, target_q, actor_lr, critic_lr):
        """See parent class.

        This method is further expanded to include placeholders needed by the
        recurrent actors.
        """
        trace_length = 8  # FIXME: trace length
        batch_size = int(batch['obs0'].shape[0] / trace_length)

        actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(
            [self.actor_grads, self.actor_loss, self.critic_grads,
             self.critic_loss],
            feed_dict={
                self.obs_ph: batch['obs0'],
                self.action_ph: batch['actions'],
                self.critic_target: target_q,
                self.batch_size: batch_size,
                self.train_length: trace_length,
                self.states_ph: (np.zeros([batch_size, self.actor_layers[0]]),
                                 np.zeros([batch_size, self.actor_layers[0]])),
            }
        )

        self.actor_optimizer.update(actor_grads, learning_rate=actor_lr)
        self.critic_optimizer.update(critic_grads, learning_rate=critic_lr)

        return actor_loss, critic_loss


# TODO
class FeudalPolicy(LSTMPolicy):
    """Policy object for the Feudal (FuN) hierarchical model.

    blank
    """
    pass


class HIROPolicy(LSTMPolicy):
    """Policy object for the HIRO hierarchical model.

    In this policy, we consider two actors and two critic, one for the manager
    and another for the worker.

    The manager receives state information and returns goals, while the worker
    receives the states and goals and returns an action. Moreover, while the
    worker performs actions at every time step, the manager only does so every
    c time steps (he is informed when to act from the algorithm).

    The manager is trained to maximize the total expected return from the
    environment, while the worker is rewarded to achieving states that are
    similar to those requested by the manager.

    See: https://arxiv.org/abs/1805.08296
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 actor_layers=None,
                 critic_layers=None,
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=tf.nn.tanh,
                 obs_rms=None,
                 ret_rms=None,
                 observation_range=(-np.inf, np.inf),
                 return_range=(-np.inf, np.inf)):
        """See parent class."""
        if actor_layers is None:
            actor_layers = [32]
        if critic_layers is None:
            critic_layers = [64, 64]

        assert len(actor_layers) == 1, 'Only one LSTM layer is allowed.'

        super(HIROPolicy, self).__init__(
            sess, ob_space, ac_space, actor_layers, critic_layers,
            hidden_nonlinearity, output_nonlinearity, obs_rms, ret_rms,
            observation_range, return_range)

        # manager policy
        self.manager = None

        # worker policy
        self.worker = None

        # manager value function
        self.manager_vf = None
        self._manager_vf = None

        # worker value function
        self.worker_vf = None
        self._worker_vf = None

        # a placeholder for goals that are communicated to the worker
        self.goal_ph = tf.placeholder(dtype=tf.float32,
                                      shape=[None, self.ob_space.shape[0]],
                                      name='goal_ph')

        # observation from the previous time step
        self.prev_obs = None

    def make_actor(self, reuse=False, scope='pi'):
        """Create an actor object.

        Parameters
        ----------
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor

        Returns
        -------
        tuple of tf.Tensor
            the output tensor from the manager and the worker
        """
        obs = self.normalized_obs_ph

        # input to the manager
        manager_in = tf.reshape(
            tf.layers.flatten(obs),
            [self.batch_size, self.train_length, self.ob_space.shape[0]]
        )

        # create the policy for the manager
        self.manager, m_state_init, m_states_ph, m_state = build_lstm(
            inputs=manager_in,
            num_outputs=self.ob_space.shape[0],
            hidden_size=self.actor_layers[0],
            scope='{}/manager'.format(scope),
            reuse=reuse,
            nonlinearity=self.output_nonlinearity,
        )

        # input to the worker
        worker_in = tf.reshape(
            tf.concat([tf.layers.flatten(obs),
                       tf.layers.flatten(self.goal_ph)], axis=-1),
            [self.batch_size, self.train_length, 2 * self.ob_space.shape[0]]
        )

        # create the policy for the worker
        self.worker, w_state_init, w_states_ph, w_state = build_lstm(
            inputs=worker_in,
            num_outputs=self.ac_space.shape[0],
            hidden_size=self.actor_layers[0],
            scope='{}/worker'.format(scope),
            reuse=reuse,
            nonlinearity=self.output_nonlinearity,
        )

        self.policy = tf.concat([self.manager, self.worker], axis=1)
        self.state_init = [m_state_init, w_state_init]
        self.states_ph = [m_states_ph, w_states_ph]
        self.state = [m_state, w_state]

    def make_critic(self, obs=None, reuse=False, scope='qf'):
        """Create a critic object.

        Parameters
        ----------
        obs : tf.Tensor
            The observation placeholder (can be None for default placeholder)
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the critic

        Returns
        -------
        tuple of tf.Tensor
            the output tensor for the manager and the worker
        """
        vf_h = tf.layers.flatten(self.normalized_obs_ph)

        # PART 1. WITHOUT ACTORS

        # value function for the manager
        m_normalized_critic = build_fcnet(
            inputs=tf.concat([vf_h, tf.layers.flatten(self.goal_ph)], axis=-1),
            num_outputs=1,
            hidden_size=self.critic_layers,
            hidden_nonlinearity=self.hidden_nonlinearity,
            output_nonlinearity=self.output_nonlinearity,
            scope=scope,
            reuse=False,
            prefix='manager/normalized_critic'
        )
        m_critic = denormalize(
            tf.clip_by_value(m_normalized_critic,
                             self.return_range[0],
                             self.return_range[1]),
            self.ret_rms)
        _m_critic = m_critic[:, 0]

        # value function for the worker
        w_normalized_critic = build_fcnet(
            inputs=tf.concat(values=[
                tf.concat([vf_h, tf.layers.flatten(self.goal_ph)], axis=-1),
                self.action_ph], axis=-1),
            num_outputs=1,
            hidden_size=self.critic_layers,
            hidden_nonlinearity=self.hidden_nonlinearity,
            output_nonlinearity=self.output_nonlinearity,
            scope=scope,
            reuse=False,
            prefix='worker/normalized_critic'
        )
        w_critic = denormalize(
            tf.clip_by_value(w_normalized_critic,
                             self.return_range[0],
                             self.return_range[1]),
            self.ret_rms)
        _w_critic = w_critic[:, 0]

        # PART 2. WITH ACTORS

        # value function for the manager
        m_normalized_critic_with_actor = build_fcnet(
            inputs=tf.concat([vf_h, tf.layers.flatten(self.manager)], axis=-1),
            num_outputs=1,
            hidden_size=self.critic_layers,
            hidden_nonlinearity=self.hidden_nonlinearity,
            output_nonlinearity=self.output_nonlinearity,
            scope=scope,
            reuse=True,
            prefix='manager/normalized_critic'
        )
        m_critic_with_actor = denormalize(
            tf.clip_by_value(m_normalized_critic_with_actor,
                             self.return_range[0],
                             self.return_range[1]),
            self.ret_rms)
        _m_critic_with_actor = m_critic_with_actor[:, 0]

        # value function for the worker
        w_normalized_critic_with_actor = build_fcnet(
            inputs=tf.concat(values=[
                tf.concat([vf_h, tf.layers.flatten(self.manager)], axis=-1),
                tf.layers.flatten(self.worker)], axis=-1),
            num_outputs=1,
            hidden_size=self.critic_layers,
            hidden_nonlinearity=self.hidden_nonlinearity,
            output_nonlinearity=self.output_nonlinearity,
            scope=scope,
            reuse=True,
            prefix='worker/normalized_critic'
        )
        w_critic_with_actor = denormalize(
            tf.clip_by_value(w_normalized_critic_with_actor,
                             self.return_range[0],
                             self.return_range[1]),
            self.ret_rms)
        _w_critic_with_actor = w_critic_with_actor[:, 0]

        # PART 3. FORWARD-FACING REPRESENTATION
        self.normalized_critic = [m_normalized_critic, w_normalized_critic]
        self.critic = [m_critic, w_critic]
        self._critic = [_m_critic, _w_critic]
        self.critic_with_actor = [m_critic_with_actor, w_critic_with_actor]
        self._critic_with_actor = [_m_critic_with_actor, _w_critic_with_actor]

    def setup_actor_optimizer(self, clip_norm, verbose):
        """See parent class.

        Separate optimizers are generated for the manager and the worker, with
        the same variables now holding a list of losses, gradients, and
        optimizers.
        """
        if verbose >= 2:
            logger.info('setting up actor optimizer')

        self.actor_loss = []
        self.actor_grads = []
        self.actor_optimizer = []

        for i, agent in enumerate(['manager', 'worker']):
            self.actor_loss.append(-tf.reduce_mean(self.critic_with_actor[i]))

            actor_shapes = [
                var.get_shape().as_list()
                for var in
                tf_util.get_trainable_vars('model/pi/{}'.format(agent))]
            actor_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                   for shape in actor_shapes])

            if verbose >= 2:
                logger.info('  {} actor shapes: {}'.format(
                    agent, actor_shapes))
                logger.info('  {} actor params: {}'.format(
                    agent, actor_nb_params))

            self.actor_grads.append(tf_util.flatgrad(
                self.actor_loss[i],
                tf_util.get_trainable_vars('model/pi/{}'.format(agent)),
                clip_norm=clip_norm))

            self.actor_optimizer.append(MpiAdam(
                var_list=tf_util.get_trainable_vars(
                    'model/pi/{}'.format(agent)),
                beta1=0.9, beta2=0.999, epsilon=1e-08))

    def setup_critic_optimizer(self,
                               critic_l2_reg,
                               clip_norm,
                               verbose):
        """See parent class.

        Separate optimizers are generated for the manager and the worker, with
        the same variables now holding a list of losses, gradients, and
        optimizers.
        """
        if verbose >= 2:
            logger.info('setting up critic optimizer')

        self.critic_loss = []
        self.critic_grads = []
        self.critic_optimizer = []

        for i, agent in enumerate(['manager', 'worker']):
            normalized_critic_target_tf = tf.clip_by_value(
                normalize(self.critic_target, self.ret_rms),
                self.return_range[0],
                self.return_range[1])

            self.critic_loss.append(tf.reduce_mean(tf.square(
                self.normalized_critic[i] - normalized_critic_target_tf)))

            if critic_l2_reg > 0.:
                critic_reg_vars = [
                    var for var
                    in tf_util.get_trainable_vars('model/qf/{}'.format(agent))
                    if 'bias' not in var.name and 'output'
                       not in var.name and 'b' not in var.name]
                if verbose >= 2:
                    for var in critic_reg_vars:
                        logger.info('  regularizing: {}'.format(var.name))
                    logger.info('  applying l2 regularization with {}'.
                                format(critic_l2_reg))

                critic_reg = tf.contrib.layers.apply_regularization(
                    tf.contrib.layers.l2_regularizer(critic_l2_reg),
                    weights_list=critic_reg_vars
                )
                self.critic_loss[i] += critic_reg
            critic_shapes = [
                var.get_shape().as_list() for var in
                tf_util.get_trainable_vars('model/qf/{}'.format(agent))]
            critic_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                    for shape in critic_shapes])

            if verbose >= 2:
                logger.info('  {} critic shapes: {}'.format(
                    agent, critic_shapes))
                logger.info('  {} critic params: {}'.format(
                    agent, critic_nb_params))

            self.critic_grads.append(tf_util.flatgrad(
                self.critic_loss[i],
                tf_util.get_trainable_vars('model/qf/{}'.format(agent)),
                clip_norm=clip_norm))

            self.critic_optimizer.append(MpiAdam(
                var_list=tf_util.get_trainable_vars(
                    'model/qf/{}'.format(agent)),
                beta1=0.9, beta2=0.999, epsilon=1e-08))

    def step(self, obs, state=None, mask=None, **kwargs):
        """See parent class."""
        state1 = deepcopy(state)
        trace_length = state[0][0].shape[0]
        batch_size = int(obs.shape[0] / trace_length)

        if kwargs['apply_manager']:
            goal, state1[0] = self.sess.run(
                [self.manager, self.state[0]],
                feed_dict={
                    self.obs_ph: obs[:, :self.ob_space.shape[0]],
                    self.states_ph[0]: state[0],
                    self.train_length: trace_length,
                    self.batch_size: batch_size
                }
            )
        else:
            goal = self._update_goal(obs, self.prev_obs)

        action, state1[1] = self.sess.run(
            [self.worker, self.state[1]],
            feed_dict={
                self.obs_ph: obs[:, :self.ob_space.shape[0]],
                self.states_ph[1]: state[1],
                self.goal_ph: obs[:, self.ob_space.shape[0]:],
                self.train_length: trace_length,
                self.batch_size: batch_size
            }
        )

        self.prev_obs = obs.copy()
        return (action, goal), state1

    def value(self,
              obs,
              action,
              state=None,
              mask=None,
              use_actor=False,
              **kwargs):
        """See parent class."""
        trace_length = state[0][0].shape[0]
        batch_size = int(obs.shape[0] / trace_length)

        m_feed_dict = {self.obs_ph: obs[:, :self.ob_space.shape[0]],
                       self.train_length: trace_length,
                       self.batch_size: batch_size}
        w_feed_dict = {self.obs_ph: obs[:, :self.ob_space.shape[0]],
                       self.goal_ph: obs[:, self.ob_space.shape[0]:],
                       self.train_length: trace_length,
                       self.batch_size: batch_size}

        if use_actor:
            _crtiic = self._critic_with_actor
            m_feed_dict[self.states_ph[0]] = state[0]
            w_feed_dict[self.states_ph[0]] = state[0]
            w_feed_dict[self.states_ph[1]] = state[1]
        else:
            _crtiic = self._critic
            m_feed_dict[self.goal_ph] = action[:, self.ac_space.shape[0]:]
            w_feed_dict[self.action_ph] = action[:, :self.ac_space.shape[0]]

        if kwargs['apply_manager']:
            v_manager = self.sess.run(_crtiic[0], feed_dict=m_feed_dict)
        else:
            v_manager = np.array([0 for _ in range(obs.shape[0])])

        v_worker = self.sess.run(_crtiic[1], feed_dict=w_feed_dict)

        return v_manager, v_worker

    def _update_goal(self, obs, prev_obs):
        """Update the goal when the manager isn't issuing commands.

        Parameters
        ----------
        obs : list of float or np.ndarray
            observations from the current time step
        prev_obs : list of float or np.ndarray
            observations from the previous time step

        Returns
        -------
        list of float or list of int
            current step goals
        """
        obs = obs[:, :self.ob_space.shape[0]]
        prev_goal = prev_obs[:, self.ob_space.shape[0]:]
        prev_obs = prev_obs[:, :self.ob_space.shape[0]]
        return prev_obs + prev_goal - obs

    def train_actor_critic(self, batch, target_q, actor_lr, critic_lr):
        """See parent class.

        This method is further expanded to include placeholders needed by the
        recurrent actors.
        """
        trace_length = 8  # FIXME: trace length
        batch_size = int(batch[0]['obs0'].shape[0] / trace_length)
        actor_loss, critic_loss = [], []

        init_state = (np.zeros([batch_size, self.actor_layers[0]]),
                      np.zeros([batch_size, self.actor_layers[0]]))

        for i in range(2):
            feed_dict = {
                self.critic_target: target_q[i],
                self.batch_size: batch_size,
                self.train_length: trace_length,
                self.states_ph[i]: deepcopy(init_state),
            }
            if i == 0:
                feed_dict[self.obs_ph] = batch[i]['obs0']
                feed_dict[self.goal_ph] = batch[i]['actions']
            else:
                feed_dict[self.action_ph] = batch[i]['actions']
                feed_dict[self.obs_ph] = batch[i]['obs0'][
                    :, :self.ob_space.shape[0]]
                feed_dict[self.goal_ph] = batch[i]['obs0'][
                    :, self.ob_space.shape[0]:]
                feed_dict[self.states_ph[0]] = deepcopy(init_state)

            a_grads, a_loss, c_grads, c_loss = self.sess.run(
                [self.actor_grads[i], self.actor_loss[i], self.critic_grads[i],
                 self.critic_loss[i]], feed_dict=feed_dict)

            actor_loss.append(a_loss)
            critic_loss.append(c_loss)
            self.actor_optimizer[i].update(a_grads, learning_rate=actor_lr)
            self.critic_optimizer[i].update(c_grads, learning_rate=critic_lr)

        return actor_loss, critic_loss
