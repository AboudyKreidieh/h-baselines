"""SAC-compatible feedforward policy."""
import tensorflow as tf
import numpy as np
from functools import reduce

from hbaselines.fcnet.base import ActorCriticPolicy
from hbaselines.fcnet.replay_buffer import ReplayBuffer
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.utils.tf_util import reduce_std


# Stabilizing term to avoid NaN (prevents division by zero or log of zero)
EPS = 1e-6
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class FeedForwardPolicy(ActorCriticPolicy):
    """Feed-forward neural network actor-critic policy.

    Attributes
    ----------
    sess : tf.compat.v1.Session
        the current TensorFlow session
    ob_space : gym.spaces.*
        the observation space of the environment
    ac_space : gym.spaces.*
        the action space of the environment
    co_space : gym.spaces.*
        the context space of the environment
    buffer_size : int
        the max number of transitions to store
    batch_size : int
        SGD batch size
    actor_lr : float
        actor learning rate
    critic_lr : float
        critic learning rate
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    layers : list of int
        the size of the Neural network for the policy
    tau : float
        target update rate
    gamma : float
        discount factor
    layer_norm : bool
        enable layer normalisation
    act_fun : tf.nn.*
        the activation function to use in the neural network
    use_huber : bool
        specifies whether to use the huber distance function as the loss for
        the critic. If set to False, the mean-squared error metric is used
        instead
    target_entropy : float
        target entropy used when learning the entropy coefficient
    zero_fingerprint : bool
        whether to zero the last two elements of the observations for the actor
        and critic computations. Used for the worker policy when fingerprints
        are being implemented.
    fingerprint_dim : bool
        the number of fingerprint elements in the observation. Used when trying
        to zero the fingerprint elements.
    replay_buffer : hbaselines.fcnet.replay_buffer.ReplayBuffer
        the replay buffer
    terminals1 : tf.compat.v1.placeholder
        placeholder for the next step terminals
    rew_ph : tf.compat.v1.placeholder
        placeholder for the rewards
    action_ph : tf.compat.v1.placeholder
        placeholder for the actions
    obs_ph : tf.compat.v1.placeholder
        placeholder for the observations
    obs1_ph : tf.compat.v1.placeholder
        placeholder for the next step observations
    deterministic_action : tf.Variable
        the output from the deterministic actor
    policy_out : tf.Variable
        the output from the stochastic actor
    logp_pi : tf.Variable
        the log-probability of a given observation given the output action from
        the policy
    qf1 : tf.Variable
        the output from the first Q-function
    qf2 : tf.Variable
        the output from the second Q-function
    value_fn : tf.Variable
        the output from the value function
    qf1_pi : tf.Variable
        the output from the first Q-function with the action provided directly
        by the actor policy
    qf2_pi : tf.Variable
        the output from the second Q-function with the action provided directly
        by the actor policy
    log_alpha : tf.Variable
        the log of the entropy coefficient
    alpha : tf.Variable
        the entropy coefficient
    value_target : tf.Variable
        the output from the target value function. Takes as input the next-step
        observations
    target_init_updates : tf.Operation
        an operation that sets the values of the trainable parameters of the
        target actor/critic to match those actual actor/critic
    target_soft_updates : tf.Operation
        soft target update function
    alpha_loss : tf.Operation
        the operation that returns the loss of the entropy term
    alpha_optimizer : tf.Operation
        the operation that updates the trainable parameters of the entropy term
    actor_loss : tf.Operation
        the operation that returns the loss of the actor
    actor_optimizer : tf.Operation
        the operation that updates the trainable parameters of the actor
    critic_loss : tf.Operation
        the operation that returns the loss of the critic
    critic_optimizer : tf.Operation
        the operation that updates the trainable parameters of the critic
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
                 verbose,
                 tau,
                 gamma,
                 layer_norm,
                 layers,
                 act_fun,
                 use_huber,
                 target_entropy,
                 scope=None,
                 zero_fingerprint=False,
                 fingerprint_dim=2):
        """Instantiate the feed-forward neural network policy.

        Parameters
        ----------
        sess : tf.compat.v1.Session
            the current TensorFlow session
        ob_space : gym.spaces.*
            the observation space of the environment
        ac_space : gym.spaces.*
            the action space of the environment
        co_space : gym.spaces.*
            the context space of the environment
        buffer_size : int
            the max number of transitions to store
        batch_size : int
            SGD batch size
        actor_lr : float
            actor learning rate
        critic_lr : float
            critic learning rate
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        tau : float
            target update rate
        gamma : float
            discount factor
        layer_norm : bool
            enable layer normalisation
        layers : list of int or None
            the size of the Neural network for the policy
        act_fun : tf.nn.*
            the activation function to use in the neural network
        use_huber : bool
            specifies whether to use the huber distance function as the loss
            for the critic. If set to False, the mean-squared error metric is
            used instead
        target_entropy : float
            target entropy used when learning the entropy coefficient. If set
            to None, a heuristic value is used.
        scope : str
            an upper-level scope term. Used by policies that call this one.
        zero_fingerprint : bool
            whether to zero the last two elements of the observations for the
            actor and critic computations. Used for the worker policy when
            fingerprints are being implemented.
        fingerprint_dim : bool
            the number of fingerprint elements in the observation. Used when
            trying to zero the fingerprint elements.
        """
        super(FeedForwardPolicy, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            co_space=co_space,
            buffer_size=buffer_size,
            batch_size=batch_size,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            verbose=verbose,
            tau=tau,
            gamma=gamma,
            layer_norm=layer_norm,
            layers=layers,
            act_fun=act_fun,
            use_huber=use_huber
        )

        if target_entropy is None:
            self.target_entropy = -np.prod(self.ac_space.shape)
        else:
            self.target_entropy = target_entropy

        self.zero_fingerprint = zero_fingerprint
        self.fingerprint_dim = fingerprint_dim
        self._ac_means = 0.5 * (ac_space.high + ac_space.low)
        self._ac_magnitudes = 0.5 * (ac_space.high - ac_space.low)

        # Compute the shape of the input observation space, which may include
        # the contextual term.
        ob_dim = self._get_ob_dim(ob_space, co_space)

        # =================================================================== #
        # Step 1: Create a replay buffer object.                              #
        # =================================================================== #

        self.replay_buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            obs_dim=ob_dim[0],
            ac_dim=self.ac_space.shape[0],
        )

        # =================================================================== #
        # Step 2: Create input variables.                                     #
        # =================================================================== #

        with tf.compat.v1.variable_scope("input", reuse=False):
            self.terminals1 = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='terminals1')
            self.rew_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='rewards')
            self.action_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ac_space.shape,
                name='actions')
            self.obs_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ob_dim,
                name='obs0')
            self.obs1_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ob_dim,
                name='obs1')

        # logging of rewards to tensorboard
        with tf.compat.v1.variable_scope("input_info", reuse=False):
            tf.compat.v1.summary.scalar('rewards', tf.reduce_mean(self.rew_ph))

        # =================================================================== #
        # Step 3: Create actor and critic variables.                          #
        # =================================================================== #

        # Create networks and core TF parts that are shared across setup parts.
        with tf.compat.v1.variable_scope("model", reuse=False):
            self.deterministic_action, self.policy_out, self.logp_pi = \
                self.make_actor(self.obs_ph)
            self.qf1, self.qf2, self.value_fn = self.make_critic(
                self.obs_ph, self.action_ph,
                create_qf=True, create_vf=True)
            self.qf1_pi, self.qf2_pi, _ = self.make_critic(
                self.obs_ph, self.policy_out,
                create_qf=True, create_vf=False, reuse=True)

            # The entropy coefficient or entropy can be learned automatically,
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            self.log_alpha = tf.compat.v1.get_variable(
                'log_alpha',
                dtype=tf.float32,
                initializer=0.0)
            self.alpha = tf.exp(self.log_alpha)

        with tf.compat.v1.variable_scope("target", reuse=False):
            # Create the value network
            _, _, value_target = self.make_critic(
                self.obs1_ph, create_qf=False, create_vf=True)
            self.value_target = value_target

        # Create the target update operations.
        init, soft = self._setup_target_updates(
            'model/value_fns/vf', 'target/value_fns/vf', scope, tau, verbose)
        self.target_init_updates = init
        self.target_soft_updates = soft

        # =================================================================== #
        # Step 4: Setup the optimizers for the actor and critic.              #
        # =================================================================== #

        with tf.compat.v1.variable_scope("Optimizer", reuse=False):
            self._setup_actor_optimizer(scope)
            self._setup_critic_optimizer(scope)
            tf.compat.v1.summary.scalar('alpha_loss', self.alpha_loss)
            tf.compat.v1.summary.scalar('actor_loss', self.actor_loss)
            tf.compat.v1.summary.scalar('Q1_loss', self.critic_loss[0])
            tf.compat.v1.summary.scalar('Q2_loss', self.critic_loss[1])

        # =================================================================== #
        # Step 5: Setup the operations for computing model statistics.        #
        # =================================================================== #

        # Setup the running means and standard deviations of the model inputs
        # and outputs.
        self.stats_ops, self.stats_names = self._setup_stats(scope or "Model")

    def make_actor(self, obs, reuse=False, scope="pi"):
        """Create the actor variables.

        Parameters
        ----------
        obs : tf.compat.v1.placeholder
            the input observation placeholder
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor

        Returns
        -------
        tf.Variable
            the output from the deterministic actor
        tf.Variable
            the output from the stochastic actor
        tf.Variable
            the log-probability of a given observation given the output action
            from the policy
        """
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            pi_h = obs

            # zero out the fingerprint observations for the worker policy
            if self.zero_fingerprint:
                pi_h = self._remove_fingerprint(
                    pi_h,
                    self.ob_space.shape[0],
                    self.fingerprint_dim,
                    self.co_space.shape[0]
                )

            # create the hidden layers
            for i, layer_size in enumerate(self.layers):
                pi_h = self._layer(
                    pi_h,  layer_size, 'fc{}'.format(i),
                    act_fun=self.act_fun,
                    layer_norm=self.layer_norm
                )

            # create the output mean
            policy_mean = self._layer(
                pi_h, self.ac_space.shape[0], 'mean',
                act_fun=None,
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3)
            )

            # create the output log_std
            log_std = self._layer(
                pi_h, self.ac_space.shape[0], 'log_std',
                act_fun=None,
            )

        # OpenAI Variation to cap the standard deviation
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        std = tf.exp(log_std)

        # Reparameterization trick
        policy = policy_mean + tf.random.normal(tf.shape(policy_mean)) * std
        logp_pi = self._gaussian_likelihood(policy, policy_mean, log_std)

        # Apply squashing and account for it in the probability
        deterministic_policy, policy, logp_pi = self._apply_squashing_func(
            policy, policy_mean, logp_pi)

        return deterministic_policy, policy, logp_pi

    def make_critic(self,
                    obs,
                    action=None,
                    reuse=False,
                    scope="value_fns",
                    create_qf=True,
                    create_vf=True):
        """Create the critic variables.

        Parameters
        ----------
        obs : tf.compat.v1.placeholder
            the input observation placeholder
        action : tf.compat.v1.placeholder
            the input action placeholder
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor
        create_qf : bool
            whether to create the Q-functions
        create_vf : bool
            whether to create the value function

        Returns
        -------
        tf.Variable
            the output from the first Q-function. Set to None if `create_qf` is
            False.
        tf.Variable
            the output from the second Q-function. Set to None if `create_qf`
            is False.
        tf.Variable
            the output from the value function. Set to None if `create_vf` is
            False.
        """
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # zero out the fingerprint observations for the worker policy
            if self.zero_fingerprint:
                obs = self._remove_fingerprint(
                    obs,
                    self.ob_space.shape[0],
                    self.fingerprint_dim,
                    self.co_space.shape[0]
                )

            # Value function
            if create_vf:
                with tf.compat.v1.variable_scope("vf", reuse=reuse):
                    vf_h = obs

                    # create the hidden layers
                    for i, layer_size in enumerate(self.layers):
                        vf_h = self._layer(
                            vf_h, layer_size, 'fc{}'.format(i),
                            act_fun=self.act_fun,
                            layer_norm=self.layer_norm
                        )

                    # create the output layer
                    value_fn = self._layer(
                        vf_h, 1, 'vf_output',
                        kernel_initializer=tf.random_uniform_initializer(
                            minval=-3e-3, maxval=3e-3)
                    )
            else:
                value_fn = None

            # Double Q values to reduce overestimation
            if create_qf:
                with tf.compat.v1.variable_scope('qf1', reuse=reuse):
                    # concatenate the observations and actions
                    qf1_h = tf.concat([obs, action], axis=-1)

                    # create the hidden layers
                    for i, layer_size in enumerate(self.layers):
                        qf1_h = self._layer(
                            qf1_h, layer_size, 'fc{}'.format(i),
                            act_fun=self.act_fun,
                            layer_norm=self.layer_norm
                        )

                    # create the output layer
                    qf1 = self._layer(
                        qf1_h, 1, 'qf_output',
                        kernel_initializer=tf.random_uniform_initializer(
                            minval=-3e-3, maxval=3e-3)
                    )

                with tf.compat.v1.variable_scope('qf2', reuse=reuse):
                    # concatenate the observations and actions
                    qf2_h = tf.concat([obs, action], axis=-1)

                    # create the hidden layers
                    for i, layer_size in enumerate(self.layers):
                        qf2_h = self._layer(
                            qf2_h, layer_size, 'fc{}'.format(i),
                            act_fun=self.act_fun,
                            layer_norm=self.layer_norm
                        )

                    # create the output layer
                    qf2 = self._layer(
                        qf2_h, 1, 'qf_output',
                        kernel_initializer=tf.random_uniform_initializer(
                            minval=-3e-3, maxval=3e-3)
                    )
            else:
                qf1, qf2 = None, None

        return qf1, qf2, value_fn

    @staticmethod
    def _gaussian_likelihood(input_, mu_, log_std):
        """Compute log likelihood of a gaussian.

        Here we assume this is a Diagonal Gaussian.

        Parameters
        ----------
        input_ : tf.Variable
            the action by the policy
        mu_ : tf.Variable
            the policy mean
        log_std : tf.Variable
            the policy log std

        Returns
        -------
        tf.Variable
            the log-probability of a given observation given the output action
            from the policy
        """
        pre_sum = -0.5 * (((input_ - mu_) / (
                    tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(
            2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    @staticmethod
    def _apply_squashing_func(mu_, pi_, logp_pi):
        """Squash the output of the Gaussian distribution.

        This method also accounts for that in the log probability.

        The squashed mean is also returned for using deterministic actions.

        Parameters
        ----------
        mu_ : tf.Variable
            mean of the gaussian
        pi_ : tf.Variable
            output of the policy before squashing
        logp_pi : tf.Variable
            log probability before squashing

        Returns
        -------
        tf.Variable
            the output from the first Q-function. Set to None if `create_qf` is
            False.
        tf.Variable
            the output from the second Q-function. Set to None if `create_qf`
            is False.
        tf.Variable
            the output from the value function. Set to None if `create_vf` is
            False.
        """
        # Squash the output
        deterministic_policy = tf.nn.tanh(mu_)
        policy = tf.nn.tanh(pi_)

        # Squash correction (from original implementation)
        logp_pi -= tf.reduce_sum(tf.math.log(1 - policy ** 2 + EPS), axis=1)

        return deterministic_policy, policy, logp_pi

    def update(self, **kwargs):
        """Perform a gradient update step.

        Returns
        -------
        [float, float]
            Q1 loss, Q2 loss
        float
            actor loss
        """
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample():
            return [0, 0], 0

        # Get a batch
        obs0, actions, rewards, obs1, _, done1 = self.replay_buffer.sample()

        return self.update_from_batch(obs0, actions, rewards, obs1, done1)

    def update_from_batch(self, obs0, actions, rewards, obs1, terminals1,
                          update_actor=True):
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
        update_actor : bool
            whether to update the actor policy. Unused by this method.

        Returns
        -------
        [float, float]
            Q1 loss, Q2 loss
        float
            actor loss
        """
        del update_actor  # unused by this method

        # Normalize the actions (bounded between [-1, 1]).
        actions = (actions - self._ac_means) / self._ac_magnitudes

        # Reshape to match previous behavior and placeholder shape.
        rewards = rewards.reshape(-1, 1)
        terminals1 = terminals1.reshape(-1, 1)

        # Collect all update and loss call operations.
        step_ops = [
            self.critic_loss[0],
            self.critic_loss[1],
            self.critic_loss[2],
            self.actor_loss,
            self.alpha_loss,
            self.critic_optimizer,
            self.actor_optimizer,
            self.alpha_optimizer,
            self.target_soft_updates,
        ]

        # Prepare the feed_dict information.
        feed_dict = {
            self.obs_ph: obs0,
            self.action_ph: actions,
            self.rew_ph: rewards,
            self.obs1_ph: obs1,
            self.terminals1: terminals1
        }

        # Perform the update operations and collect the actor and critic loss.
        q1_loss, q2_loss, vf_loss, actor_loss, *_ = self.sess.run(
            step_ops, feed_dict)

        return [q1_loss, q2_loss], actor_loss  # FIXME: add vf_loss

    def get_action(self, obs, context, apply_noise, random_actions):
        """See parent class."""
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        if random_actions:
            return np.array([self.ac_space.sample()])
        elif apply_noise:
            normalized_action = self.sess.run(
                self.policy_out, feed_dict={self.obs_ph: obs})
            return self._ac_magnitudes * normalized_action + self._ac_means
        else:
            normalized_action = self.sess.run(
                self.deterministic_action, feed_dict={self.obs_ph: obs})
            return self._ac_magnitudes * normalized_action + self._ac_means

    def value(self, obs, context, action):
        """See parent class."""
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        # Normalize the actions (bounded between [-1, 1]).
        action = (action - self._ac_means) / self._ac_magnitudes

        return self.sess.run(
            [self.qf1, self.qf2],  # , self.value_fn],  FIXME
            feed_dict={
                self.obs_ph: obs,
                self.action_ph: action
            }
        )

    def _setup_critic_optimizer(self, scope):
        """Create minimization operation for critic Q-function.

        Create a `tf.optimizer.minimize` operation for updating critic
        Q-function with gradient descent.

        See Equations (5, 6) in [1], for further information of the Q-function
        update rule.
        """
        if self.verbose >= 2:
            print('setting up critic optimizer')

        scope_name = 'model/value_fns'
        if scope is not None:
            scope_name = scope + '/' + scope_name

        if self.verbose >= 2:
            for name in ['qf1', 'qf2', 'vf']:
                actor_shapes = [
                    var.get_shape().as_list() for var in
                    get_trainable_vars('{}/{}'.format(scope_name, name))]
                actor_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                       for shape in actor_shapes])
                print('  {} shapes: {}'.format(name, actor_shapes))
                print('  {} params: {}'.format(name, actor_nb_params))

        # Take the min of the two Q-Values (Double-Q Learning)
        min_qf_pi = tf.minimum(self.qf1_pi, self.qf2_pi)

        # Target for Q value regression
        q_backup = tf.stop_gradient(
            self.rew_ph +
            (1 - self.terminals1) * self.gamma * self.value_target)

        # choose the loss function
        if self.use_huber:
            loss_fn = tf.compat.v1.losses.huber_loss
        else:
            loss_fn = tf.compat.v1.losses.mean_squared_error

        # Compute Q-Function loss
        qf1_loss = loss_fn(q_backup, self.qf1)
        qf2_loss = loss_fn(q_backup, self.qf2)

        # Target for value fn regression
        # We update the vf towards the min of two Q-functions in order to
        # reduce overestimation bias from function approximation error.
        v_backup = tf.stop_gradient(min_qf_pi - self.alpha * self.logp_pi)
        value_loss = loss_fn(self.value_fn, v_backup)

        self.critic_loss = (qf1_loss, qf2_loss, value_loss)

        # Combine the loss functions for the optimizer.
        critic_loss = qf1_loss + qf2_loss + value_loss

        # Critic train op
        critic_optimizer = tf.compat.v1.train.AdamOptimizer(self.critic_lr)
        self.critic_optimizer = critic_optimizer.minimize(
            critic_loss,
            var_list=get_trainable_vars(scope_name))

    def _setup_actor_optimizer(self, scope):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating policy and
        entropy with gradient descent.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """
        if self.verbose >= 2:
            print('setting up actor and alpha optimizers')

        scope_name = 'model/pi/'
        if scope is not None:
            scope_name = scope + '/' + scope_name

        if self.verbose >= 2:
            actor_shapes = [var.get_shape().as_list()
                            for var in get_trainable_vars(scope_name)]
            actor_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                   for shape in actor_shapes])
            print('  actor shapes: {}'.format(actor_shapes))
            print('  actor params: {}'.format(actor_nb_params))

        # Take the min of the two Q-Values (Double-Q Learning)
        min_qf_pi = tf.minimum(self.qf1_pi, self.qf2_pi)

        # Compute the entropy temperature loss.
        self.alpha_loss = -tf.reduce_mean(
            self.log_alpha
            * tf.stop_gradient(self.logp_pi + self.target_entropy))

        alpha_optimizer = tf.compat.v1.train.AdamOptimizer(self.actor_lr)

        self.alpha_optimizer = alpha_optimizer.minimize(
            self.alpha_loss,
            var_list=self.log_alpha)

        # Compute the policy loss
        self.actor_loss = tf.reduce_mean(self.alpha * self.logp_pi - min_qf_pi)

        # Policy train op (has to be separate from value train op, because
        # min_qf_pi appears in policy_loss)
        actor_optimizer = tf.compat.v1.train.AdamOptimizer(self.actor_lr)

        self.actor_optimizer = actor_optimizer.minimize(
            self.actor_loss,
            var_list=get_trainable_vars(scope_name))

    def _setup_stats(self, base):
        """Create the running means and std of the model inputs and outputs.

        This method also adds the same running means and stds as scalars to
        tensorboard for additional storage.
        """
        ops = []
        names = []

        ops += [tf.reduce_mean(self.qf1)]
        names += ['{}/reference_Q1_mean'.format(base)]
        ops += [reduce_std(self.qf1)]
        names += ['{}/reference_Q1_std'.format(base)]

        ops += [tf.reduce_mean(self.qf2)]
        names += ['{}/reference_Q2_mean'.format(base)]
        ops += [reduce_std(self.qf2)]
        names += ['{}/reference_Q2_std'.format(base)]

        ops += [tf.reduce_mean(self.qf1_pi)]
        names += ['{}/reference_actor_Q1_mean'.format(base)]
        ops += [reduce_std(self.qf1_pi)]
        names += ['{}/reference_actor_Q1_std'.format(base)]

        ops += [tf.reduce_mean(self.qf2_pi)]
        names += ['{}/reference_actor_Q2_mean'.format(base)]
        ops += [reduce_std(self.qf2_pi)]
        names += ['{}/reference_actor_Q2_std'.format(base)]

        ops += [tf.reduce_mean(self.policy_out)]
        names += ['{}/reference_action_mean'.format(base)]
        ops += [reduce_std(self.policy_out)]
        names += ['{}/reference_action_std'.format(base)]

        ops += [tf.reduce_mean(self.logp_pi)]
        names += ['{}/reference_log_probability_mean'.format(base)]
        ops += [reduce_std(self.logp_pi)]
        names += ['{}/reference_log_probability_std'.format(base)]

        # Add all names and ops to the tensorboard summary.
        for op, name in zip(ops, names):
            tf.compat.v1.summary.scalar(name, op)

        return ops, names

    def initialize(self):
        """See parent class."""
        self.sess.run(self.target_init_updates)

    def store_transition(self, obs0, context0, action, reward, obs1, context1,
                         done, is_final_step, evaluate=False):
        """See parent class."""
        if not evaluate:
            # Add the contextual observation, if applicable.
            obs0 = self._get_obs(obs0, context0, axis=0)
            obs1 = self._get_obs(obs1, context1, axis=0)

            self.replay_buffer.add(obs0, action, reward, obs1, float(done))

    def get_td_map(self):
        """See parent class."""
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample():
            return {}

        # Get a batch.
        obs0, actions, rewards, obs1, _, done1 = self.replay_buffer.sample()

        return self.get_td_map_from_batch(obs0, actions, rewards, obs1, done1)

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
