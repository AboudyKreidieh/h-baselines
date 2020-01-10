"""TD3-compatible feedforward policy."""
import tensorflow as tf
import numpy as np
from functools import reduce

from hbaselines.fcnet.base import ActorCriticPolicy
from hbaselines.fcnet.replay_buffer import ReplayBuffer
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.utils.tf_util import reduce_std


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
    noise : float
        scaling term to the range of the action space, that is subsequently
        used as the standard deviation of Gaussian noise added to the action if
        `apply_noise` is set to True in `get_action`
    target_policy_noise : float
        standard deviation term to the noise from the output of the target
        actor policy. See TD3 paper for more.
    target_noise_clip : float
        clipping term for the noise injected in the target actor policy
    zero_fingerprint : bool
        whether to zero the last two elements of the observations for the actor
        and critic computations. Used for the worker policy when fingerprints
        are being implemented.
    fingerprint_dim : int
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
    actor_tf : tf.Variable
        the output from the actor network
    critic_tf : list of tf.Variable
        the output from the critic networks. Two networks are used to stabilize
        training.
    critic_with_actor_tf : list of tf.Variable
        the output from the critic networks with the action provided directly
        by the actor policy
    target_init_updates : tf.Operation
        an operation that sets the values of the trainable parameters of the
        target actor/critic to match those actual actor/critic
    target_soft_updates : tf.Operation
        soft target update function
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
                 noise,
                 target_policy_noise,
                 target_noise_clip,
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
        noise : float
            scaling term to the range of the action space, that is subsequently
            used as the standard deviation of Gaussian noise added to the
            action if `apply_noise` is set to True in `get_action`
        target_policy_noise : float
            standard deviation term to the noise from the output of the target
            actor policy. See TD3 paper for more.
        target_noise_clip : float
            clipping term for the noise injected in the target actor policy
        scope : str
            an upper-level scope term. Used by policies that call this one.
        zero_fingerprint : bool
            whether to zero the last two elements of the observations for the
            actor and critic computations. Used for the worker policy when
            fingerprints are being implemented.
        fingerprint_dim : int
            the number of fingerprint elements in the observation. Used when
            trying to zero the fingerprint elements.

        Raises
        ------
        AssertionError
            if the layers is not a list of at least size 1
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

        # action magnitudes
        ac_mag = 0.5 * (ac_space.high - ac_space.low)

        self.noise = noise * ac_mag
        self.target_policy_noise = np.array([ac_mag * target_policy_noise])
        self.target_noise_clip = np.array([ac_mag * target_noise_clip])
        self.zero_fingerprint = zero_fingerprint
        self.fingerprint_dim = fingerprint_dim
        assert len(self.layers) >= 1, \
            "Error: must have at least one hidden layer for the policy."

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
            self.actor_tf = self.make_actor(self.obs_ph)
            self.critic_tf = [
                self.make_critic(self.obs_ph, self.action_ph,
                                 scope="qf_{}".format(i))
                for i in range(2)
            ]
            self.critic_with_actor_tf = [
                self.make_critic(self.obs_ph, self.actor_tf, reuse=True,
                                 scope="qf_{}".format(i))
                for i in range(2)
            ]

        with tf.compat.v1.variable_scope("target", reuse=False):
            # create the target actor policy
            actor_target = self.make_actor(self.obs1_ph)

            # smooth target policy by adding clipped noise to target actions
            target_noise = tf.random_normal(
                tf.shape(actor_target), stddev=self.target_policy_noise)
            target_noise = tf.clip_by_value(
                target_noise, -self.target_noise_clip, self.target_noise_clip)

            # clip the noisy action to remain in the bounds
            noisy_actor_target = tf.clip_by_value(
                actor_target + target_noise,
                self.ac_space.low,
                self.ac_space.high
            )

            # create the target critic policies
            critic_target = [
                self.make_critic(self.obs1_ph, noisy_actor_target,
                                 scope="qf_{}".format(i))
                for i in range(2)
            ]

        # Create the target update operations.
        init, soft = self._setup_target_updates(
            'model', 'target', scope, tau, verbose)
        self.target_init_updates = init
        self.target_soft_updates = soft

        # =================================================================== #
        # Step 4: Setup the optimizers for the actor and critic.              #
        # =================================================================== #

        with tf.compat.v1.variable_scope("Optimizer", reuse=False):
            self._setup_actor_optimizer(scope)
            self._setup_critic_optimizer(critic_target, scope)
            tf.compat.v1.summary.scalar('actor_loss', self.actor_loss)
            tf.compat.v1.summary.scalar('Q1_loss', self.critic_loss[0])
            tf.compat.v1.summary.scalar('Q2_loss', self.critic_loss[1])

        # =================================================================== #
        # Step 5: Setup the operations for computing model statistics.        #
        # =================================================================== #

        # Setup the running means and standard deviations of the model inputs
        # and outputs.
        self.stats_ops, self.stats_names = self._setup_stats(scope or "Model")

    def _setup_actor_optimizer(self, scope):
        """Create the actor loss, gradient, and optimizer."""
        if self.verbose >= 2:
            print('setting up actor optimizer')

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

        # compute the actor loss
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf[0])

        # create an optimizer object
        optimizer = tf.compat.v1.train.AdamOptimizer(self.actor_lr)

        self.actor_optimizer = optimizer.minimize(
            self.actor_loss,
            var_list=get_trainable_vars(scope_name)
        )

    def _setup_critic_optimizer(self, critic_target, scope):
        """Create the critic loss, gradient, and optimizer."""
        if self.verbose >= 2:
            print('setting up critic optimizer')

        # compute the target critic term
        with tf.compat.v1.variable_scope("loss", reuse=False):
            q_obs1 = tf.minimum(critic_target[0], critic_target[1])
            target_q = tf.stop_gradient(
                self.rew_ph + (1. - self.terminals1) * self.gamma * q_obs1)

            tf.compat.v1.summary.scalar('critic_target',
                                        tf.reduce_mean(target_q))

        # choose the loss function
        if self.use_huber:
            loss_fn = tf.compat.v1.losses.huber_loss
        else:
            loss_fn = tf.compat.v1.losses.mean_squared_error

        self.critic_loss = [loss_fn(q, target_q) for q in self.critic_tf]

        self.critic_optimizer = []

        for i, critic_loss in enumerate(self.critic_loss):
            scope_name = 'model/qf_{}/'.format(i)
            if scope is not None:
                scope_name = scope + '/' + scope_name

            if self.verbose >= 2:
                critic_shapes = [var.get_shape().as_list()
                                 for var in get_trainable_vars(scope_name)]
                critic_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                        for shape in critic_shapes])
                print('  critic shapes: {}'.format(critic_shapes))
                print('  critic params: {}'.format(critic_nb_params))

            # create an optimizer object
            optimizer = tf.compat.v1.train.AdamOptimizer(self.critic_lr)

            # create the optimizer object
            self.critic_optimizer.append(optimizer.minimize(
                loss=critic_loss,
                var_list=get_trainable_vars(scope_name)))

    def make_actor(self, obs, reuse=False, scope="pi"):
        """Create an actor tensor.

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
            the output from the actor
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

            # create the output layer
            policy = self._layer(
                pi_h, self.ac_space.shape[0], 'output',
                act_fun=tf.nn.tanh,
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3)
            )

            # scaling terms to the output from the policy
            ac_means = (self.ac_space.high + self.ac_space.low) / 2.
            ac_magnitudes = (self.ac_space.high - self.ac_space.low) / 2.

            policy = ac_means + ac_magnitudes * tf.to_float(policy)

        return policy

    def make_critic(self, obs, action, reuse=False, scope="qf"):
        """Create a critic tensor.

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

        Returns
        -------
        tf.Variable
            the output from the critic
        """
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # concatenate the observations and actions
            qf_h = tf.concat([obs, action], axis=-1)

            # zero out the fingerprint observations for the worker policy
            if self.zero_fingerprint:
                qf_h = self._remove_fingerprint(
                    qf_h,
                    self.ob_space.shape[0],
                    self.fingerprint_dim,
                    self.co_space.shape[0] + self.ac_space.shape[0]
                )

            # create the hidden layers
            for i, layer_size in enumerate(self.layers):
                qf_h = self._layer(
                    qf_h,  layer_size, 'fc{}'.format(i),
                    act_fun=self.act_fun,
                    layer_norm=self.layer_norm
                )

            # create the output layer
            qvalue_fn = self._layer(
                qf_h, 1, 'qf_output',
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3)
            )

        return qvalue_fn

    def update(self, update_actor=True, **kwargs):
        """Perform a gradient update step.

        **Note**; The target update soft updates occur at the same frequency as
        the actor update frequencies.

        Parameters
        ----------
        update_actor : bool
            specifies whether to update the actor policy. The critic policy is
            still updated if this value is set to False.

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

        return self.update_from_batch(obs0, actions, rewards, obs1, done1,
                                      update_actor=update_actor)

    def update_from_batch(self,
                          obs0,
                          actions,
                          rewards,
                          obs1,
                          terminals1,
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
        update_actor : bool, optional
            specified whether to perform gradient update procedures to the
            actor policy. Default set to True. Note that the update procedure
            for the critic is always performed when calling this method.

        Returns
        -------
        [float, float]
            Q1 loss, Q2 loss
        float
            actor loss
        """
        # Reshape to match previous behavior and placeholder shape.
        rewards = rewards.reshape(-1, 1)
        terminals1 = terminals1.reshape(-1, 1)

        # Update operations for the critic networks.
        step_ops = [self.critic_loss,
                    self.critic_optimizer[0],
                    self.critic_optimizer[1]]

        if update_actor:
            # Actor updates and target soft update operation.
            step_ops += [self.actor_loss,
                         self.actor_optimizer,
                         self.target_soft_updates]

        # Perform the update operations and collect the critic loss.
        critic_loss, *_vals = self.sess.run(step_ops, feed_dict={
            self.obs_ph: obs0,
            self.action_ph: actions,
            self.rew_ph: rewards,
            self.obs1_ph: obs1,
            self.terminals1: terminals1
        })

        # Extract the actor loss.
        actor_loss = _vals[2] if update_actor else 0

        return critic_loss, actor_loss

    def get_action(self, obs, context, apply_noise, random_actions):
        """See parent class."""
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        if random_actions:
            action = np.array([self.ac_space.sample()])
        else:
            action = self.sess.run(self.actor_tf, {self.obs_ph: obs})

            if apply_noise:
                # compute noisy action
                if apply_noise:
                    action += np.random.normal(0, self.noise, action.shape)

                # clip by bounds
                action = np.clip(action, self.ac_space.low, self.ac_space.high)

        return action

    def value(self, obs, context, action):
        """See parent class."""
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        return self.sess.run(
            self.critic_tf,
            feed_dict={self.obs_ph: obs, self.action_ph: action})

    def store_transition(self, obs0, context0, action, reward, obs1, context1,
                         done, is_final_step, evaluate=False):
        """See parent class."""
        if not evaluate:
            # Add the contextual observation, if applicable.
            obs0 = self._get_obs(obs0, context0, axis=0)
            obs1 = self._get_obs(obs1, context1, axis=0)

            # Modify the done mask in accordance with the TD3 algorithm. Done
            # masks that correspond to the final step are set to False.
            done = done and not is_final_step

            self.replay_buffer.add(obs0, action, reward, obs1, float(done))

    def initialize(self):
        """See parent class.

        This method syncs the actor and critic optimizers across CPUs, and
        initializes the target parameters to match the model parameters.
        """
        self.sess.run(self.target_init_updates)

    def _setup_stats(self, base="Model"):
        """Create the running means and std of the model inputs and outputs.

        This method also adds the same running means and stds as scalars to
        tensorboard for additional storage.
        """
        ops = []
        names = []

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
            tf.compat.v1.summary.scalar(name, op)

        return ops, names

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
