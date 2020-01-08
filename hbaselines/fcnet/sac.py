"""GaussianPolicy."""
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.contrib.framework import nest

from hbaselines.fcnet.replay_buffer import ReplayBuffer
from hbaselines.fcnet.base import ActorCriticPolicy


def create_input(name, input_shape):
    """TODO

    Parameters
    ----------
    name : TODO
        TODO
    input_shape : TODO
        TODO

    Returns
    -------
    TODO
        TODO
    """
    input_ = tf.keras.layers.Input(
        shape=input_shape, name=name, dtype=tf.float32
    )
    return input_


def create_inputs(input_shapes):
    """TODO

    Parameters
    ----------
    input_shapes : TODO
        TODO

    Returns
    -------
    TODO
        TODO
    """
    inputs = nest.map_structure_with_paths(create_input, input_shapes)
    inputs_flat = nest.flatten(inputs)

    return inputs_flat


def feedforward_model(hidden_layer_sizes,
                      output_size,
                      activation='relu',
                      output_activation='linear',
                      name='feedforward_model',
                      *args,
                      **kwargs):
    """TODO

    Parameters
    ----------
    hidden_layer_sizes : TODO
        TODO
    output_size : TODO
        TODO
    activation : TODO
        TODO
    output_activation : TODO
        TODO
    name : TODO
        TODO
    args : TODO
        TODO
    kwargs : TODO
        TODO

    Returns
    -------
    TODO
        TODO
    """
    def cast_and_concat(x):
        x = nest.map_structure(
            lambda element: tf.cast(element, tf.float32), x)
        x = nest.flatten(x)
        x = tf.concat(x, axis=-1)
        return x

    model = tf.keras.Sequential((
        tf.keras.layers.Lambda(cast_and_concat),
        *[
            tf.keras.layers.Dense(
                hidden_layer_size, *args, activation=activation, **kwargs)
            for hidden_layer_size in hidden_layer_sizes
        ],
        tf.keras.layers.Dense(
            output_size, *args, activation=output_activation, **kwargs)
    ), name=name)

    return model


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
    TODO
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
                 squash=True,
                 target_entropy=None,
                 action_prior="uniform",
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
        squash : bool
            TODO
        target_entropy : float
            target entropy used when learning the entropy coefficient. If set
            to None, a heuristic value is used.
        action_prior : str
            TODO
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

        self.squash = squash
        self.action_prior = action_prior
        self.zero_fingerprint = zero_fingerprint
        self.fingerprint_dim = fingerprint_dim

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

        self.condition_inputs = None
        self.latents_model = None
        self.latents_input = None
        self.actions_model = None
        self.deterministic_actions_model = None
        self.actions_input = None
        self.log_pis_model = None
        self.diagnostics_model = None

        input_shapes = {
            "observations": [self.ob_space.shape[0]]
        }
        critic_input_shapes = {
            'observations': {"observations": [self.ob_space.shape[0]]},
            'actions': self.ac_space.shape
        }

        # Create networks and core TF parts that are shared across setup parts.
        self.make_actor(input_shapes)
        self.critic_tf = tuple(
            self.make_critic(critic_input_shapes) for _ in range(2))
        self.critic_target = tuple(
            tf.keras.models.clone_model(Q) for Q in self.critic_tf)

        # =================================================================== #
        # Step 4: Setup the optimizers for the actor and critic.              #
        # =================================================================== #

        self._training_ops = {}
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

    def make_actor(self, input_shapes):
        inputs_flat = create_inputs(input_shapes)

        def cast_and_concat(x):
            x = nest.map_structure(
                lambda element: tf.cast(element, tf.float32), x)
            x = nest.flatten(x)
            x = tf.concat(x, axis=-1)
            return x

        conditions = tf.keras.layers.Lambda(
            cast_and_concat
        )(inputs_flat)

        self.condition_inputs = inputs_flat

        shift_and_log_scale_diag = feedforward_model(
            hidden_layer_sizes=self.layers,
            output_size=np.prod(self.ac_space.shape) * 2,
            activation=self.act_fun,
            output_activation="linear"
        )(conditions)

        shift, log_scale_diag = tf.keras.layers.Lambda(
            lambda x: tf.split(x, num_or_size_splits=2, axis=-1)
        )(shift_and_log_scale_diag)

        batch_size = tf.keras.layers.Lambda(
            lambda x: tf.shape(input=x)[0])(conditions)

        base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(self.ac_space.shape),
            scale_diag=tf.ones(self.ac_space.shape))

        latents = tf.keras.layers.Lambda(
            lambda x: base_distribution.sample(x)
        )(batch_size)

        self.latents_model = tf.keras.Model(self.condition_inputs, latents)
        self.latents_input = tf.keras.layers.Input(
            shape=self.ac_space.shape, name='latents')

        def raw_actions_fn(inputs):
            shift, log_scale_diag, latents = inputs
            bijector = tfp.bijectors.Affine(
                shift=shift,
                scale_diag=tf.exp(log_scale_diag))
            return bijector.forward(latents)

        raw_actions = tf.keras.layers.Lambda(
            raw_actions_fn
        )((shift, log_scale_diag, latents))

        squash_bijector = (
            tfp.bijectors.Tanh()
            if self.squash
            else tfp.bijectors.Identity())

        actions = tf.keras.layers.Lambda(
            lambda x: squash_bijector.forward(x)
        )(raw_actions)
        self.actions_model = tf.keras.Model(self.condition_inputs, actions)

        deterministic_actions = tf.keras.layers.Lambda(
            lambda x: squash_bijector.forward(x)
        )(shift)

        self.deterministic_actions_model = tf.keras.Model(
            self.condition_inputs, deterministic_actions)

        def log_pis_fn(inputs):
            shift, log_scale_diag, actions = inputs
            base_distribution = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self.ac_space.shape),
                scale_diag=tf.ones(self.ac_space.shape))
            bijector = tfp.bijectors.Chain((
                squash_bijector,
                tfp.bijectors.Affine(
                    shift=shift,
                    scale_diag=tf.exp(log_scale_diag)),
            ))
            distribution = (
                tfp.distributions.TransformedDistribution(
                    distribution=base_distribution,
                    bijector=bijector))

            return distribution.log_prob(actions)[:, None]

        self.actions_input = tf.keras.layers.Input(
            shape=self.ac_space.shape, name='actions')

        log_pis = tf.keras.layers.Lambda(
            log_pis_fn)([shift, log_scale_diag, actions])

        log_pis_for_action_input = tf.keras.layers.Lambda(
            log_pis_fn)([shift, log_scale_diag, self.actions_input])

        self.log_pis_model = tf.keras.Model(
            (*self.condition_inputs, self.actions_input),
            log_pis_for_action_input)

        self.diagnostics_model = tf.keras.Model(
            self.condition_inputs,
            (shift, log_scale_diag, log_pis, raw_actions, actions))

    def make_critic(self, input_shapes):
        inputs_flat = create_inputs(input_shapes)

        q_function = feedforward_model(
            hidden_layer_sizes=self.layers, output_size=1)
        q_function = tf.keras.Model(inputs_flat, q_function(inputs_flat))
        q_function.observation_keys = None

        return q_function

    def log_pis(self, observations, actions):
        return self.log_pis_model([*observations, actions])

    def update(self, **kwargs):
        """Perform a gradient update step.

        Returns
        -------
        float
            critic loss
        float
            actor loss
        """
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample():
            return 0, 0

        # Get a batch
        obs0, actions, rewards, obs1, _, done1 = self.replay_buffer.sample()

        return self.update_from_batch(obs0, actions, rewards, obs1, done1)

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

        # Collect all update and loss call operations.
        step_ops = [
            self.critic_loss[0],
            self.critic_loss[1],
            self.actor_loss,
            self.alpha_loss,
            self.critic_optimizer[0],
            self.critic_optimizer[1],
            self.actor_optimizer,
            self.alpha_optimizer,
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
        q1_loss, q2_loss, actor_loss, *_ = self.sess.run(step_ops, feed_dict)

        # Run target update ops.
        self.update_target()

        return q1_loss + q2_loss, actor_loss

    def get_action(self, obs, context, apply_noise, random_actions):
        """See parent class."""
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        if random_actions:
            return self.ac_space.sample()
        elif apply_noise:
            return self.actions_model.predict(obs)
        else:
            return self.deterministic_actions_model.predict(obs)

    def value(self, obs, context, action):
        """See parent class."""
        return 0, 0  # FIXME

    def log_pis_np(self, observations, actions):
        return self.log_pis_model.predict([*observations, actions])

    def update_target(self, tau=None):
        tau = tau or self.tau

        for Q, Q_target in zip(self.critic_tf, self.critic_target):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _get_q_target(self):
        policy_inputs = nest.flatten({
            "observations": self.obs1_ph
        })
        next_actions = self.actions_model(policy_inputs)
        next_log_pis = self.log_pis(policy_inputs, next_actions)

        next_q_observations = {"observations": self.obs1_ph}
        next_q_inputs = nest.flatten({
            **next_q_observations,
            'actions': next_actions
        })
        next_qs_values = tuple(q(next_q_inputs) for q in self.critic_target)

        min_next_q = tf.reduce_min(next_qs_values, axis=0)
        next_values = min_next_q - self.alpha * next_log_pis

        terminals = tf.cast(self.terminals1, next_values.dtype)

        q_target = self.rew_ph + self.gamma * (1 - terminals) * next_values

        return tf.stop_gradient(q_target)

    def _setup_critic_optimizer(self, scope):
        """Create minimization operation for critic Q-function.

        Create a `tf.optimizer.minimize` operation for updating critic
        Q-function with gradient descent, and append it to `self._training_ops`
        attribute.

        See Equations (5, 6) in [1], for further information of the Q-function
        update rule.
        """
        q_target = self._get_q_target()
        assert q_target.shape.as_list() == [None, 1]

        q_observations = {"observations": self.obs_ph}
        q_inputs = nest.flatten({
            **q_observations, 'actions': self.action_ph})
        q_values = self._Q_values = tuple(q(q_inputs) for q in self.critic_tf)

        # choose the loss function
        if self.use_huber:
            loss_fn = tf.compat.v1.losses.huber_loss
        else:
            loss_fn = tf.compat.v1.losses.mean_squared_error

        self.critic_loss = tuple(
            loss_fn(q_target, q_value) for q_value in q_values)

        self.critic_optimizer = []

        for i, (q, q_loss) in enumerate(zip(self.critic_tf, self.critic_loss)):
            # create an optimizer object
            optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.critic_lr,
                name='{}_{}_optimizer'.format(q._name, i))

            # create the optimizer object
            self.critic_optimizer.append(optimizer.minimize(
                loss=q_loss,
                var_list=q.trainable_variables))

        # Add the new operations to training_ops
        self._training_ops.update({'Q': tf.group(self.critic_optimizer)})

    def _setup_actor_optimizer(self, scope):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating policy and
        entropy with gradient descent, and adds them to `self._training_ops`
        attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """
        policy_inputs = nest.flatten({"observations": self.obs_ph})
        actions = self.actions_model(policy_inputs)
        log_pis = self.log_pis(policy_inputs, actions)
        assert log_pis.shape.as_list() == [None, 1]

        # Create the temperature term.
        log_alpha = tf.compat.v1.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        self.alpha = tf.exp(log_alpha)

        # Compute the temperature loss.
        self.alpha_loss = -tf.reduce_mean(
            log_alpha * tf.stop_gradient(log_pis + self.target_entropy))

        # Create an optimizer object.
        alpha_optimizer = tf.compat.v1.train.AdamOptimizer(
            self.actor_lr,
            name='alpha_optimizer')

        # Create the optimizer for the alpha term.
        self.alpha_optimizer = alpha_optimizer.minimize(
            loss=self.alpha_loss,
            var_list=[log_alpha])

        # Add the new operations to training_ops
        self._training_ops.update({'temperature_alpha': self.alpha_optimizer})

        # TODO: describe
        if self.action_prior == 'normal':
            policy_prior = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self.ac_space.shape),
                scale_diag=tf.ones(self.ac_space.shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self.action_prior == 'uniform':
            policy_prior_log_probs = 0.0
        else:
            raise ValueError("action_prior must be one of: {'normal', "
                             "'uniform'}")

        q_observations = {"observations": self.obs_ph}
        q_inputs = nest.flatten({
            **q_observations, 'actions': actions})
        q_log_targets = tuple(q(q_inputs) for q in self.critic_tf)
        min_q_log_target = tf.reduce_min(q_log_targets, axis=0)

        # Compute the actor loss.
        self.actor_loss = tf.reduce_mean(
            self.alpha * log_pis - min_q_log_target - policy_prior_log_probs)

        # Create an optimizer object.
        actor_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.actor_lr,
            name="policy_optimizer")

        # Create the optimizer for the actor.
        self.actor_optimizer = actor_optimizer.minimize(
            loss=self.actor_loss,
            var_list=self.actions_model.trainable_variables)

        # Add the new operations to training_ops
        self._training_ops.update({'policy_train_op': self.actor_optimizer})

    def _setup_stats(self, scope):  # FIXME
        diagnosables = OrderedDict((
            ('Q_value', self._Q_values),
        ))

        diagnostic_metrics = OrderedDict((
            ('mean', tf.reduce_mean),
            ('std', lambda x: tfp.stats.stddev(x, sample_axis=None)),
        ))

        self._diagnostics_ops = OrderedDict([
            (f'{key}-{metric_name}', metric_fn(values))
            for key, values in diagnosables.items()
            for metric_name, metric_fn in diagnostic_metrics.items()
        ])

        return [], []

    def initialize(self):
        """See parent class."""
        self.update_target(tau=1.0)

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
