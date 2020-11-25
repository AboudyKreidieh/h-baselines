"""Script containing the fcnet variant of the imitation learning policy."""
import tensorflow as tf

from hbaselines.base_policies import ImitationLearningPolicy
from hbaselines.fcnet.sac import LOG_STD_MIN
from hbaselines.fcnet.sac import LOG_STD_MAX
from hbaselines.fcnet.replay_buffer import ReplayBuffer
from hbaselines.utils.tf_util import layer
from hbaselines.utils.tf_util import reduce_std
from hbaselines.utils.tf_util import gaussian_likelihood
from hbaselines.utils.tf_util import apply_squashing_func
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.utils.tf_util import print_params_shape


class FeedForwardPolicy(ImitationLearningPolicy):
    """Fully-connected neural network imitation learning policy.

    Attributes
    ----------
    replay_buffer : hbaselines.fcnet.replay_buffer.ReplayBuffer
        the replay buffer
    action_ph : tf.compat.v1.placeholder
        placeholder for the actions
    obs_ph : tf.compat.v1.placeholder
        placeholder for the observations
    policy : tf.Variable
        the output from the imitation learning policy
    logp_ac : tf.Operation
        the operation that computes the log-probability of a given action. Only
        applies to stochastic policies.
    loss : tf.Operation
        the operation that computes the loss
    optimizer : tf.Operation
        the operation that updates the trainable parameters of the policy
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 co_space,
                 buffer_size,
                 batch_size,
                 learning_rate,
                 verbose,
                 layer_norm,
                 layers,
                 act_fun,
                 use_huber,
                 stochastic,
                 scope=None):
        """Instantiate the policy object.

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
        learning_rate : float
            the learning rate for the policy
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        layer_norm : bool
            enable layer normalisation
        layers : list of int or None
            the size of the Neural network for the policy
        act_fun : tf.nn.*
            the activation function to use in the neural network
        use_huber : bool
            specifies whether to use the huber distance function as the loss
            function. If set to False, the mean-squared error metric is used
            instead
        stochastic : bool
            specifies whether the policies are stochastic or deterministic
        scope : str
            an upper-level scope term. Used by policies that call this one.
        """
        super(FeedForwardPolicy, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            co_space=co_space,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose,
            layer_norm=layer_norm,
            layers=layers,
            act_fun=act_fun,
            use_huber=use_huber,
            stochastic=stochastic
        )

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
            self.action_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ac_space.shape,
                name='actions')
            self.obs_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ob_dim,
                name='obs0')

        # =================================================================== #
        # Step 3: Create policy variables.                                    #
        # =================================================================== #

        self.policy = None
        self.logp_ac = None

        # Create networks and core TF parts that are shared across setup parts.
        with tf.compat.v1.variable_scope("model", reuse=False):
            if self.stochastic:
                self._setup_stochastic_policy(self.obs_ph, self.action_ph)
            else:
                self._setup_deterministic_policy(self.obs_ph)

        # =================================================================== #
        # Step 4: Setup the optimizer.                                        #
        # =================================================================== #

        self.loss = None
        self.optimizer = None

        with tf.compat.v1.variable_scope("Optimizer", reuse=False):
            if self.stochastic:
                self._setup_stochastic_optimizer(scope)
            else:
                self._setup_deterministic_optimizer(self.action_ph, scope)

        # =================================================================== #
        # Step 5: Setup the operations for computing model statistics.        #
        # =================================================================== #

        # Setup the running means and standard deviations of the model inputs
        # and outputs.
        self.stats_ops, self.stats_names = self._setup_stats(scope or "Model")

    def _setup_stochastic_policy(self, obs, action, reuse=False, scope="pi"):
        """Create the variables of a stochastic policy.

        Parameters
        ----------
        obs : tf.compat.v1.placeholder
            the input observation placeholder
        action : tf.compat.v1.placeholder
            the input action placeholder
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the policy
        """
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            pi_h = obs

            # create the hidden layers
            for i, layer_size in enumerate(self.layers):
                pi_h = layer(
                    pi_h,  layer_size, 'fc{}'.format(i),
                    act_fun=self.act_fun,
                    layer_norm=self.layer_norm
                )

            # create the output mean
            policy_mean = layer(
                pi_h, self.ac_space.shape[0], 'mean',
                act_fun=None,
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3)
            )

            # create the output log_std
            log_std = layer(
                pi_h, self.ac_space.shape[0], 'log_std',
                act_fun=None,
            )

        # OpenAI Variation to cap the standard deviation
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        std = tf.exp(log_std)

        # Reparameterization trick
        policy = policy_mean + tf.random.normal(tf.shape(policy_mean)) * std
        logp_pi = gaussian_likelihood(policy, policy_mean, log_std)
        logp_ac = gaussian_likelihood(action, policy_mean, log_std)

        # Apply squashing and account for it in the probability
        _, _, logp_ac = apply_squashing_func(policy_mean, action, logp_ac)
        _, policy, _ = apply_squashing_func(policy_mean, policy, logp_pi)

        # Store the variables under their respective parameters.
        self.policy = policy
        self.logp_ac = logp_ac

    def _setup_stochastic_optimizer(self, scope):
        """Create the loss and optimizer of a stochastic policy."""
        scope_name = 'model/pi/'
        if scope is not None:
            scope_name = scope + '/' + scope_name

        if self.verbose >= 2:
            print('setting up optimizer')
            print_params_shape(scope_name, "policy")

        # Define the loss function.
        self.loss = - tf.reduce_mean(self.logp_ac)

        # Create an optimizer object.
        optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)

        # Create the optimizer operation.
        self.optimizer = optimizer.minimize(
            loss=self.loss,
            var_list=get_trainable_vars(scope_name)
        )

    def _setup_deterministic_policy(self, obs, reuse=False, scope="pi"):
        """Create the variables of deterministic a policy.

        Parameters
        ----------
        obs : tf.compat.v1.placeholder
            the input observation placeholder
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the policy
        """
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            pi_h = obs

            # create the hidden layers
            for i, layer_size in enumerate(self.layers):
                pi_h = layer(
                    pi_h,  layer_size, 'fc{}'.format(i),
                    act_fun=self.act_fun,
                    layer_norm=self.layer_norm
                )

            # create the output layer
            policy = layer(
                pi_h, self.ac_space.shape[0], 'output',
                act_fun=tf.nn.tanh,
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3)
            )

            # scaling terms to the output from the policy
            ac_means = (self.ac_space.high + self.ac_space.low) / 2.
            ac_magnitudes = (self.ac_space.high - self.ac_space.low) / 2.

            policy = ac_means + ac_magnitudes * tf.to_float(policy)

        # Store the variables under their respective parameters.
        self.policy = policy

    def _setup_deterministic_optimizer(self, action, scope=None):
        """Create the loss and optimizer of a deterministic policy."""
        scope_name = 'model/pi/'
        if scope is not None:
            scope_name = scope + '/' + scope_name

        if self.verbose >= 2:
            print('setting up optimizer')
            print_params_shape(scope_name, "policy")

        # Choose the loss function.
        if self.use_huber:
            loss_fn = tf.compat.v1.losses.huber_loss
        else:
            loss_fn = tf.compat.v1.losses.mean_squared_error

        # Define the loss function.
        self.loss = loss_fn(action, self.policy)

        # Create an optimizer object.
        optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)

        # Create the optimizer operation.
        self.optimizer = optimizer.minimize(
            loss=self.loss,
            var_list=get_trainable_vars(scope_name)
        )

    def _setup_stats(self, base):
        """Create the running means and std of the model inputs and outputs.

        This method also adds the same running means and stds as scalars to
        tensorboard for additional storage.
        """
        ops = []
        names = []

        ops += [tf.reduce_mean(self.policy)]
        names += ['{}/reference_action_mean'.format(base)]
        ops += [reduce_std(self.policy)]
        names += ['{}/reference_action_std'.format(base)]

        ops += [tf.reduce_mean(self.loss)]
        names += ['{}/reference_loss_mean'.format(base)]
        ops += [reduce_std(self.loss)]
        names += ['{}/reference_loss_std'.format(base)]

        # Add all names and ops to the tensorboard summary.
        for op, name in zip(ops, names):
            tf.compat.v1.summary.scalar(name, op)

        return ops, names

    def update(self):
        """See parent class."""
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample():
            return

        # Get a batch.
        obs0, actions, _, _, _ = self.replay_buffer.sample()

        return self.update_from_batch(obs0, actions)

    def update_from_batch(self, obs0, actions):
        """Perform gradient update step given a batch of data.

        Parameters
        ----------
        obs0 : array_like
            batch of observations
        actions : array_like
            batch of actions executed given obs_batch
        """
        self.sess.run([self.optimizer], feed_dict={
            self.obs_ph: obs0,
            self.action_ph: actions,
        })

    def get_action(self, obs, context):
        """See parent class."""
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        # Compute the action by the policy.
        action = self.sess.run(self.policy, {self.obs_ph: obs})

        if self.stochastic:
            # Scale the action by the action space of the environment.
            ac_means = 0.5 * (self.ac_space.high + self.ac_space.low)
            ac_magnitudes = 0.5 * (self.ac_space.high - self.ac_space.low)
            action = ac_magnitudes * action + ac_means

        return action

    def store_transition(self, obs0, context0, action, obs1, context1):
        """See parent class."""
        # Add the contextual observation, if applicable.
        obs0 = self._get_obs(obs0, context0, axis=0)
        obs1 = self._get_obs(obs1, context1, axis=0)

        self.replay_buffer.add(obs0, action, 0, obs1, float(False))

    def get_td_map(self):
        """See parent class."""
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample():
            return {}

        # Get a batch.
        obs0, actions, _, _, _ = self.replay_buffer.sample()

        return self.get_td_map_from_batch(obs0, actions)

    def get_td_map_from_batch(self, obs0, actions):
        """Convert a batch to a td_map."""
        return {
            self.obs_ph: obs0,
            self.action_ph: actions,
        }
