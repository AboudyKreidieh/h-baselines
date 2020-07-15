"""PPO-compatible feedforward policy."""
import numpy as np
import tensorflow as tf

from hbaselines.base_policies.base import Policy
from hbaselines.utils.tf_util import layer
from hbaselines.utils.tf_util import get_trainable_vars


class FeedForwardPolicy(Policy):
    """Feed-forward neural network policy.

    Attributes
    ----------
    learning_rate : float
        the learning rate
    ent_coef : float
        entropy coefficient for the loss calculation
    vf_coef : float
        value function coefficient for the loss calculation
    max_grad_norm : float
        the maximum value for the gradient clipping
    cliprange : float or callable
        clipping parameter, it can be a function
    cliprange_vf : float or callable
        clipping parameter for the value function, it can be a function. This
        is a parameter specific to the OpenAI implementation. If None is passed
        (default), then `cliprange` (that is used for the policy) will be used.
        IMPORTANT: this clipping depends on the reward scaling. To deactivate
        value function clipping (and recover the original PPO implementation),
        you have to pass a negative value (e.g. -1).
    zero_fingerprint : TODO
        TODO
    fingerprint_dim : TODO
        TODO
    rew_ph : tf.compat.v1.placeholder
        TODO
    action_ph : tf.compat.v1.placeholder
        TODO
    obs_ph : tf.compat.v1.placeholder
        TODO
    advs_ph : tf.compat.v1.placeholder
        TODO
    old_neglog_pac_ph : tf.compat.v1.placeholder
        TODO
    old_vpred_ph : tf.compat.v1.placeholder
        TODO
    action : tf.Variable
        TODO
    pi_mean : tf.Variable
        TODO
    pi_logstd : tf.Variable
        TODO
    pi_std : tf.Variable
        TODO
    neglogp : tf.Variable
        TODO
    value_fn : tf.Variable
        TODO
    value_flat : tf.Variable
        TODO
    entropy : tf.Variable
        TODO
    vf_loss : tf.Variable
        TODO
    pg_loss : tf.Variable
        TODO
    approxkl : tf.Variable
        TODO
    loss : tf.Variable
        TODO
    optimizer : tf.Operation
        TODO
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 co_space,
                 verbose,
                 layer_norm,
                 layers,
                 act_fun,
                 learning_rate,
                 ent_coef,
                 vf_coef,
                 max_grad_norm,
                 cliprange,
                 cliprange_vf,
                 scope=None,
                 zero_fingerprint=False,
                 fingerprint_dim=2):
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
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        layer_norm : bool
            enable layer normalisation
        layers : list of int or None
            the size of the Neural network for the policy
        act_fun : tf.nn.*
            the activation function to use in the neural network
        learning_rate : float
            the learning rate
        ent_coef : float
            entropy coefficient for the loss calculation
        vf_coef : float
            value function coefficient for the loss calculation
        max_grad_norm : float
            the maximum value for the gradient clipping
        cliprange : float or callable
            clipping parameter, it can be a function
        cliprange_vf : float or callable
            clipping parameter for the value function, it can be a function.
            This is a parameter specific to the OpenAI implementation. If None
            is passed (default), then `cliprange` (that is used for the policy)
            will be used. IMPORTANT: this clipping depends on the reward
            scaling. To deactivate value function clipping (and recover the
            original PPO implementation), you have to pass a negative value
            (e.g. -1).
        """
        super(FeedForwardPolicy, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            co_space=co_space,
            verbose=verbose,
            layer_norm=layer_norm,
            layers=layers,
            act_fun=act_fun,
        )

        self.learning_rate = learning_rate
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.cliprange = cliprange
        self.cliprange_vf = cliprange_vf
        self.zero_fingerprint = zero_fingerprint
        self.fingerprint_dim = fingerprint_dim

        # Compute the shape of the input observation space, which may include
        # the contextual term.
        ob_dim = self._get_ob_dim(ob_space, co_space)

        # =================================================================== #
        # Step 1: Create input variables.                                     #
        # =================================================================== #

        with tf.compat.v1.variable_scope("input", reuse=False):
            self.rew_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,),
                name='rewards')
            self.action_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ac_space.shape,
                name='actions')
            self.obs_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ob_dim,
                name='obs0')
            self.advs_ph = tf.placeholder(
                tf.float32,
                shape=(None,),
                name="advs_ph")
            self.old_neglog_pac_ph = tf.placeholder(
                tf.float32,
                shape=(None,),
                name="old_neglog_pac_ph")
            self.old_vpred_ph = tf.placeholder(
                tf.float32,
                shape=(None,),
                name="old_vpred_ph")

        # =================================================================== #
        # Step 2: Create actor and critic variables.                          #
        # =================================================================== #

        # Create networks and core TF parts that are shared across setup parts.
        with tf.variable_scope("model", reuse=False):
            # Create the policy.
            self.action, self.pi_mean, self.pi_logstd = self._create_mlp(
                input_ph=self.obs_ph,
                num_output=self.ac_space.shape[0],
                layers=self.layers,
                act_fun=self.act_fun,
                stochastic=True,
                layer_norm=self.layer_norm,
                reuse=False,
                scope="pi",
            )
            self.pi_std = tf.exp(self.pi_logstd)

            # Create a method the log-probability of current actions.
            self.neglogp = self._neglogp(self.action)

            # Create the value function.
            self.value_fn = self._create_mlp(
                input_ph=self.obs_ph,
                num_output=1,
                layers=self.layers,
                act_fun=self.act_fun,
                stochastic=False,
                layer_norm=self.layer_norm,
                reuse=False,
                scope="vf",
            )
            self.value_flat = self.value_fn[:, 0]

        # =================================================================== #
        # Step 4: Setup the optimizers for the actor and critic.              #
        # =================================================================== #

        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.approxkl = None
        self.loss = None
        self.optimizer = None

        with tf.compat.v1.variable_scope("Optimizer", reuse=False):
            self._setup_optimizers(scope)

        # =================================================================== #
        # Step 5: Setup the operations for computing model statistics.        #
        # =================================================================== #

        self._setup_stats(scope or "Model")

    def _create_mlp(self,
                    input_ph,
                    num_output,
                    layers,
                    act_fun,
                    stochastic,
                    layer_norm,
                    reuse=False,
                    scope="pi"):
        """Create a multi-layer perceptron (MLP) model.

        Parameters
        ----------
        input_ph : tf.compat.v1.placeholder
            the input placeholder
        num_output : int
            number of output elements from the model
        layers : list of int
            the size of the Neural network for the policy
        act_fun : tf.nn.*
            the activation function to use in the neural network
        stochastic : bool
            whether the output should be stochastic or deterministic. If
            stochastic, a tuple of two elements is returned (the mean and
            logstd)
        layer_norm : bool
            enable layer normalisation
        reuse : bool
            TODO
        scope : str
            TODO

        Returns
        -------
        tf.Variable or (tf.Variable, tf.Variable)
            the output from the model
        """
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            pi_h = input_ph

            # Zero out the fingerprint observations for the worker policy.
            if self.zero_fingerprint:
                pi_h = self._remove_fingerprint(
                    pi_h,
                    self.ob_space.shape[0],
                    self.fingerprint_dim,
                    self.co_space.shape[0]
                )

            # Create the hidden layers.
            for i, layer_size in enumerate(layers):
                pi_h = layer(
                    pi_h,  layer_size, 'fc{}'.format(i),
                    act_fun=act_fun,
                    layer_norm=layer_norm
                )

            if stochastic:
                # Create the output mean.
                policy_mean = layer(
                    pi_h, num_output, 'mean',
                    act_fun=None,
                    kernel_initializer=tf.random_uniform_initializer(
                        minval=-3e-3, maxval=3e-3)
                )

                # Create the output log_std.
                log_std = tf.get_variable(
                    name='logstd',
                    shape=[1, num_output],
                    initializer=tf.zeros_initializer()
                )

                # Create a method to sample from the distribution.
                std = tf.exp(log_std)
                action = policy_mean + std * tf.random_normal(
                    shape=tf.shape(policy_mean),
                    dtype=tf.float32
                )

                policy_out = (action, policy_mean, log_std)
            else:
                # Create the output layer.
                policy = layer(
                    pi_h, num_output, 'output',
                    act_fun=None,
                    kernel_initializer=tf.random_uniform_initializer(
                        minval=-3e-3, maxval=3e-3)
                )

                policy_out = policy

        return policy_out

    def _neglogp(self, x):
        """TODO."""
        return 0.5 * tf.reduce_sum(
            tf.square((x - self.pi_mean) / self.pi_std), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) \
            * tf.cast(tf.shape(x)[-1], tf.float32) \
            + tf.reduce_sum(self.pi_logstd, axis=-1)

    def _setup_optimizers(self, scope):
        """Setup the actor and critic optimizers."""
        scope_name = 'model/'
        if scope is not None:
            scope_name = scope + '/' + scope_name

        neglogpac = self._neglogp(self.action_ph)
        self.entropy = tf.reduce_sum(
            tf.reshape(self.pi_logstd, [-1])
            + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

        # Value function clipping: not present in the original PPO
        if self.cliprange_vf is None:
            # Default behavior (legacy from OpenAI baselines):
            # use the same clipping as for the policy
            self.cliprange_vf = self.cliprange

        if self.cliprange_vf < 0:
            # Original PPO implementation: no value function clipping.
            vpred_clipped = self.value_flat
        else:
            # Clip the different between old and new value
            # NOTE: this depends on the reward scaling
            vpred_clipped = self.old_vpred_ph + tf.clip_by_value(
                self.value_flat - self.old_vpred_ph,
                -self.cliprange_vf, self.cliprange_vf)

        vf_losses1 = tf.square(self.value_flat - self.rew_ph)
        vf_losses2 = tf.square(vpred_clipped - self.rew_ph)
        self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
        pg_losses = -self.advs_ph * ratio
        pg_losses2 = -self.advs_ph * tf.clip_by_value(
            ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        self.approxkl = .5 * tf.reduce_mean(
            tf.square(neglogpac - self.old_neglog_pac_ph))
        self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(
            tf.abs(ratio - 1.0), self.cliprange), tf.float32))
        self.loss = self.pg_loss - self.entropy * self.ent_coef \
            + self.vf_loss * self.vf_coef

        # Compute the gradients of the loss.
        var_list = get_trainable_vars(scope_name)
        grads = tf.gradients(self.loss, var_list)

        # Perform gradient clipping if requested.
        if self.max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(
                grads, self.max_grad_norm)
        grads = list(zip(grads, var_list))

        # Create the operation that applies the gradients.
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            epsilon=1e-5
        ).apply_gradients(grads)

    def _setup_stats(self, base):
        """Create the running means and std of the model inputs and outputs.

        This method also adds the same running means and stds as scalars to
        tensorboard for additional storage.
        """
        ops = {
            'reference_action_mean': tf.reduce_mean(self.pi_mean),
            'reference_action_std': tf.reduce_mean(self.pi_logstd),
            'rewards': tf.reduce_mean(self.rew_ph),
            'advantage': tf.reduce_mean(self.advs_ph),
            'old_neglog_action_probability': tf.reduce_mean(
                self.old_neglog_pac_ph),
            'old_value_pred': tf.reduce_mean(self.old_vpred_ph),
            'entropy_loss': self.entropy,
            'policy_gradient_loss': self.pg_loss,
            'value_function_loss': self.vf_loss,
            'approximate_kullback-leibler': self.approxkl,
            'clip_factor': self.clipfrac,
            'loss': self.loss,
        }

        # Add all names and ops to the tensorboard summary.
        for key in ops.keys():
            name = "{}/{}".format(base, key)
            op = ops[key]
            tf.compat.v1.summary.scalar(name, op)

    def initialize(self):
        """See parent class."""
        pass

    def get_action(self, obs, context, apply_noise, random_actions, env_num=0):
        """See parent class."""
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        return self.sess.run(
            [self.action if apply_noise else self.pi_mean,
             self.value_flat, self.neglogp],
            {self.obs_ph: obs}
        )

    def value(self, obs, context):
        """See parent class."""
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    def update(self, obs, context, returns, actions, values, neglogpacs):
        """See parent class."""
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        # Compute the advantages.
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.obs_ph: obs,
            self.action_ph: actions,
            self.advs_ph: advs,
            self.rew_ph: returns,
            self.old_neglog_pac_ph: neglogpacs,
            self.old_vpred_ph: values,
        }

        policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = \
            self.sess.run(
                [self.pg_loss,
                 self.vf_loss,
                 self.entropy,
                 self.approxkl,
                 self.clipfrac,
                 self.optimizer],
                td_map
            )

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac

    def get_td_map(self, obs, context, returns, actions, values, neglogpacs):
        """See parent class."""
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        # Compute the advantages.
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        return {
            self.obs_ph: obs,
            self.action_ph: actions,
            self.advs_ph: advs,
            self.rew_ph: returns,
            self.old_neglog_pac_ph: neglogpacs,
            self.old_vpred_ph: values,
        }
