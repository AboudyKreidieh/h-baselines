import numpy as np
import tensorflow as tf

from hbaselines.base_policies.base import Policy
from hbaselines.utils.tf_util import layer
from hbaselines.utils.tf_util import get_trainable_vars


class DiagGaussianProbabilityDistribution(object):

    def __init__(self, flat):
        """
        Probability distributions from multivariate Gaussian input

        :param flat: ([float]) the multivariate Gaussian input data
        """
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2,
                                value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        # Bounds are taken into account outside this class (during training
        # only)
        return self.mean

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std),
                                   axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1],
                                                     tf.float32) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianProbabilityDistribution)
        return tf.reduce_sum(other.logstd - self.logstd + (
                    tf.square(self.std) + tf.square(self.mean - other.mean)) /
                             (2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e),
                             axis=-1)

    def sample(self):
        # Bounds are taken into acount outside this class (during training
        # only) Otherwise, it changes the distribution and breaks PPO2 for
        # instance
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean),
                                                       dtype=self.mean.dtype)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new multivariate Gaussian input

        :param flat: ([float]) the multivariate Gaussian input data
        :return: (ProbabilityDistribution) the instance from the given
            multivariate Gaussian input data
        """
        return cls(flat)

    def logp(self, x):
        """
        returns the of the log likelihood

        :param x: (str) the labels of each index
        :return: ([float]) The log likelihood of the distribution
        """
        return - self.neglogp(x)


class FeedForwardPolicy(Policy):
    """
    Policy object that implements actor critic

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param reuse: (bool) If the policy is reusable or not
    """

    recurrent = False

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 co_space,
                 verbose,
                 layer_norm,
                 layers,
                 act_fun,
                 use_huber,
                 gamma,
                 learning_rate,
                 ent_coef,
                 vf_coef,
                 max_grad_norm,
                 lam,
                 cliprange,
                 cliprange_vf,
                 scope=None,
                 reuse=False,
                 zero_fingerprint=False,
                 fingerprint_dim=2):
        """Instantiate the policy object.

        Parameters
        ----------
        sess : TODO
            TODO
        ob_space : TODO
            TODO
        ac_space : TODO
            TODO
        co_space : TODO
            TODO
        verbose : TODO
            TODO
        layer_norm : TODO
            TODO
        layers : TODO
            TODO
        act_fun : TODO
            TODO
        use_huber : TODO
            TODO
        gamma : float
            discount factor
        learning_rate : float
            the learning rate
        ent_coef : float
            entropy coefficient for the loss calculation
        vf_coef : float
            value function coefficient for the loss calculation
        max_grad_norm : float
            the maximum value for the gradient clipping
        lam : float
            factor for trade-off of bias vs variance for Generalized Advantage
            Estimator
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
            use_huber=use_huber,
        )

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lam = lam
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
            self.terminals1 = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='terminals1')
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
            self.obs1_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ob_dim,
                name='obs1')
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

        self.reuse = reuse
        self._action = None
        self._deterministic_action = None

        # Create networks and core TF parts that are shared across setup parts.
        with tf.variable_scope("model", reuse=reuse):
            # Create the policy.
            self.action, pi_mean, pi_logstd = self._create_mlp(
                input_ph=self.obs_ph,
                num_output=self.ac_space.shape[0],
                layers=self.layers,
                act_fun=self.act_fun,
                stochastic=True,
                layer_norm=self.layer_norm,
                reuse=False,
                scope="pi",
            )
            self.deterministic_action = pi_mean
            self.action_logstd = pi_logstd

            # Create a method the log-probability of current actions.  TODO
            pi_std = tf.exp(pi_logstd)
            self.neglogp = 0.5 * tf.reduce_sum(tf.square(
                (self.action - pi_mean) / pi_std), axis=-1) \
                + 0.5 * np.log(2.0 * np.pi) * tf.cast(
                tf.shape(self.action)[-1], tf.float32) \
                + tf.reduce_sum(pi_logstd, axis=-1)

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

            # TODO: remove
            pdparam = tf.concat([pi_mean, pi_mean * 0.0 + pi_logstd], axis=1)
            self.proba_distribution = DiagGaussianProbabilityDistribution(
                pdparam)

        # =================================================================== #
        # Step 4: Setup the optimizers for the actor and critic.              #
        # =================================================================== #

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
            TODO
        act_fun : tf.nn.*
            TODO
        stochastic : bool
            whether the output should be stochastic or deterministic. If
            stochastic, a tuple of two elements is returned (the mean and
            logstd)
        layer_norm : bool
            TODO
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

    def _setup_optimizers(self, scope):
        """Setup the actor and critic optimizers."""
        scope_name = 'model/'
        if scope is not None:
            scope_name = scope + '/' + scope_name

        neglogpac = self.proba_distribution.neglogp(self.action_ph)
        self.entropy = tf.reduce_mean(self.proba_distribution.entropy())

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
            'reference_action_mean': tf.reduce_mean(self.deterministic_action),
            'reference_action_std': tf.reduce_mean(self.action_logstd),
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

    def step(self, obs, deterministic=False):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the
            environment
        :param deterministic: (bool) Whether or not to return deterministic
            actions.
        :return: ([float], [float], [float], [float]) actions, values, states,
            neglogp
        """
        return self.sess.run(
            [self.deterministic_action if deterministic else self.action,
             self.value_flat, self.neglogp],
            {self.obs_ph: obs}
        )

    def value(self, obs):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the
            environment
        :return: ([float]) The associated value of the action
        """
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    def update(self,
               obs,
               returns,
               masks,
               actions,
               values,
               neglogpacs,
               states=None):
        """Training of PPO2 Algorithm

        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in
        recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of
        Actions
        :param states: (np.ndarray) For recurrent policies, the internal state
        of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range,
                training update operation
        """
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

    def get_td_map(self, obs, returns, actions, values, neglogpacs):
        """See parent class."""
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
