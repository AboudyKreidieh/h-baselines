import numpy as np
import tensorflow as tf
from gym.spaces import Discrete

from hbaselines.base_policies.base import Policy
from hbaselines.utils.tf_util import layer


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
                 gamma,
                 learning_rate,
                 ent_coef,
                 vf_coef,
                 max_grad_norm,
                 lam,
                 nminibatches,
                 noptepochs,
                 cliprange,
                 cliprange_vf,
                 layer_norm,
                 layers,
                 act_fun,
                 use_huber,
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
        gamma : TODO
            TODO
        learning_rate : TODO
            TODO
        ent_coef : TODO
            TODO
        vf_coef : TODO
            TODO
        max_grad_norm : TODO
            TODO
        lam : TODO
            TODO
        nminibatches : TODO
            TODO
        noptepochs : TODO
            TODO
        cliprange : TODO
            TODO
        cliprange_vf : TODO
            TODO
        layer_norm : TODO
            TODO
        layers : TODO
            TODO
        act_fun : TODO
            TODO
        use_huber : TODO
            TODO
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
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
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

            pdparam = tf.concat([pi_mean, pi_mean * 0.0 + pi_logstd], axis=1)
            self.proba_distribution = DiagGaussianProbabilityDistribution(
                pdparam)
            self.policy = pi_mean

        with tf.variable_scope("output", reuse=True):
            self.deterministic_action = self.proba_distribution.mode()
            self.neglogp = self.proba_distribution.neglogp(self.action)
            self.policy_proba = [
                self.proba_distribution.mean,
                self.proba_distribution.std]
            self.value_flat = self.value_fn[:, 0]

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

    @property
    def is_discrete(self):
        """bool: is action space discrete."""
        return isinstance(self.ac_space, Discrete)

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
        if deterministic:
            action, value, neglogp = self.sess.run(
                [self.deterministic_action, self.value_flat, self.neglogp],
                {self.obs_ph: obs}
            )
        else:
            action, value, neglogp = self.sess.run(
                [self.action, self.value_flat, self.neglogp],
                {self.obs_ph: obs}
            )
        return action, value, neglogp

    def proba_step(self, obs):
        """
        Returns the action probability for a single step

        :param obs: ([float] or [int]) The current observation of the
            environment
        :return: ([float]) the action probability
        """
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the
            environment
        :return: ([float]) The associated value of the action
        """
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
