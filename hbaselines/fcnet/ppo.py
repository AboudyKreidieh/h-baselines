from itertools import zip_longest

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete

from hbaselines.base_policies.base import Policy

from stable_baselines.common.tf_layers import linear
from stable_baselines.common.input import observation_input


def mlp_extractor(flat_observations, net_arch, act_fun):
    """
    Constructs an MLP that receives observations as an input and outputs a
    latent representation for the policy and a value network. The ``net_arch``
    parameter allows to specify the amount and size of the hidden layers and
    how many of them are shared between the policy network and the value
    network. It is assumed to be a list with the following structure:

    1. An arbitrary length (zero allowed) number of integers each specifying
       the number of units in a shared layer. If the number of ints is zero,
       there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the
       value network and the policy network. It is formatted like
       ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``. If it is
       missing any of the keys (pi or vf), no non-shared layers (empty list) is
       assumed.

    For example to construct a network with one shared layer of size 55
    followed by two non-shared layers for the value network of size 255 and a
    single non-shared layer of size 128 for the policy network, the following
    layers_spec would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A
    simple shared network topology with two layers of size 128 would be
    specified as [128, 128].

    :param flat_observations: (tf.Tensor) The observations to base policy and
        value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value
        networks. See above for details on its formatting.
    :param act_fun: (tf function) The activation function to use for the
        networks.
    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the
        specified network. If all layers are shared, then ``latent_policy ==
        latent_value``
    """
    latent = flat_observations
    # Layer sizes of the network that only belongs to the policy network
    policy_only_layers = []
    # Layer sizes of the network that only belongs to the value network
    value_only_layers = []

    # Iterate through the shared layers and build the shared parts of the
    # network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(linear(latent, "shared_fc{}".format(idx),
                                    layer_size, init_scale=np.sqrt(2)))
        else:
            assert isinstance(layer, dict), \
                "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), \
                    "Error: net_arch[-1]['pi'] must contain a list of " \
                    "integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), \
                    "Error: net_arch[-1]['vf'] must contain a list of " \
                    "integers."
                value_only_layers = layer['vf']
            # From here on the network splits up in policy and value network
            break

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    for idx, (pi_layer_size, vf_layer_size) in enumerate(
            zip_longest(policy_only_layers, value_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), \
                "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = act_fun(linear(
                latent_policy, "pi_fc{}".format(idx), pi_layer_size,
                init_scale=np.sqrt(2)))

        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), \
                "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = act_fun(linear(
                latent_value, "vf_fc{}".format(idx), vf_layer_size,
                init_scale=np.sqrt(2)))

    return latent_policy, latent_value


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
        super(DiagGaussianProbabilityDistribution, self).__init__()

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


class DiagGaussianProbabilityDistributionType(object):
    def __init__(self, size):
        """
        The probability distribution type for multivariate Gaussian input

        :param size: (int) the number of dimensions of the multivariate
            gaussian
        """
        self.size = size

    @staticmethod
    def proba_distribution_from_flat(flat):
        """
        returns the probability distribution from flat probabilities

        :param flat: ([float]) the flat probabilities
        :return: (ProbabilityDistribution) the instance of the
            ProbabilityDistribution associated
        """
        return DiagGaussianProbabilityDistribution(flat)

    def proba_distribution_from_latent(self,
                                       pi_latent_vector,
                                       vf_latent_vector,
                                       init_scale=1.0,
                                       init_bias=0.0):
        mean = linear(pi_latent_vector, 'pi', self.size,
                      init_scale=init_scale, init_bias=init_bias)
        logstd = tf.get_variable(name='pi/logstd', shape=[1, self.size],
                                 initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        q_values = linear(vf_latent_vector, 'q', self.size,
                          init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), mean, q_values

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    @staticmethod
    def sample_dtype():
        return tf.float32

    def param_placeholder(self, prepend_shape, name=None):
        """
        returns the TensorFlow placeholder for the input parameters

        :param prepend_shape: ([int]) the prepend shape
        :param name: (str) the placeholder name
        :return: (TensorFlow Tensor) the placeholder
        """
        return tf.placeholder(
            dtype=tf.float32,
            shape=prepend_shape + self.param_shape(),
            name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        """
        returns the TensorFlow placeholder for the sampling

        :param prepend_shape: ([int]) the prepend shape
        :param name: (str) the placeholder name
        :return: (TensorFlow Tensor) the placeholder
        """
        return tf.placeholder(
            dtype=self.sample_dtype(),
            shape=prepend_shape + self.sample_shape(),
            name=name)


class FeedForwardPolicy(Policy):
    """
    Policy object that implements actor critic

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
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
                 n_env,
                 n_steps,
                 n_batch,
                 reuse=False,
                 scale=False,
                 obs_phs=None,
                 add_action_ph=False):
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

        self.n_env = n_env
        self.n_steps = n_steps
        self.n_batch = n_batch
        with tf.variable_scope("input", reuse=False):
            if obs_phs is None:
                self._obs_ph, self._processed_obs = observation_input(
                    ob_space, n_batch, scale=scale)
            else:
                self._obs_ph, self._processed_obs = obs_phs

            self._action_ph = None
            if add_action_ph:
                self._action_ph = tf.placeholder(
                    dtype=ac_space.dtype,
                    shape=(n_batch,) + ac_space.shape,
                    name="action_ph")
        self.reuse = reuse
        self._pdtype = DiagGaussianProbabilityDistributionType(
            ac_space.shape[0])
        self._action = None
        self._deterministic_action = None

        net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            pi_latent, vf_latent = mlp_extractor(
                tf.layers.flatten(self.processed_obs), net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(
                    pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    @property
    def is_discrete(self):
        """bool: is action space discrete."""
        return isinstance(self.ac_space, Discrete)

    @property
    def obs_ph(self):
        """tf.Tensor: placeholder for observations, shape (self.n_batch, ) +
        self.ob_space.shape."""
        return self._obs_ph

    @property
    def processed_obs(self):
        """tf.Tensor: processed observations, shape (self.n_batch, ) +
        self.ob_space.shape.

        The form of processing depends on the type of the observation space,
        and the parameters whether scale is passed to the constructor; see
        observation_input for more information."""
        return self._processed_obs

    @property
    def action_ph(self):
        """tf.Tensor: placeholder for actions, shape (self.n_batch, ) +
        self.ac_space.shape."""
        return self._action_ph

    def _setup_init(self):
        """Sets up the distributions, actions, and value."""
        with tf.variable_scope("output", reuse=True):
            assert self.policy is not None \
                and self.proba_distribution is not None \
                and self.value_fn is not None
            self._action = self.proba_distribution.sample()
            self._deterministic_action = self.proba_distribution.mode()
            self._neglogp = self.proba_distribution.neglogp(self.action)
            self._policy_proba = [
                self.proba_distribution.mean,
                self.proba_distribution.std]
            self._value_flat = self.value_fn[:, 0]

    @property
    def pdtype(self):
        """ProbabilityDistributionType: type of the distribution for stochastic
        actions."""
        return self._pdtype

    @property
    def policy(self):
        """tf.Tensor: policy output, e.g. logits."""
        return self._policy

    @property
    def proba_distribution(self):
        """ProbabilityDistribution: distribution of stochastic actions."""
        return self._proba_distribution

    @property
    def value_fn(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, 1)"""
        return self._value_fn

    @property
    def value_flat(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, )"""
        return self._value_flat

    @property
    def action(self):
        """tf.Tensor: stochastic action, of shape (self.n_batch, ) +
        self.ac_space.shape."""
        return self._action

    @property
    def deterministic_action(self):
        """tf.Tensor: deterministic action, of shape (self.n_batch, ) +
        self.ac_space.shape."""
        return self._deterministic_action

    @property
    def neglogp(self):
        """tf.Tensor: negative log likelihood of the action sampled by
        self.action."""
        return self._neglogp

    @property
    def policy_proba(self):
        """tf.Tensor: parameters of the probability distribution.
        Depends on pdtype."""
        return self._policy_proba

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
