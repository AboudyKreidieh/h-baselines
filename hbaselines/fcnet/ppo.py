import warnings
from itertools import zip_longest

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete

from stable_baselines.common.tf_layers import linear
from stable_baselines.common.distributions import make_proba_dist_type
from stable_baselines.common.distributions import \
    CategoricalProbabilityDistribution
from stable_baselines.common.distributions import \
    MultiCategoricalProbabilityDistribution
from stable_baselines.common.distributions import \
    DiagGaussianProbabilityDistribution
from stable_baselines.common.distributions import \
    BernoulliProbabilityDistribution
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


class FeedForwardPolicy(object):
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
                 n_env,
                 n_steps,
                 n_batch,
                 layers=None,
                 net_arch=None,
                 act_fun=tf.tanh,
                 feature_extraction="mlp",
                 reuse=False,
                 scale=False,
                 obs_phs=None,
                 add_action_ph=False,
                 **kwargs):
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
        self.sess = sess
        self.reuse = reuse
        self.ob_space = ob_space
        self.ac_space = ac_space
        self._pdtype = make_proba_dist_type(ac_space)
        self._policy = None
        self._proba_distribution = None
        self._value_fn = None
        self._action = None
        self._deterministic_action = None

        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn(
                "Usage of the `layers` parameter is deprecated! Use net_arch "
                "instead (it has a different semantics though).",
                DeprecationWarning
            )
            if net_arch is not None:
                warnings.warn(
                    "The new `net_arch` parameter overrides the deprecated "
                    "`layers` parameter!",
                    DeprecationWarning
                )

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
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
    def initial_state(self):
        """
        The initial state of the policy. For feedforward policies, None. For a
        recurrent policy, a NumPy array of shape (self.n_env, ) + state_shape.
        """
        assert not self.recurrent, \
            "When using recurrent policies, you must overwrite " \
            "`initial_state()` method"
        return None

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

    @staticmethod
    def _kwargs_check(feature_extraction, kwargs):
        """
        Ensure that the user is not passing wrong keywords
        when using policy_kwargs.

        :param feature_extraction: (str)
        :param kwargs: (dict)
        """
        if feature_extraction == 'mlp' and len(kwargs) > 0:
            raise ValueError("Unknown keywords for policy: {}".format(kwargs))

    def _setup_init(self):
        """Sets up the distributions, actions, and value."""
        with tf.variable_scope("output", reuse=True):
            assert self.policy is not None \
                and self.proba_distribution is not None \
                and self.value_fn is not None
            self._action = self.proba_distribution.sample()
            self._deterministic_action = self.proba_distribution.mode()
            self._neglogp = self.proba_distribution.neglogp(self.action)
            if isinstance(self.proba_distribution,
                          CategoricalProbabilityDistribution):
                self._policy_proba = tf.nn.softmax(self.policy)
            elif isinstance(self.proba_distribution,
                            DiagGaussianProbabilityDistribution):
                self._policy_proba = [
                    self.proba_distribution.mean,
                    self.proba_distribution.std]
            elif isinstance(self.proba_distribution,
                            BernoulliProbabilityDistribution):
                self._policy_proba = tf.nn.sigmoid(self.policy)
            elif isinstance(self.proba_distribution,
                            MultiCategoricalProbabilityDistribution):
                self._policy_proba = [
                    tf.nn.softmax(categorical.flatparam())
                    for categorical in self.proba_distribution.categoricals]
            else:
                # it will return nothing, as it is not implemented
                self._policy_proba = []
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
        return action, value, self.initial_state, neglogp

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
