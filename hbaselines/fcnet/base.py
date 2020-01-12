"""Script containing the abstract policy class."""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.utils.tf_util import get_target_updates


class ActorCriticPolicy(object):
    """Base Actor Critic Policy.

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
        specifies whether to use the huber distance function as the loss for
        the critic. If set to False, the mean-squared error metric is used
        instead
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
                 use_huber):
        """Instantiate the base policy object.

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
        """
        self.sess = sess
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.co_space = co_space
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.verbose = verbose
        self.layers = layers
        self.tau = tau
        self.gamma = gamma
        self.layer_norm = layer_norm
        self.act_fun = act_fun
        self.use_huber = use_huber

    def initialize(self):
        """Initialize the policy.

        This is used at the beginning of training by the algorithm, after the
        model parameters have been initialized.
        """
        raise NotImplementedError

    def update(self, update_actor=True, **kwargs):
        """Perform a gradient update step.

        Parameters
        ----------
        update_actor : bool
            specifies whether to update the actor policy. The critic policy is
            still updated if this value is set to False.

        Returns
        -------
        float
            critic loss
        float
            actor loss
        """
        raise NotImplementedError

    def get_action(self, obs, context, apply_noise, random_actions):
        """Call the actor methods to compute policy actions.

        Parameters
        ----------
        obs : array_like
            the observation
        context : array_like or None
            the contextual term. Set to None if no context is provided by the
            environment.
        apply_noise : bool
            whether to add Gaussian noise to the output of the actor. Defaults
            to False
        random_actions : bool
            if set to True, actions are sampled randomly from the action space
            instead of being computed by the policy. This is used for
            exploration purposes.

        Returns
        -------
        array_like
            computed action by the policy
        """
        raise NotImplementedError

    def value(self, obs, context, action):
        """Call the critic methods to compute the value.

        Parameters
        ----------
        obs : array_like
            the observation
        context : array_like or None
            the contextual term. Set to None if no context is provided by the
            environment.
        action : array_like
            the actions performed in the given observation

        Returns
        -------
        array_like
            computed value by the critic
        """
        raise NotImplementedError

    def store_transition(self, obs0, context0, action, reward, obs1, context1,
                         done, is_final_step, evaluate=False):
        """Store a transition in the replay buffer.

        Parameters
        ----------
        obs0 : array_like
            the last observation
        context0 : array_like or None
            the last contextual term. Set to None if no context is provided by
            the environment.
        action : array_like
            the action
        reward : float
            the reward
        obs1 : array_like
            the current observation
        context1 : array_like or None
            the current contextual term. Set to None if no context is provided
            by the environment.
        done : float
            is the episode done
        is_final_step : bool
            whether the time horizon was met in the step corresponding to the
            current sample. This is used by the TD3 algorithm to augment the
            done mask.
        evaluate : bool
            whether the sample is being provided by the evaluation environment.
            If so, the data is not stored in the replay buffer.
        """
        raise NotImplementedError

    def get_td_map(self):
        """Return dict map for the summary (to be run in the algorithm)."""
        raise NotImplementedError

    @staticmethod
    def _get_obs(obs, context, axis=0):
        """Return the processed observation.

        If the contextual term is not None, this will look as follows:

                                    -----------------
                    processed_obs = | obs | context |
                                    -----------------

        Otherwise, this method simply returns the observation.

        Parameters
        ----------
        obs : array_like
            the original observation
        context : array_like or None
            the contextual term. Set to None if no context is provided by the
            environment.
        axis : int
            the axis to concatenate the observations and contextual terms by

        Returns
        -------
        array_like
            the processed observation
        """
        if context is not None:
            context = context.flatten() if axis == 0 else context
            obs = np.concatenate((obs, context), axis=axis)
        return obs

    @staticmethod
    def _get_ob_dim(ob_space, co_space):
        """Return the processed observation dimension.

        If the context space is not None, it is included in the computation of
        this term.

        Parameters
        ----------
        ob_space : gym.spaces.*
            the observation space of the environment
        co_space : gym.spaces.*
            the context space of the environment

        Returns
        -------
        tuple
            the true observation dimension
        """
        ob_dim = ob_space.shape
        if co_space is not None:
            ob_dim = tuple(map(sum, zip(ob_dim, co_space.shape)))
        return ob_dim

    @staticmethod
    def _layer(val,
               num_outputs,
               name,
               act_fun=None,
               kernel_initializer=slim.variance_scaling_initializer(
                   factor=1.0 / 3.0, mode='FAN_IN', uniform=True),
               layer_norm=False):
        """Create a fully-connected layer.

        Parameters
        ----------
        val : tf.Variable
            the input to the layer
        num_outputs : int
            number of outputs from the layer
        name : str
            the scope of the layer
        act_fun : tf.nn.* or None
            the activation function
        kernel_initializer : Any
            the initializing operation to the weights of the layer
        layer_norm : bool
            whether to enable layer normalization

        Returns
        -------
        tf.Variable
            the output from the layer
        """
        val = tf.layers.dense(
            val, num_outputs, name=name, kernel_initializer=kernel_initializer)

        if layer_norm:
            val = tf.contrib.layers.layer_norm(val, center=True, scale=True)

        if act_fun is not None:
            val = act_fun(val)

        return val

    @staticmethod
    def _setup_target_updates(model_scope, target_scope, scope, tau, verbose):
        """Create the soft and initial target updates.

        The initial model parameters are assumed to be stored under the scope
        name "model", while the target policy parameters are assumed to be
        under the scope name "target".

        If an additional outer scope was provided when creating the policies,
        they can be passed under the `scope` parameter.

        Parameters
        ----------
        model_scope : str
            the scope of the model parameters
        target_scope : str
            the scope of the target parameters
        scope : str or None
            the outer scope, set to None if not available
        tau : float
            target update rate
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug

        Returns
        -------
        tf.Operation
            initial target updates, to match the target with the model
        tf.Operation
            soft target update operations
        """
        if scope is not None:
            model_scope = scope + '/' + model_scope
            target_scope = scope + '/' + target_scope

        return get_target_updates(
            get_trainable_vars(model_scope),
            get_trainable_vars(target_scope),
            tau, verbose)

    @staticmethod
    def _remove_fingerprint(val, ob_dim, fingerprint_dim, additional_dim):
        """Remove the fingerprint from the input.

        This is a hacky procedure to remove the fingerprint elements from the
        computation. The fingerprint elements are the last few elements of the
        observation dimension, before any additional concatenated observations
        (e.g. contexts or actions).

        Parameters
        ----------
        val : tf.Variable
            the original input
        ob_dim : int
            number of environmental observation elements
        fingerprint_dim : int
            number of fingerprint elements
        additional_dim : int
            number of additional elements that were added to the input variable

        Returns
        -------
        tf.Variable
            the input with the fingerprints zeroed out
        """
        return val * tf.constant([1.0] * (ob_dim - fingerprint_dim) +
                                 [0.0] * fingerprint_dim +
                                 [1.0] * additional_dim)
