"""Script containing the abstract imitation learning policy class."""
import numpy as np


class ImitationLearningPolicy(object):
    """Base imitation learning policy.

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
    learning_rate : float
        the learning rate for the policy
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    layer_norm : bool
        enable layer normalisation
    layers : list of int or None
        the size of the Neural network for the policy
    act_fun : tf.nn.*
        the activation function to use in the neural network
    use_huber : bool
        specifies whether to use the huber distance function as the loss
        function. If set to False, the mean-squared error metric is used
        instead. Only applies to deterministic policies.
    stochastic : bool
        specifies whether the policies are stochastic or deterministic
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
                 stochastic):
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
            instead. Only applies to deterministic policies.
        stochastic : bool
            specifies whether the policies are stochastic or deterministic
        """
        self.sess = sess
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.co_space = co_space
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.layer_norm = layer_norm
        self.layers = layers
        self.act_fun = act_fun
        self.use_huber = use_huber
        self.stochastic = stochastic

    def update(self):
        """Perform a gradient update step.

        Returns
        -------
        float
            policy loss
        """
        raise NotImplementedError

    def get_action(self, obs, context):
        """Compute the policy actions.

        Parameters
        ----------
        obs : array_like
            the observation
        context : array_like or None
            the contextual term. Set to None if no context is provided by the
            environment.

        Returns
        -------
        array_like
            computed action by the policy
        """
        raise NotImplementedError

    def store_transition(self, obs0, context0, action, obs1, context1):
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
        obs1 : array_like
            the current observation
        context1 : array_like or None
            the current contextual term. Set to None if no context is provided
            by the environment.
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
