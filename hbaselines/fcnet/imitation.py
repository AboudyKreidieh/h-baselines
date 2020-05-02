"""Script containing the fcnet variant of the imitation learning policy."""


class ImitationLearningPolicy(object):
    """Fully-connected neural network imitation learning policy.

    Attributes
    ----------
    TODO
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
                 layers,
                 act_fun,
                 use_huber):
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
        layers : list of int or None
            the size of the Neural network for the policy
        act_fun : tf.nn.*
            the activation function to use in the neural network
        use_huber : bool
            specifies whether to use the huber distance function as the loss
            function. If set to False, the mean-squared error metric is used
            instead
        """
        pass  # TODO

    def initialize(self):
        """Initialize the policy.

        This is used at the beginning of training by the algorithm, after the
        model parameters have been initialized.
        """
        pass  # TODO: remove?

    def update(self):
        """Perform a gradient update step.

        Returns
        -------
        float
            policy loss
        """
        pass  # TODO

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
        pass  # TODO

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
        pass  # TODO

    def get_td_map(self):
        """Return dict map for the summary (to be run in the algorithm)."""
        pass  # TODO
