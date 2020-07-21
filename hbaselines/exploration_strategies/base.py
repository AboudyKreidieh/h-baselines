"""Script containing the base exploration strategy object."""


class ExplorationStrategy(object):
    """Base exploration strategy object.

    Attributes
    ----------
    ac_space : gym.spaces.*
        the action space of the agent
    """

    def __init__(self, ac_space):
        """Instantiate the base exploration strategy object.

        Parameters
        ----------
        ac_space : gym.spaces.*
            the action space of the agent
        """
        self.ac_space = ac_space

    def apply_noise(self, action):
        """Apply exploration noise to the original action.

        Parameters
        ----------
        action : array_like
            the original action

        Returns
        -------
        array_like
            the updated action
        """
        raise NotImplementedError

    def update(self):
        """

        :return:
        """
        raise NotImplementedError
