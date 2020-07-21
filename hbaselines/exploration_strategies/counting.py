"""Script containing various counting-based exploration strategies.

We implement the following exploration strategies here:

*
"""
from hbaselines.exploration_strategies.base import ExplorationStrategy


class DensityCountingExploration(ExplorationStrategy):
    """TODO.

    """

    def __init__(self, ac_space):
        """Instantiate the exploration strategy object.

        Parameters
        ----------
        ac_space : gym.space.*
            the action space of the agent
        """
        super(DensityCountingExploration, self).__init__(ac_space)

    def apply_noise(self, action):
        """See parent class."""
        return action  # TODO

    def update(self):
        """See parent class."""
        pass  # TODO


class HashCountingExploration(ExplorationStrategy):
    """TODO.

    https://arxiv.org/pdf/1611.04717.pdf

    https://arxiv.org/pdf/1707.00524.pdf
    """

    def __init__(self, ac_space):
        """Instantiate the exploration strategy object.

        Parameters
        ----------
        ac_space : gym.space.*
            the action space of the agent
        """
        super(HashCountingExploration, self).__init__(ac_space)

    def apply_noise(self, action):
        """See parent class."""
        return action  # TODO

    def update(self):
        """See parent class."""
        pass  # TODO
