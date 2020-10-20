"""Script containing various counting-based exploration strategies.

We implement the following exploration strategies here:

*
"""
from hbaselines.exploration_strategies.base import ExplorationStrategy
import numpy as np
from hbaselines.exploration_strategies.density_models import GaussianMixtureModel


class DensityCountingExploration(ExplorationStrategy):
    """TODO.

    """

    def __init__(self, ac_space, obs_dim, density_model=GaussianMixtureModel):
        """Instantiate the exploration strategy object.

        Parameters
        ----------
        ac_space : gym.space.*
            the action space of the agent
        density_model :
            The density model used to assign state distribution
        """
        super(DensityCountingExploration, self).__init__(ac_space)
        self.density_model = density_model(dimension=obs_dim)

    def apply_noise(self, action):
        """See parent class."""
        return action  # TODO

    def update(self, obs1):
        """See parent class."""
        self.density_model.update(obs1)

    def update_loss(self, loss, obs):
        """Update the loss function in the policy.
        Parameters
        ----------
        loss : tf.Variable
            the original loss
        obs : tf.compat.v1.placeholder
            the observation placeholder
        action : tf.compat.v1.placeholder
            the action placeholder
        reward : tf.compat.v1.placeholder
            the reward placeholder
        terminal : tf.comapt.v1.placeholder
            the done mask placeholder
        Returns
        -------
        tf.Variable
            the new loss
        """
        prob = self.density_model.probability(obs)
        recoding_prob = self.density_model.recoding_probability(obs)
        prediction_gain = np.log(recoding_prob) - np.log(prob)
        pseudo_count = 1 / (np.exp(prediction_gain) - 1)
        bonus_reward = np.power(pseudo_count, -0.5)
        return loss - bonus_reward


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

    def update(self, obs0, actions, rewards, obs1, terminals1):
        """See parent class."""
        pass  # TODO
