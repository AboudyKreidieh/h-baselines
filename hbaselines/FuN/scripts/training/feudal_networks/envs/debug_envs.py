"""
        Note: adapted from the original debugging
        environment to have Box obs space

        Simple environment with known optimal policy and value function.

        This environment has just two actions.
        Action 0 yields 0 reward and then terminates the session.
        Action 1 yields 1 reward and then terminates the session.

        Optimal policy: action 1.

        Optimal value function: v(0)=1 (there is only one state, state 0)
"""

import numpy as np
import gym
from gym import spaces


class OneRoundDeterministicRewardBoxObsEnv(gym.Env):
    """
    Used when operating for one round and in a deterministic box observation.

    """

    def __init__(self, obs_shape=(64, 64, 1)):
        """
        Instantiate a OneRoundDeterministicRewardBoxObsEnv object.

        Parameters
        ----------
        obs_shape : 3-tuple
            dimensions of the observations
        """
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=0, shape=obs_shape)
        self._obs = np.zeros(obs_shape)

    def _step(self, action):
        """
        Private utility step function to help store rewards.

        Parameters
        ----------
        action : object
            action object
        """
        assert self.action_space.contains(action)
        reward = 1 if action == 1 else 0
        return self._obs, reward, True, {}

    def _reset(self):
        """
        Private utility function to help reset the environment.
        """
        return self._obs
