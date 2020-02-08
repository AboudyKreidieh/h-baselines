"""Script containing the DeepLoco environments."""
import gym
import numpy as np
import os
import sys

try:
    sys.path.append(os.path.join(os.environ["TERRAINRL_PATH"], "simAdapter"))
    import terrainRLSim  # noqa: F401
except (KeyError, ImportError, ModuleNotFoundError):
    pass


class BipedalSoccer(gym.Env):
    """Bipedal Soccer environment.

    In this environment, a bipedal agent is placed in an open field with a
    soccer ball. The agent is rewarded for moving to the ball, and additionally
    dribbling the ball to the target. The reward function is a weighted sum of
    the agent's distance from the ball and the distance of the ball from a
    desired goal position. This reward is positive to discourage the agent from
    falling prematurely.

    Attributes
    ----------
    wrapped_env : gym.Env
        the original environment, which add more dimensions than wanted here
    """

    def __init__(self, render):
        """Instantiate the environment.

        Parameters
        ----------
        render : bool
            whether to render the environment
        """
        if render:
            self.wrapped_env = gym.make("PD-Biped3D-HLC-Soccer-Render-v1")
        else:
            self.wrapped_env = gym.make("PD-Biped3D-HLC-Soccer-v1")

    @property
    def observation_space(self):
        """See parent class."""
        return self.wrapped_env.observation_space

    @property
    def action_space(self):
        """See parent class."""
        return self.wrapped_env.action_space

    def step(self, action):
        """See parent class."""
        obs, rew, done, info = self.wrapped_env.step(np.array([action]))
        return obs[0], rew[0][0], done, info

    def reset(self):
        """See parent class."""
        return self.wrapped_env.reset()[0]

    def render(self, mode='human'):
        """See parent class."""
        pass
