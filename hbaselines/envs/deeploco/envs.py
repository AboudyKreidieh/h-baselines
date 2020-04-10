"""Script containing the DeepLoco environments."""
import gym
import numpy as np
import os
import sys
import cv2

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


class BipedalObstacles(gym.Env):
    """Bipedal Obstacles environment.

    In this environment, a bipedal agent is placed in an open field with
    obstacles scattered throughout the world. The goal of the agent is to
    walk around the world and reach a goal position.

    Attributes
    ----------
    wrapped_env : gym.Env
        the original environment, which add more dimensions than wanted here
    """

    observation_space = gym.spaces.Box(
        low=np.array(1024 * [-1.] + [
            0.06586086, -0.33134258, -0.22473772, -0.08842427, -0.24940665,
            -0.15451589, -0.49955064, -0.44008026, -0.59096092, -0.27878672,
            -0.14038287, -0.28852576, -0.20101279, -0.22234532, -0.22769515,
            -0.54376805, -0.35379013, -0.3725186, -0.33276483, -0.67418987,
            -0.35524186, -0.45274141, -0.25600547, -0.86293733, -0.60379982,
            -1.3963486, -1.35225046, -1.56099963, -1.59434652, -2.93630743,
            -3.02572751, -4.52309895, -5.14550066, -1.79466832, -1.95292163,
            -2.29718137, -2.53373265, -2.79888201, -3.67393041, -1.96048367,
            -2.22237873, -4.52637959, -5.36702728, -1.79870808, -1.6695528,
            -2.83235455, -2.84780359, -1.73784041, -2.26103067, -0.062334,
            -0.61482263, -0.61719877, -0.61664611, -0.61482263, -1.00000000,
            -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
            -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
            -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
            -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
            -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
            -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
            -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
            -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
            -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
            -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
            -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.0]),
        high=np.array(1024 * [1.] + [
            0.61076754, 0.30874607, 0.16389988, 0.278528, 0.10735691,
            0.55370122, 0.28979328, 0.88343042, 0.46615249, 0.24841864,
            0.25305298, 0.20827545, 0.35527417, 0.10670558, 0.34333566,
            0.46612564, 0.34286582, 0.24609406, 0.55321878, 0.50907576,
            0.41017145, 0.19810088, 0.49811089, 0.83155686, 0.40484139,
            1.4751488, 1.06637669, 1.60812414, 1.50176299, 3.01205444,
            3.09199214, 4.45173025, 5.29966736, 1.6375221, 1.83521891,
            2.14798474, 2.5548656, 2.72522235, 3.7703712, 2.17525077,
            1.90829098, 4.67793322, 5.20727777, 1.98003554, 1.36583793,
            2.76746488, 2.68556261, 2.02427745, 1.82794178, 1.07712889,
            1.10115075, 1.13575351, 1.12543523, 1.10115075, 1.00000000,
            1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
            1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
            1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
            1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
            1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
            1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
            1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
            1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
            1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
            1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
            1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000]),
        dtype=np.float32)

    context_space = gym.spaces.Box(
        low=np.array([-1.00000000, -1.00000000]),
        high=np.array([1.00000000, 1.00000000]),
        dtype=np.float32)

    action_space = gym.spaces.Box(
        low=np.array([
            -1.0, -1.0, -1.0, -0.5, -1.0, -1.0, -1.0, -2.57, -3.14,
            -1.0, -1.0, -1.0, -1.57, -1.0, -1.0, -1.0, -2.57, -3.14,
            -1.0, -1.0, -1.0, -1.57]),
        high=np.array([
            1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 2.57, 0.5,
            1.0, 1.0, 1.0, 1.57, 1.0, 1.0, 1.0, 2.57, 0.50,
            1.0, 1.0, 1.0, 1.57]),
        dtype=np.float32)

    def __init__(self, render):
        """Instantiate the environment.

        Parameters
        ----------
        render : bool
            whether to render the environment
        """
        # TODO: is it possible to set the horizon outside the environment
        self.horizon = 2000
        self.t = 0

        if render:
            self.wrapped_env = gym.make("PD-Biped3D-HLC-Obstacles-render-v2")
        else:
            self.wrapped_env = gym.make("PD-Biped3D-HLC-Obstacles-v2")

    @property
    def current_context(self):
        """See parent class."""
        return self.wrapped_env.env.getObservation()[-2:]

    def step(self, action):
        """See parent class."""
        self.t += 1
        obs, rew, done, info = self.wrapped_env.step(action)
        done = done or self.t >= self.horizon
        return obs[:-2], rew, done, info

    def reset(self):
        """See parent class."""
        self.t = 0
        return self.wrapped_env.reset()[:-2]

    def render(self, mode='human'):
        """See parent class."""
        image = self.wrapped_env.env.render(
            headless_step=True).astype(np.float32)

        if mode == 'human':
            f = np.flip(image / 255.0, axis=0)
            f = np.flip(f, axis=2)
            cv2.imshow("PD-Biped3D-HLC-Obstacles-v2", f)
            cv2.waitKey(1)

        elif mode == 'rgb_array':
            return image
