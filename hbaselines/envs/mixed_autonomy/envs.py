"""Script containing environment generator for modified Flow benchmarks."""
import gym
from gym.spaces import Box
from copy import deepcopy
import numpy as np
from flow.utils.registry import make_create_env


class FlowEnv(gym.Env):
    """Create a flow-specific environment, as provided by this repository.

    The original environments are wrapped by an additional environment that
    accounts for return done values of True when the time horizon is met.

    Attributes
    ----------
    wrapped_env : flow.envs.Env
        the original Flow environment
    step_number : int
        number of steps taken in the current rollout
    horizon : int
        the environment's time horizon
    """

    def __init__(self,
                 flow_params,
                 multiagent=False,
                 shared=False,
                 maddpg=False,
                 render=False,
                 version=0):
        """Create the environment.

        Parameters
        ----------
        flow_params : dict
            environment-specific parameters
        multiagent : bool
            whether the environment is a multi-agent environment
        shared : bool
            whether the policies in the environment are shared or independent.
            This is only relevant if `shared` is set to True.
        render : bool
            whether to render the environment
        version : int
            environment version number, needed for testing purposes
        """
        # Initialize some variables.
        self.multiagent = multiagent
        self.shared = shared
        self.maddpg = maddpg

        if "full_observation_fn" in flow_params["env"].additional_params:
            self.full_observation_fn = deepcopy(
                flow_params["env"].additional_params["full_observation_fn"])
            del flow_params["env"].additional_params["full_observation_fn"]
        else:
            self.full_observation_fn = None

        # Create the wrapped environment.
        create_env, _ = make_create_env(flow_params, version, render)
        self.wrapped_env = create_env()

        # Collect the IDs of individual vehicles if using a multi-agent env.
        if self.multiagent:
            self.agents = list(self.wrapped_env.reset().keys())

        # for tracking the time horizon
        self.step_number = 0
        self.horizon = self.wrapped_env.env_params.horizon

    @property
    def action_space(self):
        """See wrapped environment."""
        if self.multiagent and not self.shared:
            return {key: self.wrapped_env.action_space for key in self.agents}
        else:
            return self.wrapped_env.action_space

    @property
    def observation_space(self):
        """See wrapped environment."""
        if self.multiagent and not self.shared:
            return {
                key: self.wrapped_env.observation_space for key in self.agents}
        else:
            return self.wrapped_env.observation_space

    def step(self, action):
        """See wrapped environment.

        The done term is also modified in case the time horizon is met.
        """
        obs, reward, done, info_dict = self.wrapped_env.step(action)

        # Check if the time horizon has been met.
        self.step_number += 1
        done = done or self.step_number == self.horizon

        # In case of a multi-agent shared environment, all policies should have
        # the same reward.
        if self.multiagent and self.shared:
            reward = reward[self.agents[0]]

        # Add the full-state observation, if needed.
        if self.maddpg:
            obs = {
                "obs": obs,
                "all_obs": np.asarray(
                    [self.full_observation_fn(self.wrapped_env)])
            }

        return obs, reward, done, info_dict

    def reset(self):
        """Reset the environment."""
        self.step_number = 0

        obs = self.wrapped_env.reset()

        # Add the full-state observation, if needed.
        if self.maddpg:
            obs = {
                "obs": obs,
                "all_obs": np.asarray(
                    [self.full_observation_fn(self.wrapped_env)])
            }

        return obs

    @property
    def all_observation_space(self):
        """Return the shape of the full observation space."""
        if self.full_observation_fn is None:
            return None
        else:
            return Box(
                low=-float("inf"),
                high=float("inf"),
                shape=self.full_observation_fn(self.wrapped_env).shape,
                dtype=np.float32,
            )
