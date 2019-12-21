"""Script containing environment generator for modified Flow benchmarks."""
import gym
from flow.utils.registry import make_create_env

from hbaselines.envs.mixed_autonomy.merge import get_flow_params as merge
from hbaselines.envs.mixed_autonomy.ring import get_flow_params as ring
from hbaselines.envs.mixed_autonomy.figure_eight import get_flow_params \
    as figure_eight


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

    def __init__(self, env_name, env_params=None, render=False, version=0):
        """Create the environment.

        Parameters
        ----------
        env_name : str
            the name of the environment to create
        env_params : dict
            environment-specific parameters
        render : bool
            whether to render the environment
        version : int
            environment version number, needed for testing purposes

        Returns
        -------
        gym.Env
            the environment

        Raises
        ------
        AssertionError
            if the `env_name` parameter is not valid
        """
        assert env_name in ["ring", "merge", "figure_eight"]

        # default to empty dictionary if not passed
        env_params = env_params or {}

        # get flow-specific parameters
        flow_params = dict()
        if env_name == "merge":
            flow_params = merge(**env_params)
        elif env_name == "ring":
            flow_params = ring(**env_params)
        elif env_name == "figure_eight":
            flow_params = figure_eight(**env_params)

        # create the wrapped environment
        create_env, _ = make_create_env(flow_params, version, render)
        self.wrapped_env = create_env()

        # for tracking the time horizon
        self.step_number = 0
        self.horizon = self.wrapped_env.env_params.horizon

    @property
    def action_space(self):
        """See wrapped environment."""
        return self.wrapped_env.action_space

    @property
    def observation_space(self):
        """See wrapped environment."""
        return self.wrapped_env.observation_space

    def step(self, action):
        """See wrapped environment.

        The done term is also modified in case the time horizon is met.
        """
        obs, reward, done, info_dict = self.wrapped_env.step(action)

        # Check if the time horizon has been met.
        self.step_number += 1
        done = done or self.step_number == self.horizon

        return obs, reward, done, info_dict

    def reset(self):
        """Reset the environment."""
        self.step_number = 0
        return self.wrapped_env.reset()
