"""Script containing the environment sampler method."""
import ray
from gym.spaces import Box

from hbaselines.algorithms.utils import get_obs
from hbaselines.utils.env_util import create_env


class Sampler(object):
    """Environment sampler object.

    Attributes
    ----------
    env : gym.Env
        the training / evaluation environment
    """

    def __init__(self, env_name, render, shared, maddpg, evaluate, env_num):
        """Instantiate the sampler object.

        Parameters
        ----------
        env_name : str
            the name of the environment
        render : bool
            whether to render the environment
        shared : bool
            specifies whether agents in an environment are meant to share
            policies. This is solely used by multi-agent Flow environments.
        maddpg : bool
            whether to use an environment variant that is compatible with the
            MADDPG algorithm
        evaluate : bool
            specifies whether this is a training or evaluation environment
        env_num : int
            the environment number. Used to handle situations when multiple
            parallel environments are being used.
        """
        self.env, self._init_obs = create_env(
            env=env_name,
            render=render,
            shared=shared,
            maddpg=maddpg,
            evaluate=evaluate,
        )

        # Collect the key for the info_dict variable.
        if isinstance(self.env.action_space, dict):
            initial_action = {key: self.env.action_space[key].sample()
                              for key in self.env.action_space.keys()}
        elif env_name.startswith("multiagent") and shared:
            initial_action = {key: self.env.action_space.sample()
                              for key in self._init_obs.keys()}
        else:
            initial_action = self.env.action_space.sample()
        _, _, _, info_dict = self.env.step(initial_action)

        self._env_num = env_num
        self._render = render
        self._info_keys = list(info_dict.keys())

    def get_init_obs(self):
        """Return the initial observation from the environment."""
        return self._init_obs.copy(), self._info_keys

    def get_context(self):
        """Collect the contextual term. None if it is not passed."""
        return [self.env.current_context] if hasattr(
            self.env, "current_context") else None

    def observation_space(self):
        """Return the environment's observation space."""
        return self.env.observation_space

    def action_space(self):
        """Return the environment's action space."""
        return self.env.action_space

    def context_space(self):
        """Return the environment's context space."""
        return getattr(self.env, "context_space", None)

    def all_observation_space(self):
        """Return the environment's full observation space."""
        return getattr(self.env, "all_observation_space", Box(-1, 1, (1,)))

    def horizon(self):
        """Return the environment's time horizon."""
        if hasattr(self.env, "horizon"):
            return self.env.horizon
        elif hasattr(self.env, "_max_episode_steps"):
            return self.env._max_episode_steps
        elif hasattr(self.env, "env_params"):
            # for Flow environments
            return self.env.env_params.horizon
        else:
            raise ValueError("Horizon attribute not found.")

    def collect_sample(self, action):
        """Perform the sample collection operation over a single step.

        This method is responsible for executing a single step of the
        environment. This is perform a number of times in the _collect_samples
        method before training is executed. The data from the rollouts is
        stored in the policy's replay buffer(s).

        Parameters
        ----------
        action : array_like
            the action to be performed by the agent(s) within the environment

        Returns
        -------
        dict
            information from the most recent environment update step,
            consisting of the following terms:

            * obs : the most recent observation. This consists of a single
              observation if no reset occured, and a tuple of (last observation
              from the previous rollout, first observation of the next rollout)
              if a reset occured.
            * context : the contextual term from the environment
            * action : the action performed by the agent(s)
            * reward : the reward from the most recent step
            * done : the done mask
            * env_num : the environment number
            * all_obs : the most recent full-state observation. This consists
              of a single observation if no reset occured, and a tuple of (last
              observation from the previous rollout, first observation of the
              next rollout) if a reset occured.
        """
        # Execute the next action.
        obs, reward, done, info = self.env.step(action)
        obs, all_obs = get_obs(obs)

        # Visualize the current step.
        if self._render and self._env_num == 0:
            self.env.render()  # pragma: no cover

        # Get the contextual term.
        context = getattr(self.env, "current_context", None)

        # Done mask for multi-agent policies is slightly different.
        if isinstance(done, dict):
            done = done["__all__"]

        if done:
            # Reset the environment.
            reset_obs = self.env.reset()
            reset_obs, reset_all_obs = get_obs(reset_obs)
        else:
            reset_obs = None
            reset_all_obs = None

        return {
            "obs": obs if not done else (obs, reset_obs),
            "context": context,
            "action": action,
            "reward": reward,
            "done": done,
            "env_num": self._env_num,
            "all_obs": all_obs if not done else (all_obs, reset_all_obs),
            "info": info,
        }


@ray.remote
class RaySampler(Sampler):
    """Ray-compatible variant of the environment sampler object.

    Used to collect samples in parallel.
    """

    pass
