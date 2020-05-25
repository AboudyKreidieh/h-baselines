"""Utility method for the algorithm classes."""
import numpy as np

from hbaselines.fcnet.td3 import FeedForwardPolicy as TD3FeedForward
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy as \
    TD3GoalConditioned
from hbaselines.goal_conditioned.sac import GoalConditionedPolicy as \
    SACGoalConditioned
from hbaselines.fcnet.sac import FeedForwardPolicy as SACFeedForward
from hbaselines.multi_fcnet.td3 import MultiFeedForwardPolicy as \
    TD3MultiFeedForwardPolicy
from hbaselines.multi_fcnet.sac import MultiFeedForwardPolicy as \
    SACMultiFeedForwardPolicy


def is_td3_policy(policy):
    """Check whether a policy is for designed to support TD3."""
    return policy in [
        TD3FeedForward, TD3GoalConditioned, TD3MultiFeedForwardPolicy]


def is_sac_policy(policy):
    """Check whether a policy is for designed to support SAC."""
    return policy in [
        SACFeedForward, SACGoalConditioned, SACMultiFeedForwardPolicy]


def is_feedforward_policy(policy):
    """Check whether a policy is a feedforward policy."""
    return policy in [TD3FeedForward, SACFeedForward]


def is_goal_conditioned_policy(policy):
    """Check whether a policy is a goal-conditioned policy."""
    return policy in [TD3GoalConditioned, SACGoalConditioned]


def is_multiagent_policy(policy):
    """Check whether a policy is a multi-agent feedforward policy."""
    return policy in [TD3MultiFeedForwardPolicy, SACMultiFeedForwardPolicy]


def add_fingerprint(obs, steps, total_steps, use_fingerprints):
    """Add a fingerprint element to the observation.

    This should be done when setting "use_fingerprints" in policy_kwargs to
    True. The new observation looks as follows:

              ---------------------------------------------------
    new_obs = || obs || 5 * frac_steps || 5 * (1 - frac_steps) ||
              ---------------------------------------------------

    where frac_steps is the fraction of the total requested number of training
    steps that have been performed. Note that the "5" term is a fixed
    hyperparameter, and can be changed based on its effect on training
    performance.

    If "use_fingerprints" is set to False in policy_kwargs, or simply not
    specified, this method returns the current observation without the
    fingerprint term.

    Parameters
    ----------
    obs : array_like
        the current observation without the fingerprint element
    steps : int
        the total number of steps that have been performed
    total_steps : int
        the total number of samples to train on. Used by the fingerprint
        element
    use_fingerprints : bool
        specifies whether to add a time-dependent fingerprint to the
        observations

    Returns
    -------
    array_like
        the observation with the fingerprint element
    """
    # If the fingerprint element should not be added, simply return the
    # current observation.
    if not use_fingerprints:
        return obs

    # Compute the fingerprint term.
    frac_steps = float(steps) / float(total_steps)
    fp = [5 * frac_steps, 5 * (1 - frac_steps)]

    # Append the fingerprint term to the current observation.
    new_obs = np.concatenate((obs, fp), axis=0)

    return new_obs


def get_obs(obs):
    """Get the observation from a (potentially unprocessed) variable.

    We assume multi-agent MADDPG style policies return a dictionary
    observations, containing the keys "obs" and "all_obs".

    Parameters
    ----------
    obs : array_like
        the current observation

    Returns
    -------
    array_like
        the agent-level observation. May be the initial observation
    array_like or None
        the full-state observation, if using environments that support MADDPG.
        Otherwise, this variable is a None value.
    """
    if isinstance(obs, dict) and "all_obs" in obs.keys():
        all_obs = obs["all_obs"]
        obs = obs["obs"]
    else:
        all_obs = None
        obs = obs

    return obs, all_obs


def collect_sample(env,
                   action,
                   env_num,
                   multiagent,
                   steps,
                   total_steps,
                   use_fingerprints=False):
    """Perform the sample collection operation over a single step.

    This method is responsible for executing a single step of the environment.
    This is perform a number of times in the _collect_samples method before
    training is executed. The data from the rollouts is stored in the policy's
    replay buffer(s).

    Parameters
    ----------
    env : gym.Env
        the environment to collect samples from
    action : array_like
        the action to be performed by the agent(s) within the environment
    env_num : int
        the environment number. Used to handle situations when multiple
        parallel environments are being used.
    multiagent : bool
        whether the policy is multi-agent
    steps : int
        the total number of steps that have been executed since training
        began
    total_steps : int
        the total number of samples to train on. Used by the fingerprint
        element
    use_fingerprints : bool
        specifies whether to add a time-dependent fingerprint to the
        observations

    Returns
    -------
    dict
        information from the most recent environment update step, consisting of
        the following terms:

        * obs : the most recent observation. This consists of a single
          observation if no reset occured, and a tuple of (last observation
          from the previous rollout, first observation of the next rollout) if
          a reset occured.
        * context : the contextual term from the environment
        * action : the action performed by the agent(s)
        * reward : the reward from the most recent step
        * done : the done mask
        * env_num : the environment number
        * all_obs : the most recent full-state observation. This consists of a
          single observation if no reset occured, and a tuple of (last
          observation from the previous rollout, first observation of the next
          rollout) if a reset occured.
    """
    # Execute the next action.
    obs, reward, done, info = env.step(action)
    obs, all_obs = get_obs(obs)

    # Done mask for multi-agent policies is slightly different.
    if multiagent:
        done = done["__all__"]

    # Get the contextual term.
    context = getattr(env, "current_context", None)

    # Add the fingerprint term to this observation, if needed.
    obs = add_fingerprint(obs, steps, total_steps, use_fingerprints)

    if done:
        # Reset the environment.
        reset_obs = env.reset()
        reset_obs, reset_all_obs = get_obs(reset_obs)

        # Add the fingerprint term, if needed.
        obs = add_fingerprint(obs, steps, total_steps, use_fingerprints)
    else:
        reset_obs = None
        reset_all_obs = None

    return {
        "obs": obs if not done else (obs, reset_obs),
        "context": context,
        "action": action,
        "reward": reward,
        "done": done,
        "env_num": env_num,
        "all_obs": all_obs if not done else (all_obs, reset_all_obs)
    }
