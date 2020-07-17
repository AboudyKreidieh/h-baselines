"""Utility method for the algorithm classes."""
from hbaselines.fcnet.td3 import FeedForwardPolicy as \
    TD3FeedForwardPolicy
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy as \
    TD3GoalConditionedPolicy
from hbaselines.goal_conditioned.sac import GoalConditionedPolicy as \
    SACGoalConditionedPolicy
from hbaselines.fcnet.sac import FeedForwardPolicy as \
    SACFeedForwardPolicy
from hbaselines.multi_fcnet.td3 import MultiFeedForwardPolicy as \
    TD3MultiFeedForwardPolicy
from hbaselines.multi_fcnet.sac import MultiFeedForwardPolicy as \
    SACMultiFeedForwardPolicy


def is_td3_policy(policy):
    """Check whether a policy is for designed to support TD3."""
    return policy in [
        TD3FeedForwardPolicy,
        TD3GoalConditionedPolicy,
        TD3MultiFeedForwardPolicy,
    ]


def is_sac_policy(policy):
    """Check whether a policy is for designed to support SAC."""
    return policy in [
        SACFeedForwardPolicy,
        SACGoalConditionedPolicy,
        SACMultiFeedForwardPolicy,
    ]


def is_feedforward_policy(policy):
    """Check whether a policy is a feedforward policy."""
    return policy in [
        TD3FeedForwardPolicy,
        SACFeedForwardPolicy,
        TD3MultiFeedForwardPolicy,
        SACMultiFeedForwardPolicy,
    ]


def is_goal_conditioned_policy(policy):
    """Check whether a policy is a goal-conditioned policy."""
    return policy in [
        TD3GoalConditionedPolicy,
        SACGoalConditionedPolicy,
    ]


def is_multiagent_policy(policy):
    """Check whether a policy is a multi-agent feedforward policy."""
    return policy in [
        TD3MultiFeedForwardPolicy,
        SACMultiFeedForwardPolicy,
    ]


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
