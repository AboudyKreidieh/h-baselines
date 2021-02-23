"""Utility method for the algorithm classes."""
from hbaselines.fcnet.td3 import FeedForwardPolicy as TD3FeedForwardPolicy
from hbaselines.fcnet.sac import FeedForwardPolicy as SACFeedForwardPolicy
from hbaselines.fcnet.ppo import FeedForwardPolicy as PPOFeedForwardPolicy
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy as \
    TD3GoalConditionedPolicy
from hbaselines.goal_conditioned.sac import GoalConditionedPolicy as \
    SACGoalConditionedPolicy
from hbaselines.multi_fcnet.td3 import MultiFeedForwardPolicy as \
    TD3MultiFeedForwardPolicyOld
from hbaselines.multi_fcnet.sac import MultiFeedForwardPolicy as \
    SACMultiFeedForwardPolicyOld
from hbaselines.multiagent.td3 import MultiFeedForwardPolicy as \
    TD3MultiFeedForwardPolicy
from hbaselines.multiagent.sac import MultiFeedForwardPolicy as \
    SACMultiFeedForwardPolicy
from hbaselines.multiagent.ppo import MultiFeedForwardPolicy as \
    PPOMultiFeedForwardPolicy
from hbaselines.multiagent.h_td3 import MultiGoalConditionedPolicy as \
    TD3MultiGoalConditionedPolicy
from hbaselines.multiagent.h_sac import MultiGoalConditionedPolicy as \
    SACMultiGoalConditionedPolicy


def is_td3_policy(policy):
    """Check whether a policy is for designed to support TD3."""
    return policy in [
        TD3FeedForwardPolicy,
        TD3GoalConditionedPolicy,
        TD3MultiFeedForwardPolicy,
        TD3MultiFeedForwardPolicyOld,
        TD3MultiGoalConditionedPolicy,
    ]


def is_sac_policy(policy):
    """Check whether a policy is for designed to support SAC."""
    return policy in [
        SACFeedForwardPolicy,
        SACGoalConditionedPolicy,
        SACMultiFeedForwardPolicy,
        SACMultiFeedForwardPolicyOld,
        SACMultiGoalConditionedPolicy,
    ]


def is_ppo_policy(policy):
    """Check whether a policy is for designed to support PPO."""
    return policy in [
        PPOFeedForwardPolicy,
        PPOMultiFeedForwardPolicy,
    ]


def is_feedforward_policy(policy):
    """Check whether a policy is a feedforward policy."""
    return policy in [
        TD3FeedForwardPolicy,
        SACFeedForwardPolicy,
        PPOFeedForwardPolicy,
        TD3MultiFeedForwardPolicy,
        TD3MultiFeedForwardPolicyOld,
        SACMultiFeedForwardPolicy,
        SACMultiFeedForwardPolicyOld,
    ]


def is_goal_conditioned_policy(policy):
    """Check whether a policy is a goal-conditioned policy."""
    return policy in [
        TD3GoalConditionedPolicy,
        SACGoalConditionedPolicy,
        TD3MultiGoalConditionedPolicy,
        SACMultiGoalConditionedPolicy,
    ]


def is_multiagent_policy(policy):
    """Check whether a policy is a multi-agent policy."""
    return policy in [
        TD3MultiFeedForwardPolicy,
        TD3MultiFeedForwardPolicyOld,
        SACMultiFeedForwardPolicy,
        SACMultiFeedForwardPolicyOld,
        PPOMultiFeedForwardPolicy,
        TD3MultiGoalConditionedPolicy,
        SACMultiGoalConditionedPolicy,
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
