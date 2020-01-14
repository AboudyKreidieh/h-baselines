"""Utility method for the algorithm classes."""
from hbaselines.fcnet.td3 import FeedForwardPolicy as TD3FeedForward
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy as \
    TD3GoalConditioned
from hbaselines.goal_conditioned.sac import GoalConditionedPolicy as \
    SACGoalConditioned
from hbaselines.fcnet.sac import FeedForwardPolicy as SACFeedForward


def is_td3_policy(policy):
    """Check whether a policy is for designed to support TD3."""
    return policy in [TD3FeedForward, TD3GoalConditioned]


def is_sac_policy(policy):
    """Check whether a policy is for designed to support SAC."""
    return policy in [SACFeedForward, SACGoalConditioned]


def is_feedforward_policy(policy):
    """Check whether a policy is a feedforward policy."""
    return policy in [TD3FeedForward, SACFeedForward]


def is_goal_conditioned_policy(policy):
    """Check whether a policy is a goal-conditioned policy."""
    return policy in [TD3GoalConditioned, SACGoalConditioned]
