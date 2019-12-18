from hbaselines.fcnet.td3 import FeedForwardPolicy as TD3FeedForward
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy as \
    TD3GoalConditioned


def is_td3_policy(policy):
    """Check whether a policy is for designed to support TD3."""
    return policy in [TD3FeedForward, TD3GoalConditioned]


def is_sac_policy(policy):
    """Check whether a policy is for designed to support SAC."""
    return policy in []


def is_feedforward_policy(policy):
    """Check whether a policy is a feedforward policy."""
    return policy in [TD3FeedForward]


def is_goal_conditioned_policy(policy):
    """Check whether a policy is a goal-conditioned policy."""
    return policy in [TD3GoalConditioned]
