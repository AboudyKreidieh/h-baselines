"""Init file for goal-conditioned submodule."""
from hbaselines.goal_conditioned.algorithm import TD3
from hbaselines.goal_conditioned.policies.td3 import FeedForwardPolicy
from hbaselines.goal_conditioned.policies.td3 import GoalConditionedPolicy

__all__ = ["FeedForwardPolicy", "GoalConditionedPolicy", "TD3"]
