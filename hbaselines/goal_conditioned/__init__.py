"""Init file for goal-conditioned submodule."""
from hbaselines.fcnet.td3 import FeedForwardPolicy
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy

__all__ = ["FeedForwardPolicy", "GoalConditionedPolicy"]
