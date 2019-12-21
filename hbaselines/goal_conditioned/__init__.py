"""Init file for goal-conditioned submodule."""
from hbaselines.goal_conditioned.policy import FeedForwardPolicy
from hbaselines.goal_conditioned.policy import GoalConditionedPolicy

__all__ = ["FeedForwardPolicy", "GoalConditionedPolicy"]
