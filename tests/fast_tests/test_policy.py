"""Contains tests for the model abstractions and different models."""
import unittest
from hbaselines.hiro.policy import FeedForwardPolicy, GoalDirectedPolicy


class TestFeedForwardPolicy(unittest.TestCase):
    """Test the FeedForwardPolicy object in hbaselines/hiro/policy.py."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        """Validate that the graph and variables are initialized properly."""
        pass


class TestGoalDirectedPolicy(unittest.TestCase):
    """Test the GoalDirectedPolicy object in hbaselines/hiro/policy.py."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        """Validate that the graph and variables are initialized properly."""
        pass

    def test_meta_period(self):
        """Verify that the rate of the Manager is dictated by meta_period."""
        pass

    def test_relative_goals(self):
        """Validate the functionality of relative goals.

        This should affect the worker reward function as well as TODO.
        """
        pass

    def test_off_policy_corrections(self):
        """Validate the functionality of the off-policy corrections.

        TODO: describe content
        """

    def test_fingerprints(self):
        """Validate the functionality of the fingerprints.

        This feature should TODO: describe content
        """
        pass

    def test_centralized_value_functions(self):
        """Validate the functionality of the centralized value function.

        TODO: describe content
        """
        pass

    def test_connected_gradients(self):
        """Validate the functionality of the connected-gradients feature.

        TODO: describe content
        """
        pass


if __name__ == '__main__':
    unittest.main()
