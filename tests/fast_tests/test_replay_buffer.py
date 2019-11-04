import unittest
import numpy as np

from hbaselines.goal_conditioned.replay_buffer import ReplayBuffer
from hbaselines.goal_conditioned.replay_buffer import HierReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    """Tests for the ReplayBuffer object."""

    def setUp(self):
        self.replay_buffer = ReplayBuffer(size=2)

    def tearDown(self):
        del self.replay_buffer

    def test_buffer_size(self):
        """Validate the buffer_size output from the replay buffer."""
        self.assertEqual(self.replay_buffer.buffer_size, 2)

    def test_add_sample(self):
        """Test the `add` and `sample` methods the replay buffer."""
        # Add an element.
        self.replay_buffer.add(
            obs_t=np.array([0]),
            action=np.array([1]),
            reward=2,
            obs_tp1=np.array([3]),
            done=False
        )

        # Check is_full in the False case.
        self.assertEqual(self.replay_buffer.is_full(), False)

        # Add an element.
        self.replay_buffer.add(
            obs_t=np.array([0]),
            action=np.array([1]),
            reward=2,
            obs_tp1=np.array([3]),
            done=False
        )

        # Check is_full in the True case.
        self.assertEqual(self.replay_buffer.is_full(), True)

        # Check can_sample in the True case.
        self.assertEqual(self.replay_buffer.can_sample(1), True)

        # Check can_sample in the True case.
        self.assertEqual(self.replay_buffer.can_sample(3), False)

        # Test the `sample` method.
        obs_t, actions, rewards, obs_tp1, done = self.replay_buffer.sample(1)
        np.testing.assert_array_almost_equal(obs_t, [[0]])
        np.testing.assert_array_almost_equal(actions, [[1]])
        np.testing.assert_array_almost_equal(rewards, [2])
        np.testing.assert_array_almost_equal(obs_tp1, [[3]])
        np.testing.assert_array_almost_equal(done, [False])


class TestHierReplayBuffer(unittest.TestCase):
    """Tests for the HierReplayBuffer object."""

    def setUp(self):
        self.replay_buffer = HierReplayBuffer(size=2)

    def tearDown(self):
        del self.replay_buffer


if __name__ == '__main__':
    unittest.main()
