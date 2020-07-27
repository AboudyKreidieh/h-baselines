import unittest
import random
import numpy as np

from hbaselines.fcnet.replay_buffer import ReplayBuffer
from hbaselines.goal_conditioned.replay_buffer import HierReplayBuffer
from hbaselines.multiagent.replay_buffer import MultiReplayBuffer
from hbaselines.multiagent.replay_buffer import SharedReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    """Tests for the ReplayBuffer object."""

    def setUp(self):
        self.replay_buffer = ReplayBuffer(
            buffer_size=2, batch_size=1, obs_dim=1, ac_dim=1)

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
        self.assertEqual(self.replay_buffer.can_sample(), True)

        # Test the `sample` method.
        obs_t, actions_t, rewards, obs_tp1, done = self.replay_buffer.sample()
        np.testing.assert_array_almost_equal(obs_t, [[0]])
        np.testing.assert_array_almost_equal(actions_t, [[1]])
        np.testing.assert_array_almost_equal(rewards, [2])
        np.testing.assert_array_almost_equal(obs_tp1, [[3]])
        np.testing.assert_array_almost_equal(done, [False])


class TestHierReplayBuffer(unittest.TestCase):
    """Tests for the HierReplayBuffer object."""

    def setUp(self):
        self.replay_buffer = HierReplayBuffer(
            buffer_size=2,
            batch_size=1,
            meta_period=3,
            obs_dim=1,
            ac_dim=1,
            co_dim=1,
            goal_dim=1,
            num_levels=3,
        )

    def tearDown(self):
        del self.replay_buffer

    def test_buffer_size(self):
        """Validate the buffer_size output from the replay buffer."""
        self.assertEqual(self.replay_buffer.buffer_size, 2)

    def test_add_sample(self):
        """Test the `add` and `sample` methods the replay buffer."""
        # Set the random seed.
        random.seed(0)

        obs_t = [np.array([0]), np.array([1]), np.array([2]),
                 np.array([3]), np.array([4]), np.array([5]),
                 np.array([6]), np.array([7]), np.array([8]),
                 np.array([9])]
        action_t = [[np.array([0]), np.array([1]), np.array([2]),
                     np.array([3])],
                    [np.array([0]), np.array([1]), np.array([2]),
                     np.array([3]), np.array([4]), np.array([5]),
                     np.array([6]), np.array([7]), np.array([8]),
                     np.array([9])],
                    [np.array([0]), np.array([1]), np.array([2]),
                     np.array([3]), np.array([4]), np.array([5]),
                     np.array([6]), np.array([7]), np.array([8]),
                     np.array([9])]]
        context_t = [np.array([0]), np.array([1])]
        reward_t = [[0], [0, 1, 2], [0, 1, 2, 3, 4, 5, 6, 7, 8]]
        done_t = [
            False, False, False, False, False, False, False, False, False]

        # Add an element.
        self.replay_buffer.add(
            obs_t=obs_t,
            action_t=action_t,
            context_t=context_t,
            reward_t=reward_t,
            done_t=done_t,
        )

        # Check is_full in the False case.
        self.assertEqual(self.replay_buffer.is_full(), False)

        # Add an element.
        self.replay_buffer.add(
            obs_t=obs_t,
            action_t=action_t,
            context_t=context_t,
            reward_t=reward_t,
            done_t=done_t,
        )

        # Check is_full in the True case.
        self.assertEqual(self.replay_buffer.is_full(), True)

        # Check can_sample in the True case.
        self.assertEqual(self.replay_buffer.can_sample(), True)

        # Test the `sample` method.
        obs0, obs1, act, rew, done, _ = self.replay_buffer.sample(False)
        np.testing.assert_array_almost_equal(obs0[0], [[0, 0]])
        np.testing.assert_array_almost_equal(obs0[1], [[6, 2]])
        np.testing.assert_array_almost_equal(obs0[2], [[6, 6]])

        np.testing.assert_array_almost_equal(obs1[0], [[9, 1]])
        np.testing.assert_array_almost_equal(obs1[1], [[9, 3]])
        np.testing.assert_array_almost_equal(obs1[2], [[7, 7]])

        np.testing.assert_array_almost_equal(act[0], [[0]])
        np.testing.assert_array_almost_equal(act[1], [[6]])
        np.testing.assert_array_almost_equal(act[2], [[6]])

        np.testing.assert_array_almost_equal(rew[0], [0])
        np.testing.assert_array_almost_equal(rew[1], [2])
        np.testing.assert_array_almost_equal(rew[2], [6])

        np.testing.assert_array_almost_equal(done[0], [0])
        np.testing.assert_array_almost_equal(done[1], [0])
        np.testing.assert_array_almost_equal(done[2], [0])


class TestMultiReplayBuffer(unittest.TestCase):
    """Tests for the MultiReplayBuffer object."""

    def setUp(self):
        self.replay_buffer = MultiReplayBuffer(
            buffer_size=2,
            batch_size=1,
            obs_dim=1,
            ac_dim=2,
            all_obs_dim=3,
            all_ac_dim=4
        )

    def tearDown(self):
        del self.replay_buffer

    def test_init(self):
        """Validate that all the attributes were initialize properly."""
        self.assertTupleEqual(self.replay_buffer.obs_t.shape, (2, 1))
        self.assertTupleEqual(self.replay_buffer.action_t.shape, (2, 2))
        self.assertTupleEqual(self.replay_buffer.reward.shape, (2,))
        self.assertTupleEqual(self.replay_buffer.obs_tp1.shape, (2, 1))
        self.assertTupleEqual(self.replay_buffer.done.shape, (2,))
        self.assertTupleEqual(self.replay_buffer.all_obs_t.shape, (2, 3))
        self.assertTupleEqual(self.replay_buffer.all_action_t.shape, (2, 4))
        self.assertTupleEqual(self.replay_buffer.all_obs_tp1.shape, (2, 3))

    def test_buffer_size(self):
        """Validate the buffer_size output from the replay buffer."""
        self.assertEqual(self.replay_buffer.buffer_size, 2)

    def test_add_sample(self):
        """Test the `add` and `sample` methods the replay buffer."""
        # Add an element.
        self.replay_buffer.add(
            obs_t=np.array([0]),
            action=np.array([1, 1]),
            reward=2,
            obs_tp1=np.array([3]),
            done=False,
            all_obs_t=np.array([4, 4, 4]),
            all_action_t=np.array([5, 5, 5, 5]),
            all_obs_tp1=np.array([6, 6, 6])
        )

        # Check is_full in the False case.
        self.assertEqual(self.replay_buffer.is_full(), False)

        # Add an element.
        self.replay_buffer.add(
            obs_t=np.array([0]),
            action=np.array([1, 1]),
            reward=2,
            obs_tp1=np.array([3]),
            done=False,
            all_obs_t=np.array([4, 4, 4]),
            all_action_t=np.array([5, 5, 5, 5]),
            all_obs_tp1=np.array([6, 6, 6])
        )

        # Check is_full in the True case.
        self.assertEqual(self.replay_buffer.is_full(), True)

        # Check can_sample in the True case.
        self.assertEqual(self.replay_buffer.can_sample(), True)

        # Test the `sample` method.
        obs_t, actions_t, rewards, obs_tp1, done, all_obs_t, all_actions_t, \
            all_obs_tp1 = self.replay_buffer.sample()
        np.testing.assert_array_almost_equal(obs_t, [[0]])
        np.testing.assert_array_almost_equal(actions_t, [[1, 1]])
        np.testing.assert_array_almost_equal(rewards, [2])
        np.testing.assert_array_almost_equal(obs_tp1, [[3]])
        np.testing.assert_array_almost_equal(done, [False])
        np.testing.assert_array_almost_equal(all_obs_t, [[4, 4, 4]])
        np.testing.assert_array_almost_equal(all_actions_t, [[5, 5, 5, 5]])
        np.testing.assert_array_almost_equal(all_obs_tp1, [[6, 6, 6]])


class TestSharedReplayBuffer(unittest.TestCase):
    """Tests for the SharedReplayBuffer object."""

    def setUp(self):
        self.replay_buffer = SharedReplayBuffer(
            buffer_size=2,
            batch_size=1,
            obs_dim=1,
            ac_dim=2,
            n_agents=3,
            all_obs_dim=4
        )

    def tearDown(self):
        del self.replay_buffer

    def test_init(self):
        """Validate that all the attributes were initialize properly."""
        # These variables are stored for all agents, so should be a list for
        # each agent.
        self.assertEqual(len(self.replay_buffer.obs_t), 3)
        self.assertEqual(len(self.replay_buffer.action), 3)
        self.assertEqual(len(self.replay_buffer.obs_tp1), 3)

        # Check the sizes of the individual variables.
        self.assertTupleEqual(self.replay_buffer.reward.shape, (2,))
        self.assertTupleEqual(self.replay_buffer.done.shape, (2,))
        self.assertTupleEqual(self.replay_buffer.all_obs_t.shape, (2, 4))
        self.assertTupleEqual(self.replay_buffer.all_obs_tp1.shape, (2, 4))
        for i in range(3):  # loop through num_agents
            self.assertTupleEqual(self.replay_buffer.obs_t[i].shape, (2, 1))
            self.assertTupleEqual(self.replay_buffer.action[i].shape, (2, 2))
            self.assertTupleEqual(self.replay_buffer.obs_tp1[i].shape, (2, 1))

    def test_buffer_size(self):
        """Validate the buffer_size output from the replay buffer."""
        self.assertEqual(self.replay_buffer.buffer_size, 2)

    def test_add_sample(self):
        """Test the `add` and `sample` methods the replay buffer."""
        # Add an element.
        self.replay_buffer.add(
            obs_t=[np.array([0]), np.array([1]), np.array([2])],
            action=[np.array([3, 3]), np.array([4, 4]), np.array([5, 5])],
            reward=6,
            obs_tp1=[np.array([7]), np.array([8]), np.array([9])],
            done=False,
            all_obs_t=np.array([10, 10, 10, 10]),
            all_obs_tp1=np.array([11, 11, 11, 11]),
        )

        # Check is_full in the False case.
        self.assertEqual(self.replay_buffer.is_full(), False)

        # Add an element.
        self.replay_buffer.add(
            obs_t=[np.array([0]), np.array([1]), np.array([2])],
            action=[np.array([3, 3]), np.array([4, 4]), np.array([5, 5])],
            reward=6,
            obs_tp1=[np.array([7]), np.array([8]), np.array([9])],
            done=False,
            all_obs_t=np.array([10, 10, 10, 10]),
            all_obs_tp1=np.array([11, 11, 11, 11]),
        )

        # Check is_full in the True case.
        self.assertEqual(self.replay_buffer.is_full(), True)

        # Check can_sample in the True case.
        self.assertEqual(self.replay_buffer.can_sample(), True)

        # Test the `sample` method.
        obs_t, actions_t, rewards, obs_tp1, done, all_obs_t, all_obs_tp1 = \
            self.replay_buffer.sample()

        np.testing.assert_array_almost_equal(rewards, [6])
        np.testing.assert_array_almost_equal(done, [False])
        np.testing.assert_array_almost_equal(all_obs_t, [[10, 10, 10, 10]])
        np.testing.assert_array_almost_equal(all_obs_tp1, [[11, 11, 11, 11]])

        np.testing.assert_array_almost_equal(obs_t[0], [[0]])
        np.testing.assert_array_almost_equal(obs_t[1], [[1]])
        np.testing.assert_array_almost_equal(obs_t[2], [[2]])

        np.testing.assert_array_almost_equal(actions_t[0], [[3, 3]])
        np.testing.assert_array_almost_equal(actions_t[1], [[4, 4]])
        np.testing.assert_array_almost_equal(actions_t[2], [[5, 5]])

        np.testing.assert_array_almost_equal(obs_tp1[0], [[7]])
        np.testing.assert_array_almost_equal(obs_tp1[1], [[8]])
        np.testing.assert_array_almost_equal(obs_tp1[2], [[9]])


if __name__ == '__main__':
    unittest.main()
