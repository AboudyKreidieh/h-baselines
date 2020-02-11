import unittest
import numpy as np

from hbaselines.fcnet.replay_buffer import ReplayBuffer
from hbaselines.goal_conditioned.replay_buffer import HierReplayBuffer
from hbaselines.multi_fcnet.replay_buffer import MultiReplayBuffer
from hbaselines.multi_fcnet.replay_buffer import SharedReplayBuffer


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
            meta_period=1,
            meta_obs_dim=2,
            meta_ac_dim=3,
            worker_obs_dim=4,
            worker_ac_dim=5
        )

    def tearDown(self):
        del self.replay_buffer

    def test_init(self):
        """Validate that all the attributes were initialize properly."""
        self.assertTupleEqual(self.replay_buffer.meta_obs0.shape, (1, 2))
        self.assertTupleEqual(self.replay_buffer.meta_obs1.shape, (1, 2))
        self.assertTupleEqual(self.replay_buffer.meta_act.shape, (1, 3))
        self.assertTupleEqual(self.replay_buffer.meta_rew.shape, (1,))
        self.assertTupleEqual(self.replay_buffer.meta_done.shape, (1,))
        self.assertTupleEqual(self.replay_buffer.worker_obs0.shape, (1, 4))
        self.assertTupleEqual(self.replay_buffer.worker_obs1.shape, (1, 4))
        self.assertTupleEqual(self.replay_buffer.worker_act.shape, (1, 5))
        self.assertTupleEqual(self.replay_buffer.worker_rew.shape, (1,))
        self.assertTupleEqual(self.replay_buffer.worker_done.shape, (1,))

    def test_buffer_size(self):
        """Validate the buffer_size output from the replay buffer."""
        self.assertEqual(self.replay_buffer.buffer_size, 2)

    def test_add_sample(self):
        """Test the `add` and `sample` methods the replay buffer."""
        """Test the `add` and `sample` methods the replay buffer."""
        # Add an element.
        self.replay_buffer.add(
            obs_t=[np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1])],
            goal_t=np.array([2, 2, 2]),
            action_t=[np.array([3, 3, 3, 3, 3])],
            reward_t=[4],
            done=[False],
            meta_obs_t=(np.array([5, 5]), np.array([6, 6])),
            meta_reward_t=7,
        )

        # Check is_full in the False case.
        self.assertEqual(self.replay_buffer.is_full(), False)

        # Add an element.
        self.replay_buffer.add(
            obs_t=[np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1])],
            goal_t=np.array([2, 2, 2]),
            action_t=[np.array([3, 3, 3, 3, 3])],
            reward_t=[4],
            done=[False],
            meta_obs_t=(np.array([5, 5]), np.array([6, 6])),
            meta_reward_t=7,
        )

        # Check is_full in the True case.
        self.assertEqual(self.replay_buffer.is_full(), True)

        # Check can_sample in the True case.
        self.assertEqual(self.replay_buffer.can_sample(), True)

        # Test the `sample` method.
        meta_obs0, meta_obs1, meta_act, meta_rew, meta_done, worker_obs0, \
            worker_obs1, worker_act, worker_rew, worker_done, _ = \
            self.replay_buffer.sample()
        np.testing.assert_array_almost_equal(meta_obs0, [[5, 5]])
        np.testing.assert_array_almost_equal(meta_obs1, [[6, 6]])
        np.testing.assert_array_almost_equal(meta_act, [[2, 2, 2]])
        np.testing.assert_array_almost_equal(meta_rew, [7])
        np.testing.assert_array_almost_equal(meta_done, [0])
        np.testing.assert_array_almost_equal(worker_obs0, [[0, 0, 0, 0]])
        np.testing.assert_array_almost_equal(worker_obs1, [[1, 1, 1, 1]])
        np.testing.assert_array_almost_equal(worker_act, [[3, 3, 3, 3, 3]])
        np.testing.assert_array_almost_equal(worker_rew, [4])
        np.testing.assert_array_almost_equal(worker_done, [0])


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
