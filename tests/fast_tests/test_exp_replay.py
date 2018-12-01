"""Contains tests for the model abstractions and different models."""
import unittest
import numpy as np

from hbaselines.utils.exp_replay import RecurrentMemory


class TestMemory(unittest.TestCase):
    pass


class TestRecurrentMemory(unittest.TestCase):
    """Tests for the RecurrentMemory object in utils/exp_replay.py."""

    def setUp(self):
        self.exp_replay = RecurrentMemory(
            limit=10,
            action_shape=(1,),
            observation_shape=(5,),
            trace_length=8
        )

    def tearDown(self):
        self.exp_replay = None

    def test_append_no_terminal(self):
        obs0 = [0, 0, 0, 0, 0]
        ac = [1]
        obs1 = [2, 2, 2, 2, 2]
        rew = 3
        terminal = 0

        self.exp_replay.append(obs0, obs1, ac, rew, terminal)
        self.assertTrue(np.all((np.array([[[0, 0, 0, 0, 0]]])
                                == self.exp_replay.observations0)))

        self.exp_replay.append(obs0, obs1, ac, rew, terminal)
        self.assertTrue(np.all(np.array([[[0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0]]])
                               == self.exp_replay.observations0))

        obs0 = [1, 1, 1, 1, 1]
        self.exp_replay.append(obs0, obs1, ac, rew, terminal)
        self.assertTrue(np.all(np.array([[[0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0],
                                          [1, 1, 1, 1, 1]]])
                               == self.exp_replay.observations0))

    def test_append_terminal(self):
        obs0 = [0, 0, 0, 0, 0]
        ac = [1]
        obs1 = [2, 2, 2, 2, 2]
        rew = 3
        terminal = 0

        for i in range(10):
            self.exp_replay.append(obs0, obs1, ac, rew, terminal)
            self.assertTrue(np.all((np.array([[[0, 0, 0, 0, 0]] * (i+1)])
                                    == self.exp_replay.observations0)))
            self.assertEqual(self.exp_replay.nb_entries, 1)

        # check terminal when episode is larger than than trace length
        terminal = 1
        self.exp_replay.append(obs0, obs1, ac, rew, terminal)
        self.assertTrue(np.all(self.exp_replay.observations0[0]
                               == np.array([[0, 0, 0, 0, 0]] * 11)))
        self.assertTrue(len(self.exp_replay.observations0[1]) == 0)
        self.assertEqual(self.exp_replay.nb_entries, 2)

        terminal = 0

        self.exp_replay.append(obs0, obs1, ac, rew, terminal)
        self.assertTrue(np.all(self.exp_replay.observations0[0]
                               == np.array([[0, 0, 0, 0, 0]] * 11)))
        self.assertTrue(np.all(self.exp_replay.observations0[1]
                               == np.array([[0, 0, 0, 0, 0]])))
        self.assertEqual(self.exp_replay.nb_entries, 2)

        # check terminal when episode is shorter than trace length
        terminal = 1
        self.exp_replay.append(obs0, obs1, ac, rew, terminal)
        self.assertTrue(np.all(self.exp_replay.observations0[0]
                               == np.array([[0, 0, 0, 0, 0]] * 11)))
        self.assertTrue(len(self.exp_replay.observations0[1]) == 0)
        self.assertEqual(self.exp_replay.nb_entries, 2)

    def test_sample(self):
        pass


class TestHierarchicalRecurrentMemory(unittest.TestCase):
    """Tests for the HierarchicalRecurrentMemory object in utils/exp_replay.py.
    """

    pass


if __name__ == '__main__':
    unittest.main()
