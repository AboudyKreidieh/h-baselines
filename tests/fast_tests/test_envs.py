"""Contains tests for the contained environments."""
import unittest
import numpy as np

from hbaselines.envs.efficient_hrl.maze_env_utils import line_intersect, \
    point_distance, construct_maze
from hbaselines.envs.efficient_hrl.envs import AntMaze, AntFall, AntPush


class TestEfficientHRLEnvironments(unittest.TestCase):
    """Test the environments in envs/efficient_hrl/."""

    def test_maze_env_utils(self):
        """Test hbaselines/envs/efficient_hrl/maze_env_utils.py."""
        # test construct_maze
        for maze_id in ["Maze", "Push", "Fall", "Block", "BlockMaze"]:
            construct_maze(maze_id)
        self.assertRaises(NotImplementedError, construct_maze, maze_id="error")

        # test point_distance
        p1 = (0, 0)
        p2 = (2, 2)
        self.assertAlmostEqual(point_distance(p1, p2), np.sqrt(2**2 + 2**2))

        # test line_intersect
        p1 = (0, 0)
        p2 = (2, 2)
        p3 = (0, 2)
        p4 = (2, 0)
        x, y, *_ = line_intersect(p1, p2, p3, p4)
        self.assertAlmostEqual(x, 1)
        self.assertAlmostEqual(y, 1)

    def test_envs(self):
        """Test hbaselines/envs/efficient_hrl/envs.py."""
        # test AntMaze
        env = AntMaze(use_contexts=True, context_range=[0, 0])
        self.assertAlmostEqual(
            env.contextual_reward(
                np.array([0, 0]), np.array([1, 1]), np.array([2, 2])),
            -0.14142135624084504
        )

        # test AntPush
        env = AntPush(use_contexts=True, context_range=[0, 0])
        self.assertAlmostEqual(
            env.contextual_reward(
                np.array([0, 0]), np.array([1, 1]), np.array([2, 2])),
            -0.14142135624084504
        )

        # test AntFall
        env = AntFall(use_contexts=True, context_range=[0, 0, 0])
        self.assertAlmostEqual(
            env.contextual_reward(
                np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2])),
            -0.17320508075977448
        )

        # test context_space
        env = AntMaze(use_contexts=True, random_contexts=True,
                      context_range=[(-4, 5), (4, 20)])
        np.testing.assert_almost_equal(
            env.context_space.low, np.array([-4, 4]))
        np.testing.assert_almost_equal(
            env.context_space.high, np.array([5, 20]))

        # test context_space without contexts
        env = AntMaze(use_contexts=False)
        self.assertIsNone(env.context_space)


if __name__ == '__main__':
    unittest.main()
