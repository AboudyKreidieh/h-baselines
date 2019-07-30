"""Contains tests for the contained environments."""
import unittest
import numpy as np

from hbaselines.envs.efficient_hrl.maze_env_utils import line_intersect, \
    point_distance, construct_maze


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


if __name__ == '__main__':
    unittest.main()
