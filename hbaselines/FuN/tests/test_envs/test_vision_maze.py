"""
#########################################
#Class of scripts to testing vision maze#
#########################################
"""


import numpy as np
import unittest
from hbaselines.FuN.scripts.training.feudal_networks.envs.vision_maze import VisionMazeEnv


def to_coords(x):
    """
    Function to convert the passed parameter to coordinates

    Parameters
    ----------
    x : object
        BLANK
    """
    return list(v[0] for v in np.where(x)[:2])


class TestVisionMaze(unittest.TestCase):
    """
    Class for testing Vision maze environment

    """

    def test_step(self):
        """
        Function for initializing the test

        """
        maze = VisionMazeEnv(room_length=3, num_rooms_per_side=2)

        # up until wall
        a = 0
        maze.state = np.array([0, 0])
        nx, _, _, _ = maze.step(a)
        nx, _, _, _ = maze.step(a)
        np.testing.assert_array_equal(to_coords(nx), [0, 2])
        nx, _, _, _ = maze.step(a)
        np.testing.assert_array_equal(to_coords(nx), [0, 2])
        # down until wall
        a = 2
        maze.state = np.array([0, 2])
        nx, _, _, _ = maze.step(a)
        nx, _, _, _ = maze.step(a)
        np.testing.assert_array_equal(to_coords(nx), [0, 0])
        nx, _, _, _ = maze.step(a)
        np.testing.assert_array_equal(to_coords(nx), [0, 0])

        # right until wall
        maze.state = np.array([0, 0])
        a = 1
        nx, _, _, _ = maze.step(a)
        nx, _, _, _ = maze.step(a)
        np.testing.assert_array_equal(to_coords(nx), [2, 0])
        nx, _, _, _ = maze.step(a)
        np.testing.assert_array_equal(to_coords(nx), [2, 0])

        # left until wall
        maze.state = np.array([2, 0])
        a = 3
        nx, _, _, _ = maze.step(a)
        nx, _, _, _ = maze.step(a)
        np.testing.assert_array_equal(to_coords(nx), [0, 0])
        nx, _, _, _ = maze.step(a)
        np.testing.assert_array_equal(to_coords(nx), [0, 0])

        # through doorway to the right until wall
        maze.state = np.array([0, 0])
        nx, _, _, _ = maze.step(0)  # up
        nx, _, _, _ = maze.step(1)  # right
        nx, _, _, _ = maze.step(1)  # right
        nx, _, _, _ = maze.step(1)  # right
        nx, _, _, _ = maze.step(1)  # right
        nx, _, _, _ = maze.step(1)  # right
        np.testing.assert_array_equal(to_coords(nx), [5, 1])
        nx, _, _, _ = maze.step(1)  # right
        np.testing.assert_array_equal(to_coords(nx), [5, 1])

# back through the doorway I came, and then up through the other doorway
        maze.state = np.array([5, 1])
        nx, _, _, _ = maze.step(3)  # left
        nx, _, _, _ = maze.step(3)  # left
        nx, _, _, _ = maze.step(3)  # left
        nx, _, _, _ = maze.step(3)  # left
        nx, _, _, _ = maze.step(0)  # up
        nx, _, _, _ = maze.step(0)  # up
        nx, _, _, _ = maze.step(0)  # up
        nx, _, _, _ = maze.step(0)  # up
        np.testing.assert_array_equal(to_coords(nx), [1, 5])
        nx, _, _, _ = maze.step(0)  # up
        np.testing.assert_array_equal(to_coords(nx), [1, 5])

        # to the goal state
        maze.state = np.array([1, 5])
        nx, _, _, _ = maze.step(1)  # right
        nx, _, _, _ = maze.step(2)  # down
        nx, _, _, _ = maze.step(1)  # right
        nx, _, _, _ = maze.step(1)  # right
        nx, _, _, _ = maze.step(1)  # right
        nx, _, _, _ = maze.step(0)  # up
        np.testing.assert_array_equal(to_coords(nx), [5, 5])

        # down until wall
        maze.state = np.array([5, 5])
        nx, _, _, _ = maze.step(2)  # down
        nx, _, _, _ = maze.step(2)  # down
        np.testing.assert_array_equal(to_coords(nx), [5, 3])
        nx, _, _, _ = maze.step(2)  # down
        np.testing.assert_array_equal(to_coords(nx), [5, 3])


if __name__ == '__main__':
    unittest.main()
