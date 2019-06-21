from sandbox.snn4hrl.envs.mujoco.maze.fast_maze_env import FastMazeEnv
from sandbox.snn4hrl.envs.mujoco.snake_env import SnakeEnv


class SnakeMazeEnv(FastMazeEnv):

    MODEL_CLASS = SnakeEnv
    ORI_IND = 2

    MAZE_HEIGHT = 0.5
    MAZE_SIZE_SCALING = 3
    MAZE_MAKE_CONTACTS = True
