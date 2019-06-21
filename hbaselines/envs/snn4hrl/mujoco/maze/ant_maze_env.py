from sandbox.snn4hrl.envs.mujoco.maze.fast_maze_env import FastMazeEnv
from sandbox.snn4hrl.envs.mujoco.ant_env import AntEnv


class AntMazeEnv(FastMazeEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 3  # the ori of Ant requires quaternion conversion and is implemented in AntEnv

    MAZE_HEIGHT = 3
    MAZE_SIZE_SCALING = 3.0

