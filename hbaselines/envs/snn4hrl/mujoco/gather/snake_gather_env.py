from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.snn4hrl.envs.mujoco.snake_env import SnakeEnv


class SnakeGatherEnv(GatherEnv):

    MODEL_CLASS = SnakeEnv
    ORI_IND = 2
