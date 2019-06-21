from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.snn4hrl.envs.mujoco.ant_env import AntEnv


class AntGatherEnv(GatherEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 3

