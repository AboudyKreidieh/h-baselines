from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.snn4hrl.envs.mujoco.swimmer_env import SwimmerEnv


class SwimmerGatherEnv(GatherEnv):

    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2
