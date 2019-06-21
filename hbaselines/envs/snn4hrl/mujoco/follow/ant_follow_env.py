from sandbox.snn4hrl.envs.mujoco.follow.follow_env import FollowEnv
from sandbox.snn4hrl.envs.mujoco.ant_env import AntEnv


class AntFollowEnv(FollowEnv):
    MODEL_CLASS = AntEnv
    ORI_IND = 3

