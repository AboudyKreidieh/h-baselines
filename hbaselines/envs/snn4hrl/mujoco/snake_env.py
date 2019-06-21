from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from sandbox.snn4hrl.envs.mujoco.mujoco_env import MujocoEnv_ObsInit as MujocoEnv
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
import numpy as np

BIG = 1e6


class SnakeEnv(MujocoEnv, Serializable):
    FILE = 'snake.xml'
    ORI_IND = 2

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,
            ctrl_cost_coeff=1e-2,
            ego_obs=False,
            sparse_rew=False,
            *args, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.ego_obs = ego_obs
        self.sparse_rew = sparse_rew
        super(SnakeEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        if self.ego_obs:
            return np.concatenate([
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
            ]).reshape(-1)
        else:
            return np.concatenate([
                self.model.data.qpos.flat,
                self.model.data.qvel.flat,
                self.get_body_com("torso").flat,
            ]).reshape(-1)

    def get_ori(self):
        return self.model.data.qpos[self.__class__.ORI_IND]

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        forward_reward = np.linalg.norm(self.get_body_comvel("torso"))  # swimmer has no problem of jumping reward
        reward = forward_reward - ctrl_cost
        done = False
        if self.sparse_rew:
            if abs(self.get_body_com("torso")[0]) > 100.0:
                reward = 1.0
                done = True
            else:
                reward = 0.
        com = np.concatenate([self.get_body_com("torso").flat]).reshape(-1)
        ori = self.get_ori()
        return Step(next_obs, reward, done, com=com, ori=ori)

    @overrides
    def log_diagnostics(self, paths, prefix=''):
        progs = [
            np.linalg.norm(path["env_infos"]["com"][-1] - path["env_infos"]["com"][0])
            for path in paths
            ]
        logger.record_tabular_misc_stat('Progress', progs, 'front')
        self.plot_visitations(paths, visit_prefix=prefix)

