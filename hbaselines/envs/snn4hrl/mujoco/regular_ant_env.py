from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
import numpy as np

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger

import time
from matplotlib import pyplot as plt

class RegularAntEnv(MujocoEnv, Serializable):

    FILE = 'ant.xml'

    def __init__(self, *args, **kwargs):
        super(RegularAntEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

    def step(self, action):
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        forward_reward = comvel[0]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))

        progs_norm = [
            np.linalg.norm(path["observations"][-1][-3:-1] - path["observations"][0][-3:-1])
            for path in paths
            ]
        logger.record_tabular('AverageForwardProgress_norm', np.mean(progs_norm))
        logger.record_tabular('MaxForwardProgress_norm', np.max(progs_norm))
        logger.record_tabular('MinForwardProgress_norm', np.min(progs_norm))
        logger.record_tabular('StdForwardProgress_norm', np.std(progs_norm))
        # now we will grid the space and check how much of it the policy is covering

        # problem with paths of different lenghts: call twice max
        furthest = np.ceil(np.abs(np.max([np.max(path["observations"][:,-3:-1]) for path in paths])))
        print('THE FUTHEST IT WENT COMPONENT-WISE IS', furthest)
        furthest = max(furthest, 10)

        # c_grid = furthest * 10 * 2
        # visitation = np.zeros((c_grid, c_grid))  # we assume the furthest it can go is 100, Check it!!
        # for path in paths:
        #     com_x = np.clip(((np.array(path['observations'][:, -3]) + furthest) * 10).astype(int), 0, c_grid - 1)
        #     com_y = np.clip(((np.array(path['observations'][:, -2]) + furthest) * 10).astype(int), 0, c_grid - 1)
        #     coms = zip(com_x, com_y)
        #     for com in coms:
        #         visitation[com] += 1
        #
        # # if you want to have a heatmap of the visitations
        # plt.figure()
        # plt.pcolor(visitation)
        # t = str(int(time.time()))
        # plt.savefig('data/local/visitation_regular_ant_trpo/visitation_map_' + t)
        #
        # total_visitation = np.count_nonzero(visitation)
        # logger.record_tabular('VisitationTotal', total_visitation)
