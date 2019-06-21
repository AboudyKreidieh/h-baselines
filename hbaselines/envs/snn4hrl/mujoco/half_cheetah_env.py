import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides


class HalfCheetah_biMod_Env(HalfCheetahEnv):
    """
    Change reward to abs value of the CoMvel: forward and backward equally rewarded
    Add logging of bimodality progress
    """
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        # instead of -1 put -abs() so that both forward and backwards vel is good (2 modes?)
        run_cost = - abs(self.get_body_comvel("torso")[0])
        cost = ctrl_cost + run_cost
        reward = -cost
        done = False
        return Step(next_obs, reward, done)

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular_misc_stat('Progress', progs, 'front')

        largest_positive_prog = max(0, np.max(progs))
        largest_negative_prog = min(0, np.min(progs))
        if abs(largest_negative_prog) > 10e-8 and abs(largest_positive_prog) > 10e-8:
            bimod_ratio = min(abs(largest_negative_prog/largest_positive_prog),
                              abs(largest_positive_prog/largest_negative_prog))
        else:
            bimod_ratio = 0
        logger.record_tabular('BimodalityProgress', bimod_ratio)

