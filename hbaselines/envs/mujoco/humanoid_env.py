"""Base humanoid environment."""
import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env

# Directory that contains mujoco xml files.
SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(SCRIPT_PATH, 'assets')


def mass_center(model, sim):
    """Compute the position of the agent's center of mass."""
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Humanoid mujoco environment."""

    FILE = 'humanoid.xml'

    def __init__(
            self,
            horizon=1000):

        self.horizon = horizon
        self.t = 0

        file_path = os.path.join(MODEL_DIR, HumanoidEnv.FILE)
        mujoco_env.MujocoEnv.__init__(self, file_path, 5)
        utils.EzPickle.__init__(self)

    def reset(self):
        """Reset the environment."""
        self.t = 0
        return mujoco_env.MujocoEnv.reset(self)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat,
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def step(self, a):
        """Advance the simulation by one step."""
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        r = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        self.t += 1
        d = bool((qpos[2] < 1.0) or (qpos[2] > 2.0) or self.t > self.horizon)
        return self._get_obs(), np.nan_to_num(r), d, dict(
            reward_linvel=lin_vel_cost,
            reward_quadctrl=-quad_ctrl_cost,
            reward_alive=alive_bonus,
            reward_impact=-quad_impact_cost)

    def reset_model(self):
        """Reset the position of the agent."""
        c = 0.01
        rand_qpos = self.np_random.uniform(low=-c, high=c, size=self.model.nq)
        rand_qvel = self.np_random.uniform(low=-c, high=c, size=self.model.nv)
        qpos = self.init_qpos + rand_qpos
        qvel = self.init_qvel + rand_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        """Create the viewer."""
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
