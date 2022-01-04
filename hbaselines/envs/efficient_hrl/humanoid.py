"""Script containing the Humanoid environment."""
import math
import numpy as np
import gym
from gym import utils
try:
    from gym.envs.mujoco import mujoco_env
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):

    def mujoco_env():
        """Create a dummy environment for testing purposes."""
        return None
    setattr(mujoco_env, "MujocoEnv", gym.Env)


def mass_center(model, sim):
    """Compute the position of the agent's center of mass."""
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


def q_inv(a):
    """Return the inverse of a quaternion."""
    return [a[0], -a[1], -a[2], -a[3]]


def q_mult(a, b):
    """Multiply two quaternion."""
    w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return [w, i, j, k]


class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Humanoid mujoco environment."""

    FILE = 'double_humanoid.xml'

    def __init__(
            self,
            file_path):
        """Create a humanoid agent."""
        self._goal = None
        mujoco_env.MujocoEnv.__init__(self, file_path, 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        """Get the observation of the humanoid."""
        data = self.sim.data
        return np.concatenate([data.qpos[:24].flat,
                               data.qvel[:23].flat,
                               data.cinert[:14].flat,
                               data.cvel[:14].flat,
                               data.qfrc_actuator[:23].flat,
                               data.cfrc_ext[:14].flat])

    def step(self, a):
        """Step the simulator forward in time."""
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        self.set_to_goal()
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        r = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        d = bool((data.qpos[2] < 0.4) or (data.qpos[2] > 2.0))
        return self._get_obs(), np.nan_to_num(r), d, dict(
            reward_linvel=lin_vel_cost,
            reward_quadctrl=-quad_ctrl_cost,
            reward_alive=alive_bonus,
            reward_impact=-quad_impact_cost)

    def reset_model(self):
        """Reset the humanoid to a starting location."""
        c = 0.01
        qpos = self.np_random.uniform(low=-c, high=c, size=self.model.nq)
        qvel = self.np_random.uniform(low=-c, high=c, size=self.model.nv)
        self.set_state(self.init_qpos + qpos, self.init_qvel + qvel)
        return self._get_obs()

    def viewer_setup(self):
        """Create the viewer."""
        self.update_viewer()

    def update_viewer(self):
        """Update the camera position and orientation."""
        if self.viewer is not None:
            x, y = self.get_xy()
            self.viewer.cam.azimuth = 0
            self.viewer.cam.distance = 35.
            self.viewer.cam.elevation = -90
            self.viewer.cam.lookat[0] = 4.
            self.viewer.cam.lookat[1] = 4.

    def get_ori(self):
        """Return the orientation of the agent."""
        rot = self.sim.data.qpos[3:7]
        ori = q_mult(q_mult(rot, [0, 1, 0, 0]), q_inv(rot))[1:3]
        return math.atan2(ori[1], ori[0])

    def set_xy(self, xy):
        """Set the x,y position of the agent."""
        qpos = np.copy(self.sim.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]
        self.set_state(qpos, self.sim.data.qvel)

    def set_goal(self, goal):
        """Set the goal position of the agent."""
        self._goal = goal

    def set_to_goal(self):
        """Update the goal position of the goal agent."""
        qpos = np.copy(self.sim.data.qpos)
        qvel = np.copy(self.sim.data.qvel)
        if self._goal is not None:
            qpos[24:24 + self._goal.size] = self._goal
        qpos[26] = 2.4
        qvel[23:46] = 0.0
        self.set_state(qpos, qvel)

    def get_xy(self):
        """Return the x,y position of the agent."""
        return self.sim.data.qpos[:2]
