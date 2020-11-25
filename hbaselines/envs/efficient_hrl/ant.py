# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Wrapper for creating the ant environment in gym_mujoco."""
import math
import numpy as np
from gym import utils
from copy import deepcopy
try:
    import mujoco_py
    from gym.envs.mujoco import mujoco_env
except ModuleNotFoundError:
    import gym
    mujoco_py = object()

    def mujoco_env():
        """Create a dummy environment for testing purposes."""
        return None
    setattr(mujoco_env, "MujocoEnv", gym.Env)


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


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Gym representation of the Ant MuJoCo environment."""

    ORI_IND = 3

    def __init__(self,
                 file_path=None,
                 expose_all_qpos=True,
                 expose_body_coms=None,
                 expose_body_comvels=None,
                 top_down_view=False,
                 ant_fall=False):
        """Instantiate the Ant environment.

        Parameters
        ----------
        file_path : str
            path to the xml file
        expose_all_qpos : bool
            whether to provide all qpos values via the observation
        expose_body_coms : list of str
            whether to provide all body_coms values via the observation
        expose_body_comvels : list of str
            whether to provide all body_comvels values via the observation
        top_down_view : bool, optional
            if set to True, the top-down view is provided via the observations
        ant_fall : bool
            specifies whether you are using the AntFall environment. The agent
            in this environment is placed on a block of height 4; the "dying"
            conditions for the agent need to be accordingly offset.
        """
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self._top_down_view = top_down_view
        self._ant_fall = ant_fall

        try:
            mujoco_env.MujocoEnv.__init__(self, file_path, 5)
        except TypeError:
            # for testing purposes
            pass
        utils.EzPickle.__init__(self)

    @property
    def physics(self):
        """Return the MuJoCo physics model."""
        # check mujoco version is greater than version 1.50 to call correct
        # physics model containing PyMjData object for getting and setting
        # position/velocity check https://github.com/openai/mujoco-py/issues/80
        # for updates to api
        if mujoco_py.get_version() >= '1.50':
            return self.sim
        else:
            return self.model

    def _step(self, a):
        """Advance the simulation by one step."""
        return self.step(a)

    def step(self, a):
        """Advance the simulation by one step."""
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .5 * np.square(a).sum()
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost + survive_reward
        state = self.state_vector()
        if self._ant_fall:
            notdone = np.isfinite(state).all() and 4.2 <= state[2] <= 5.0
        else:
            notdone = np.isfinite(state).all() and 0.2 <= state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        """Return the Ant observations."""
        # No cfrc observation
        if self._expose_all_qpos:
            obs = np.concatenate([
                self.physics.data.qpos.flat[:15],  # Ensures only ant obs.
                self.physics.data.qvel.flat[:14],
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat[:84],
            ])
        else:
            obs = np.concatenate([
                self.physics.data.qpos.flat[2:15],
                self.physics.data.qvel.flat[:14],
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat[:84],
            ])

        if self._expose_body_coms is not None:
            for name in self._expose_body_coms:
                com = self.get_body_com(name)
                if name not in self._body_com_indices:
                    indices = range(len(obs), len(obs) + len(com))
                    self._body_com_indices[name] = indices
                obs = np.concatenate([obs, com])

        if self._expose_body_comvels is not None:
            for name in self._expose_body_comvels:
                comvel = self.get_body_comvel(name)
                if name not in self._body_comvel_indices:
                    indices = range(len(obs), len(obs) + len(comvel))
                    self._body_comvel_indices[name] = indices
                obs = np.concatenate([obs, comvel])

        return obs

    def reset_model(self):
        """Reset the state of the agent to a particle original pos/vel."""
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1

        # Set everything other than ant to original position and 0 velocity.
        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        """Create the viewer."""
        if self._top_down_view:
            self.update_cam()
        else:
            self.viewer.cam.distance = self.model.stat.extent * 0.5
            self.viewer.cam.distance = 55
            self.viewer.cam.elevation = -90
            self.viewer.cam.lookat[0] = 8
            self.viewer.cam.lookat[1] = 8

    def update_cam(self):
        """Update the position of the camera."""
        if self.viewer is not None:
            x, y = self.get_xy()
            self.viewer.cam.azimuth = 0
            self.viewer.cam.distance = 15.
            self.viewer.cam.elevation = -90
            self.viewer.cam.lookat[0] = x
            self.viewer.cam.lookat[1] = y

    def get_ori(self):
        """Return the orientation of the agent."""
        ori = [0, 1, 0, 0]
        # take the quaternion
        rot = self.physics.data.qpos[
              self.__class__.ORI_IND:self.__class__.ORI_IND + 4]
        # project onto x-y plane
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]
        ori = math.atan2(ori[1], ori[0])
        return ori

    def set_xy(self, xy):
        """Set the x,y position of the agent."""
        qpos = np.copy(self.physics.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]

        qvel = self.physics.data.qvel
        self.set_state(qpos, qvel)

    def get_xy(self):
        """Return the x,y position of the agent."""
        return self.physics.data.qpos[:2]

    def set_goal(self, goal):
        """Set the goal position of the agent.

        Parameters
        ----------
        goal : array_like
            the desired position of the agent
        """
        goal = deepcopy(goal)
        self.physics.data.qpos.flat[15:] = goal
        self.physics.data.qpos.flat[17] = 8
