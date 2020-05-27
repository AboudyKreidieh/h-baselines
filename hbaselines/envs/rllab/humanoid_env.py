import gym
import gym.spaces
import math
import numpy as np
import mujoco_py
import random
import os
import tempfile
import xml.etree.ElementTree as ET
import cv2
from gym import utils
from gym.envs.mujoco import mujoco_env

# Directory that contains mujoco xml files.
SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(SCRIPT_PATH, 'assets')


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    FILE = 'humanoid.xml'

    def __init__(
            self,
            horizon=1000):

        file_path = os.path.join(MODEL_DIR, HumanoidEnv.FILE)
        mujoco_env.MujocoEnv.__init__(self, file_path, 5)
        utils.EzPickle.__init__(self)

        self.horizon = horizon

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat,
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def step(self, a):
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
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return self._get_obs(), r, done, dict(
            reward_linvel=lin_vel_cost,
            reward_quadctrl=-quad_ctrl_cost,
            reward_alive=alive_bonus,
            reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        rand_qpos = self.np_random.uniform(low=-c, high=c, size=self.model.nq)
        rand_qvel = self.np_random.uniform(low=-c, high=c, size=self.model.nv)
        qpos = self.init_qpos + rand_qpos
        qvel = self.init_qvel + rand_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
