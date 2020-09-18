"""Implements a ant which is sparsely rewarded for reaching a goal"""
import os
import numpy as np
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.envs.base import Step

import hbaselines.config as config

MODELS_PATH = os.path.abspath(os.path.join(
    config.PROJECT_PATH, 'hbaselines/envs/composition/mujoco_models'))


class HalfCheetahHurdleEnv(HalfCheetahEnv):

    def __init__(self):
        self.step_number = 0
        self.block_passed = 0

        path = os.path.join(MODELS_PATH, 'half_cheetah_hurdle.xml')
        MujocoEnv.__init__(self, file_path=path)

    def get_current_obs(self):
        proprioceptive_observation = super().get_current_obs()
        pos = self.get_body_com('bfoot')[0]
        next_hurdle_pos = [2 + 3 * self.block_passed]
        bf_dist_frm_next_hurdle = [np.linalg.norm(next_hurdle_pos[0] - pos)]
        observation = np.concatenate([
            proprioceptive_observation,
            next_hurdle_pos,
            bf_dist_frm_next_hurdle
        ]).reshape(-1)

        return observation

    def step(self, action):
        self.step_number += 1

        # Update the model.
        self.forward_dynamics(action)

        # Compute the next observation.
        next_obs = self.get_current_obs()
        x = self.get_body_com('bfoot')[0]

        # Check if a new forward block was surpassed.
        if x > 2 + 3 * self.block_passed:
            self.block_passed += 1
            reward = 1
        # Check if the agent moved back a block.
        elif x < -1 + 3 * self.block_passed:
            self.block_passed += -1
            reward = -1
        # Rewards of zero otherwise.
        else:
            reward = 0

        # Some logging.
        if reward != 0:
            print(reward)

        # Compute the done mask based on whether the time horizon was met.
        done = self.step_number >= self.horizon
        if done:
            self.step_number = 0
            self.block_passed = 0

        return Step(next_obs, reward, done, **{})

    @property
    def horizon(self):
        return 500
