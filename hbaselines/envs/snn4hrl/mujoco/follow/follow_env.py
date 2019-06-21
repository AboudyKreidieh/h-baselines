import math
import numpy as np

from rllab import spaces
from rllab.envs.base import Step
from rllab.envs.mujoco.gather.gather_env import GatherEnv
from rllab.core.serializable import Serializable
from rllab.envs.mujoco.mujoco_env import BIG
from rllab.misc import logger
from rllab.misc.overrides import overrides

APPLE = 0
BOMB = 1


class FollowEnv(GatherEnv, Serializable):
    MODEL_CLASS = None
    ORI_IND = None

    def __init__(
            self,
            n_apples=1,
            n_bombs=0,
            displ_std=0.3,
            goal_vector_obs=False,
            goal_dist_rew=True,
            *args, **kwargs
    ):
        Serializable.quick_init(self, locals())
        self.displ_std = displ_std
        self.goal_vector_obs = goal_vector_obs
        self.goal_dist_rew = goal_dist_rew
        super(FollowEnv, self).__init__(n_apples=n_apples, n_bombs=n_bombs, *args, **kwargs)

    def step(self, action):
        _, inner_rew, done, info = self.wrapped_env.step(action)
        info['inner_rew'] = inner_rew
        info['dist_rew'] = 0
        info['outer_rew'] = 0
        if done:
            return Step(self.get_current_obs(), self.dying_cost, done, **info)  # give a -10 rew if the robot dies
        com = self.wrapped_env.get_body_com("torso")
        x, y = com[:2]
        reward = self.coef_inner_rew * inner_rew
        new_objs = []
        for obj in self.objects:
            ox, oy, typ = obj
            # object within zone!
            if (ox - x) ** 2 + (oy - y) ** 2 < self.catch_range ** 2:
                if typ == APPLE:
                    reward += 1
                    info['outer_rew'] = 1
                else:
                    reward -= 1
                    info['outer_rew'] = -1
            else:
                new_objs.append(obj)

        if self.goal_dist_rew:  # suppose there is only one object
            goal_vector = self.get_readings(vector_obs=True)
            if np.sum(goal_vector) > 0:  # only give the dist rew if you see the goal!
                ox, oy, typ = self.objects[0]
                dist_reward = self.goal_dist_rew * 1. / max((ox - x) ** 2 + (oy - y) ** 2, 1)
                info['dist_rew'] = dist_reward
                reward += dist_reward

        # move objects randomly
        self.objects = []
        for obj in new_objs:
            ox, oy, typ = obj
            ox_eps, oy_eps = np.random.normal(size=2) * self.displ_std
            if np.abs(ox + ox_eps) > self.activity_range:
                ox_eps = -ox_eps
            if np.abs(oy + oy_eps) > self.activity_range:
                oy_eps = -oy_eps
            self.objects.append((ox + ox_eps, oy + oy_eps, typ))

        # create another ball if it manages to take the previous one (so it doesn't just wait for the ball to come)
        if len(self.objects) == 0:
            self.reset(also_wrapped=False)
        return Step(self.get_current_obs(), reward, done, **info)

    def get_readings(self, vector_obs=None):  # equivalent to get_current_maze_obs in maze_env.py
        """
        :param vector_obs: the default behavior does NOT give the obs used for path, just the one the Viewer wants! So
        always call this function with vector_obs=self.goal_vector_obs
        """
        # compute sensor readings
        # first, obtain current orientation
        apple_readings = np.zeros(self.n_bins)
        bomb_readings = np.zeros(self.n_bins)
        # new reading
        vector_readings = np.zeros((self.n_apples + self.n_bombs, 3))
        robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
        # sort objects by distance to the robot, so that farther objects'
        # signals will be occluded by the closer ones'
        sorted_objects = sorted(
            self.objects, key=lambda o:
            (o[0] - robot_x) ** 2 + (o[1] - robot_y) ** 2)[::-1]
        # fill the readings
        bin_res = self.sensor_span / self.n_bins

        ori = self.get_ori()  # overwrite this for Ant!

        for i, obj in enumerate(sorted_objects):
            ox, oy, typ = obj
            vector_readings[i] = [0, 0, typ]  # if it's too far, I know there is a certain type, but not where
            # compute distance between object and robot
            dist = ((oy - robot_y) ** 2 + (ox - robot_x) ** 2) ** 0.5
            # only include readings for objects within range
            if dist > self.sensor_range:
                continue
            angle = math.atan2(oy - robot_y, ox - robot_x) - ori
            if math.isnan(angle):
                import ipdb
                ipdb.set_trace()
            angle = angle % (2 * math.pi)
            if angle > math.pi:
                angle = angle - 2 * math.pi
            if angle < -math.pi:
                angle = angle + 2 * math.pi
            # outside of sensor span - skip this
            half_span = self.sensor_span * 0.5
            if abs(angle) > half_span:
                continue
            bin_number = int((angle + half_span) / bin_res)
            #    ((angle + half_span) +
            #     ori) % (2 * math.pi) / bin_res)
            intensity = 1.0 - dist / self.sensor_range
            if typ == APPLE:
                apple_readings[bin_number] = intensity
            else:
                bomb_readings[bin_number] = intensity
            vector = np.array(ox - robot_x, oy - robot_y)
            if np.linalg.norm(vector):
                vector = vector / np.linalg.norm(vector) * intensity
            vector_readings[i, :2] = vector
        if vector_obs:
            return vector_readings  # this has to be a list of 1D arrays (or a 2D array) for unpacking.
        else:
            return apple_readings, bomb_readings  # in maze this is given already concatenated!! @@@@@@@

    def get_current_obs(self):
        # return sensor data along with data about itself
        self_obs = self.wrapped_env.get_current_obs()
        return np.concatenate([self_obs, *self.get_readings(vector_obs=self.goal_vector_obs)])

    @property
    @overrides
    def observation_space(self):
        dim = self.wrapped_env.observation_space.flat_dim
        if self.goal_vector_obs:
            newdim = dim + (self.n_apples + self.n_bombs) * 3
        else:
            newdim = dim + self.n_bins * 2
        ub = BIG * np.ones(newdim)
        return spaces.Box(ub * -1, ub)

    @property
    def maze_observation_space(self):
        shp = np.concatenate(self.get_readings(vector_obs=self.goal_vector_obs)).shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    @overrides
    def log_diagnostics(self, paths, *args, **kwargs):
        # we call here any logging related to the follow, strip the maze obs and call log_diag with the stripped paths
        # we need to log the purely follow reward!!
        with logger.tabular_prefix('Follow_'):
            follow_undiscounted_returns = [sum(path['env_infos']['outer_rew']) for path in paths]
            logger.record_tabular_misc_stat('Return', follow_undiscounted_returns, placement='front')
            dist_undiscounted_returns = [sum(path['env_infos']['dist_rew']) for path in paths]
            logger.record_tabular_misc_stat('DistReturn', dist_undiscounted_returns, placement='front')
        stripped_paths = []
        for path in paths:
            stripped_path = {}
            for k, v in path.items():
                stripped_path[k] = v
            stripped_path['observations'] = \
                stripped_path['observations'][:, :self.wrapped_env.observation_space.flat_dim]
            #  this breaks if the obs of the robot are d>1 dimensional (not a vector)
            stripped_paths.append(stripped_path)
        with logger.tabular_prefix('wrapped_'):
            if 'env_infos' in paths[0].keys() and 'inner_rew' in paths[0]['env_infos'].keys():
                wrapped_undiscounted_return = np.mean([np.sum(path['env_infos']['inner_rew']) for path in paths])
                logger.record_tabular('AverageReturn', wrapped_undiscounted_return)
            self.wrapped_env.log_diagnostics(stripped_paths)  # see swimmer_env.py for a scketch of the maze plotting!
