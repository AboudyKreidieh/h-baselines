import math
from contextlib import contextmanager

import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.mujoco.maze.maze_env import MazeEnv
from rllab.envs.mujoco.mujoco_env import BIG
from rllab.misc.overrides import overrides


class FastMazeEnv(MazeEnv, Serializable):
    """
    Changes the MazeEnv for speed. It has to be a maze defined with a grid (horizontal/vertical walls)
    - cache all the different observation spaces in the __init__
    - get_current_maze_obs now uses efficient intersection method for readings
    - The option
    """

    def __init__(
            self,
            *args,
            **kwargs):

        Serializable.quick_init(self, locals())
        MazeEnv.__init__(self, *args, **kwargs)
        self._blank_maze = False
        self.blank_maze_obs = np.concatenate([np.zeros(self._n_bins), np.zeros(self._n_bins)])

        # The following caches the spaces so they are not re-instantiated every time
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        self._observation_space = spaces.Box(ub * -1, ub)

        shp = self.get_current_robot_obs().shape
        ub = BIG * np.ones(shp)
        self._robot_observation_space = spaces.Box(ub * -1, ub)

        shp = self.get_current_maze_obs().shape
        ub = BIG * np.ones(shp)
        self._maze_observation_space = spaces.Box(ub * -1, ub)

    @overrides
    def get_current_maze_obs(self):
        # The observation would include both information about the robot itself as well as the sensors around its
        # environment
        structure = self.__class__.MAZE_STRUCTURE
        size_scaling = self.__class__.MAZE_SIZE_SCALING

        # compute origin cell i_o, j_o coordinates and center of it x_o, y_o (with 0,0 in the top-right corner of struc)
        o_xy = np.array(self._find_robot())  # this is self.init_torso_x, self.init_torso_y !!: center of the cell xy!
        o_ij = (o_xy / size_scaling).astype(int)  # this is the position in the grid (check if correct..)

        robot_xy = np.array(self.wrapped_env.get_body_com("torso")[:2])  # the coordinates of this are wrt the init!!
        ori = self.get_ori()  # for Ant this is computed with atan2, which gives [-pi, pi]

        c_ij = o_ij + np.rint(robot_xy / size_scaling)
        c_xy = (c_ij - o_ij) * size_scaling  # the xy position of the current cell center in init_robot origin

        R = int(self._sensor_range // size_scaling)

        wall_readings = np.zeros(self._n_bins)
        goal_readings = np.zeros(self._n_bins)

        for ray_idx in range(self._n_bins):
            ray_ori = ori - self._sensor_span * 0.5 + ray_idx / (
            self._n_bins - 1) * self._sensor_span  # make the ray in [-pi, pi]
            if ray_ori > math.pi:
                ray_ori -= 2 * math.pi
            elif ray_ori < - math.pi:
                ray_ori += 2 * math.pi
            x_dir, y_dir = 1, 1
            if math.pi / 2. <= ray_ori <= math.pi:
                x_dir = -1
            elif 0 > ray_ori >= - math.pi / 2.:
                y_dir = -1
            elif - math.pi / 2. > ray_ori >= - math.pi:
                x_dir, y_dir = -1, -1

            for r in range(R):
                next_x = c_xy[0] + x_dir * (0.5 + r) * size_scaling  # x of the next vertical segment, in init_rob coord
                next_i = int(c_ij[0] + x_dir * (r + 1))  # this is the i of the cells on the other side of the segment
                delta_y = (next_x - robot_xy[0]) * math.tan(ray_ori)
                y = robot_xy[1] + delta_y  # y of the intersection pt, wrt robot_init origin
                dist = np.sqrt(np.sum(np.square(robot_xy - (next_x, y))))
                if dist <= self._sensor_range and 0 <= next_i < len(structure[0]):
                    j = int(np.rint((y + o_xy[1]) / size_scaling))
                    if 0 <= j < len(structure):
                        if structure[j][next_i] == 1:
                            wall_readings[ray_idx] = (self._sensor_range - dist) / self._sensor_range
                            # plot_ray(wall_readings[ray_idx], ray_idx)
                            break
                        elif structure[j][next_i] == 'g':  # remember to flip the ij when referring to the matrix!!
                            goal_readings[ray_idx] = (self._sensor_range - dist) / self._sensor_range
                            # plot_ray(goal_readings[ray_idx], ray_idx, 'g')
                            break
                    else:
                        break
                else:
                    break

            # same for next horizontal segment. If the distance is less (higher intensity), update the goal/wall reading
            for r in range(R):
                next_y = c_xy[1] + y_dir * (0.5 + r) * size_scaling  # y of the next horizontal segment
                next_j = int(
                    c_ij[1] + y_dir * (r + 1))  # this is the i and j of the cells on the other side of the segment
                # first check the intersection with the next horizontal segment:
                delta_x = (next_y - robot_xy[1]) / math.tan(ray_ori)
                x = robot_xy[0] + delta_x
                dist = np.sqrt(np.sum(np.square(robot_xy - (x, next_y))))
                if dist <= self._sensor_range and 0 <= next_j < len(structure):
                    i = int(np.rint((x + o_xy[0]) / size_scaling))
                    if 0 <= i < len(structure[0]):
                        intensity = (self._sensor_range - dist) / self._sensor_range
                        if structure[next_j][i] == 1:
                            if wall_readings[ray_idx] == 0 or intensity > wall_readings[ray_idx]:
                                wall_readings[ray_idx] = intensity
                                # plot_ray(wall_readings[ray_idx], ray_idx)
                            break
                        elif structure[next_j][i] == 'g':
                            if goal_readings[ray_idx] == 0 or intensity > goal_readings[ray_idx]:
                                goal_readings[ray_idx] = intensity
                                # plot_ray(goal_readings[ray_idx], ray_idx, 'g')
                            break
                    else:
                        break
                else:
                    break

        # errase the goal readings behind a wall and the walls behind a goal:
        for n, wr in enumerate(wall_readings):
            if wr > goal_readings[n]:
                goal_readings[n] = 0
            elif wr <= goal_readings[n]:
                wall_readings[n] = 0

        obs = np.concatenate([
            wall_readings,
            goal_readings
        ])

        return obs

    @overrides
    def get_current_obs(self):
        if self._blank_maze:
            return np.concatenate([self.wrapped_env.get_current_obs(),
                                   self.blank_maze_obs
                                   ])
        else:
            return np.concatenate([self.wrapped_env.get_current_obs(),
                                   self.get_current_maze_obs()
                                   ])

    @contextmanager
    def blank_maze(self):
        previous_blank_maze_obs = self._blank_maze
        self._blank_maze = True
        yield
        self._blank_maze = previous_blank_maze_obs

    @property
    @overrides
    def observation_space(self):
        return self._observation_space

    @property
    @overrides
    def robot_observation_space(self):
        return self._robot_observation_space

    @property
    @overrides
    def maze_observation_space(self):
        return self._maze_observation_space

