from functools import reduce
import os.path as osp
import collections
import numpy as np
import gc

from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab import spaces
from rllab.misc.overrides import overrides
from rllab.misc import logger

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

BIG = 1e6


class MujocoEnv_ObsInit(MujocoEnv):
    """
    - add plot_visitation (possibly used by robots moving in 2D). Compatible with latents.
    - get_ori() base method, to implement in each robot
    - Cached observation_space at initialization to speed up training (x2)
    """

    def __init__(self,
                 visit_axis_bound=None,
                 *args, **kwargs):
        super(MujocoEnv_ObsInit, self).__init__(*args, **kwargs)

        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        self._observation_space = spaces.Box(ub * -1, ub)
        self.visit_axis_bound = visit_axis_bound

    @property
    @overrides
    def observation_space(self):
        return self._observation_space

    def get_ori(self):
        raise NotImplementedError

    def plot_visitations(self, paths, mesh_density=20, visit_prefix='', visit_axis_bound=None, maze=None, scaling=2):
        if 'env_infos' not in paths[0].keys() or 'com' not in paths[0]['env_infos'].keys():
            raise KeyError("No 'com' key in your path['env_infos']: please change you step function")
        fig, ax = plt.subplots()
        # now we will grid the space and check how much of it the policy is covering
        x_max = np.int(np.ceil(np.max(np.abs(np.concatenate([path["env_infos"]['com'][:, 0] for path in paths])))))
        y_max = np.int(np.ceil(np.max(np.abs(np.concatenate([path["env_infos"]['com'][:, 1] for path in paths])))))
        furthest = max(x_max, y_max)
        print('THE FUTHEST IT WENT COMPONENT-WISE IS: x_max={}, y_max={}'.format(x_max, y_max))
        if visit_axis_bound is None:
            visit_axis_bound = self.visit_axis_bound
        if visit_axis_bound and visit_axis_bound >= furthest:
            furthest = max(furthest, visit_axis_bound)
        # if maze:
        #     x_max = max(scaling * len(
        #         maze) / 2. - 1, x_max)  # maze enlarge plot to include the walls. ASSUME ROBOT STARTS IN CENTER!
        #     y_max = max(scaling * len(maze[0]) / 2. - 1, y_max)  # the max here should be useless...
        #     print("THE MAZE LIMITS ARE: x_max={}, y_max={}".format(x_max, y_max))
        delta = 1. / mesh_density
        y, x = np.mgrid[-furthest:furthest + delta:delta, -furthest:furthest + delta:delta]

        if 'agent_infos' in list(paths[0].keys()) and (('latents' in list(paths[0]['agent_infos'].keys())
                                                        and np.size(paths[0]['agent_infos']['latents'])) or
                                                           ('selectors' in list(paths[0]['agent_infos'].keys())
                                                            and np.size(paths[0]['agent_infos']['selectors']))):
            selectors_name = 'selectors' if 'selectors' in list(paths[0]['agent_infos'].keys()) else 'latents'
            dict_visit = collections.OrderedDict()  # keys: latents, values: np.array with number of visitations
            num_latents = np.size(paths[0]["agent_infos"][selectors_name][0])
            # set all the labels for the latents and initialize the entries of dict_visit
            for i in range(num_latents):  # use integer to define the latents
                dict_visit[i] = np.zeros((2 * furthest * mesh_density + 1, 2 * furthest * mesh_density + 1))

            # keep track of the overlap
            overlap = 0
            # now plot all the paths
            for path in paths:
                lats = [np.argmax(lat, axis=-1) for lat in path['agent_infos'][selectors_name]]  # list of all lats by idx
                com_x = np.ceil(((np.array(path['env_infos']['com'][:, 0]) + furthest) * mesh_density)).astype(int)
                com_y = np.ceil(((np.array(path['env_infos']['com'][:, 1]) + furthest) * mesh_density)).astype(int)
                coms = list(zip(com_x, com_y))
                for i, com in enumerate(coms):
                    dict_visit[lats[i]][com] += 1

            # fix the colors for each latent
            num_colors = num_latents + 2  # +2 for the 0 and Repetitions NOT COUNTING THE WALLS
            cmap = plt.get_cmap('nipy_spectral', num_colors)  # add one color for the walls
            # create a matrix with entries corresponding to the latent that was there (or other if several/wall/nothing)
            visitation_by_lat = np.zeros((2 * furthest * mesh_density + 1, 2 * furthest * mesh_density + 1))
            for i, visit in dict_visit.items():
                lat_visit = np.where(visit == 0, visit, i + 1)  # transform the map into 0 or i+1
                visitation_by_lat += lat_visit
                overlap += np.sum(np.where(visitation_by_lat > lat_visit))  # add the overlaps of this latent
                visitation_by_lat = np.where(visitation_by_lat <= i + 1, visitation_by_lat,
                                             num_colors - 1)  # mark overlaps
            # if maze:  # remember to also put a +1 for cmap!!
            #     for row in range(len(maze)):
            #         for col in range(len(maze[0])):
            #             if maze[row][col] == 1:
            #                 wall_min_x = max(0, (row - 0.5) * mesh_density * scaling)
            #                 wall_max_x = min(2 * furthest * mesh_density * scaling + 1,
            #                                  (row + 0.5) * mesh_density * scaling)
            #                 wall_min_y = max(0, (col - 0.5) * mesh_density * scaling)
            #                 wall_max_y = min(2 * furthest * mesh_density * scaling + 1,
            #                                  (col + 0.5) * mesh_density * scaling)
            #                 visitation_by_lat[wall_min_x: wall_max_x,
            #                 wall_min_y: wall_max_y] = num_colors
            #     gx_min, gfurthest, gy_min, gfurthest = self._find_goal_range()
            #     ax.add_patch(patches.Rectangle(
            #         (gx_min, gy_min),
            #         gfurthest - gx_min,
            #         gfurthest - gy_min,
            #         edgecolor='g', fill=False, linewidth=2,
            #     ))
            #     ax.annotate('G', xy=(0.5*(gx_min+gfurthest), 0.5*(gy_min+gfurthest)), color='g', fontsize=20)
            map_plot = ax.pcolormesh(x, y, visitation_by_lat, cmap=cmap, vmin=0.1,
                                     vmax=num_latents + 1)  # before 1 (will it affect when no walls?)
            color_len = (num_colors - 1.) / num_colors
            ticks = np.arange(color_len / 2., num_colors - 1, color_len)
            cbar = fig.colorbar(map_plot, ticks=ticks)
            latent_tick_labels = ['latent: ' + str(i) for i in list(dict_visit.keys())]
            cbar.ax.set_yticklabels(
                ['No visitation'] + latent_tick_labels + ['Repetitions'])  # horizontal colorbar
            # still log the total visitation
            visitation_all = reduce(np.add, [visit for visit in dict_visit.values()])
        else:
            visitation_all = np.zeros((2 * furthest * mesh_density + 1, 2 * furthest * mesh_density + 1))
            for path in paths:
                com_x = np.ceil(((np.array(path['env_infos']['com'][:, 0]) + furthest) * mesh_density)).astype(int)
                com_y = np.ceil(((np.array(path['env_infos']['com'][:, 1]) + furthest) * mesh_density)).astype(int)
                coms = list(zip(com_x, com_y))
                for com in coms:
                    visitation_all[com] += 1

            plt.pcolormesh(x, y, visitation_all, vmax=mesh_density)
            overlap = np.sum(np.where(visitation_all > 1, visitation_all, 0))  # sum of all visitations larger than 1
        ax.set_xlim([x[0][0], x[0][-1]])
        ax.set_ylim([y[0][0], y[-1][0]])

        log_dir = logger.get_snapshot_dir()
        exp_name = log_dir.split('/')[-1] if log_dir else '?'
        ax.set_title(visit_prefix + 'visitation: ' + exp_name)

        plt.savefig(osp.join(log_dir, visit_prefix + 'visitation.png'))  # this saves the current figure, here f
        plt.close()

        with logger.tabular_prefix(visit_prefix):
            total_visitation = np.count_nonzero(visitation_all)
            logger.record_tabular('VisitationTotal', total_visitation)
            logger.record_tabular('VisitationOverlap', overlap)

        ####
        # This was giving some problem with matplotlib and maximum number of colors
        ####
        # # now downsample the visitation
        # for down in [5, 10, 20]:
        #     visitation_down = np.zeros(tuple((i//down for i in visitation_all.shape)))
        #     delta_down = delta * down
        #     y_down, x_down = np.mgrid[-furthest:furthest+delta_down:delta_down, -furthest:furthest+delta_down:delta_down]
        #     for i, row in enumerate(visitation_down):
        #         for j, v in enumerate(row):
        #             visitation_down[i, j] = np.sum(visitation_all[down*i:down*(1+i), down*j:down*(j+1)])
        #     plt.figure()
        #     plt.pcolormesh(x_down, y_down, visitation_down, vmax=mesh_density)
        #     plt.title('Visitation_down')
        #     plt.xlim([x_down[0][0], x_down[0][-1]])
        #     plt.ylim([y_down[0][0], y_down[-1][0]])
        #     plt.title('visitation_down{}: {}'.format(down, exp_name))
        #     plt.savefig(osp.join(log_dir, 'visitation_down{}.png'.format(down)))
        #     plt.close()
        #
        #     total_visitation_down = np.count_nonzero(visitation_down)
        #     overlap_down = np.sum(np.where(visitation_down > 1, 1, 0))  # sum of all visitations larger than 1
        #     logger.record_tabular('VisitationTotal_down{}'.format(down), total_visitation_down)
        #     logger.record_tabular('VisitationOverlap_down{}'.format(down), overlap_down)

        plt.cla()
        plt.clf()
        plt.close('all')
        # del fig, ax, cmap, cbar, map_plot
        gc.collect()

