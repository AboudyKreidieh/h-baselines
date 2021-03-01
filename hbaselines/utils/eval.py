"""Utility methods when performing evaluations."""
import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from hbaselines.fcnet.td3 import FeedForwardPolicy \
    as TD3FeedForwardPolicy
from hbaselines.fcnet.sac import FeedForwardPolicy \
    as SACFeedForwardPolicy
from hbaselines.fcnet.ppo import FeedForwardPolicy \
    as PPOFeedForwardPolicy
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy \
    as TD3GoalConditionedPolicy
from hbaselines.goal_conditioned.sac import GoalConditionedPolicy \
    as SACGoalConditionedPolicy
from hbaselines.multiagent.td3 import MultiFeedForwardPolicy \
    as TD3MultiFeedForwardPolicy
from hbaselines.multiagent.sac import MultiFeedForwardPolicy \
    as SACMultiFeedForwardPolicy
from hbaselines.multiagent.ppo import MultiFeedForwardPolicy \
    as PPOMultiFeedForwardPolicy
from hbaselines.multiagent.h_td3 import MultiGoalConditionedPolicy \
    as TD3MultiGoalConditionedPolicy
from hbaselines.multiagent.h_sac import MultiGoalConditionedPolicy \
    as SACMultiGoalConditionedPolicy

# offset used to positions when drawing trajectories
OBJECT_OFFSET = 1

# dictionary that maps policy names to policy objects
POLICY_DICT = {
    "FeedForwardPolicy": {
        "TD3": TD3FeedForwardPolicy,
        "SAC": SACFeedForwardPolicy,
        "PPO": PPOFeedForwardPolicy,
    },
    "GoalConditionedPolicy": {
        "TD3": TD3GoalConditionedPolicy,
        "SAC": SACGoalConditionedPolicy,
    },
    "MultiFeedForwardPolicy": {
        "TD3": TD3MultiFeedForwardPolicy,
        "SAC": SACMultiFeedForwardPolicy,
        "PPO": PPOMultiFeedForwardPolicy,
    },
    "MultiGoalConditionedPolicy": {
        "TD3": TD3MultiGoalConditionedPolicy,
        "SAC": SACMultiGoalConditionedPolicy,
    },
}


def parse_options(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        description='Run evaluation episodes of a given checkpoint.',
        epilog='python run_eval "/path/to/dir_name" ckpt_num')

    # required input parameters
    parser.add_argument(
        'dir_name', type=str, help='the path to the checkpoints folder')

    # optional arguments
    parser.add_argument(
        '--ckpt_num', type=int, default=None,
        help='the checkpoint number. If not specified, the last checkpoint is '
             'used.')
    parser.add_argument(
        '--num_rollouts', type=int, default=1,
        help='number of eval episodes')
    parser.add_argument(
        '--video', type=str, default='output',
        help='path to the video to render')
    parser.add_argument(
        '--save_video', action='store_true',
        help='whether to save the rendering')
    parser.add_argument(
        '--save_trajectory', action='store_true',
        help='whether to save the per-step trajectory data')
    parser.add_argument(
        '--no_render', action='store_true',
        help='shuts off rendering')
    parser.add_argument(
        '--random_seed', action='store_true',
        help='whether to run the simulation on a random seed. If not added, '
             'the original seed is used.')

    return parser.parse_args(args)


def get_hyperparameters_from_dir(ckpt_path):
    """Collect the algorithm-specific hyperparameters from the checkpoint.

    Parameters
    ----------
    ckpt_path : str
        the path to the checkpoints folder

    Returns
    -------
    str
        environment name
    hbaselines.goal_conditioned.*
        policy object
    dict
        algorithm and policy hyperparaemters
    int
        the seed value
    """
    # import the dictionary of hyperparameters
    with open(os.path.join(ckpt_path, 'hyperparameters.json'), 'r') as f:
        hp = json.load(f)

    # collect the policy object
    policy_name = hp['policy_name']
    alg_name = hp['algorithm']
    policy = POLICY_DICT[policy_name][alg_name]

    # collect the environment name
    env_name = hp['env_name']

    # collect the seed value
    seed = hp['seed']

    # remove unnecessary features from hp dict
    hp = hp.copy()
    del hp['policy_name'], hp['env_name'], hp['seed']
    del hp['algorithm'], hp['date/time']

    return env_name, policy, hp, seed


class TrajectoryLogger(object):
    """Logger object for evaluation trajectory data.

    This method logs, save, and plots trajectory data during evaluations for a
    number of tasks.

    Attributes
    ----------
    env_name : str
        the name of the environment
    data : dict
        a dictionary containing the data to log/visualize
    """

    def __init__(self, env_name):
        """Instantiate the logger object.

        Parameters
        ----------
        env_name : str
            the name of the environment
        """
        self.env_name = env_name
        self.data = {}

    def reset(self, env):
        """Reset the stored sample data-points."""
        if self.env_name == "AntGather":
            # Save the initial position of the objects (apples and bombs).
            self.data = {
                "objects": env.objects,
                "obs": [],
                "goal": [],
            }
        elif self.env_name in ["AntMaze", "AntFourRooms"]:
            # Save the goal position.
            self.data = {
                "context": list(env.current_context),
                "obs": [],
                "goal": [],
            }
        elif self.env_name.startswith("ring-v"):  # ring-v{0,1,2,3,4}{-fast}
            # Save the speeds and goal speeds.
            self.data = {
                "speed": [],
                "goal": [],
            }
        else:
            raise NotImplementedError("Unknown environment: {}".format(
                self.env_name))

    def log_sample(self, obs, policy):
        """Update the dataset with current step data.

        Parameters
        ----------
        obs : array_like
            the current observation
        policy : hbaselines.base_policies.Policy
            the policy object
        """
        if self.env_name in ["AntGather", "AntMaze", "AntFourRooms"]:
            # Log the agent position.
            self.data["obs"].append(obs[:2])

            # Log the goals.
            if hasattr(policy, "meta_action"):
                goal = np.array([
                    policy.meta_action[0][i] +
                    (obs[policy.goal_indices]
                     if policy.relative_goals else 0)
                    for i in range(policy.num_levels - 1)
                ])
                self.data["goal"].append(goal.flatten()[:2])

        elif self.env_name.startswith("ring-v"):  # ring-v{0,1,2,3,4}{-fast}
            # Log the vehicle speeds and desired speeds.
            pass
        else:
            raise NotImplementedError("Unknown environment: {}".format(
                self.env_name))

    def save(self, fp, plot=False):
        """Save, and potential plot, the trajectory data.

        Parameters
        ----------
        fp : str
            the path to save the data to
        plot : bool
            Whether to plot the data
        """
        # Save the data dict under a json file.
        with open(fp, 'w') as f:
            json.dump(self.data, f, sort_keys=True, indent=4)

        # Plot trajectories.
        if plot:
            if self.env_name == "AntGather":
                draw_antgather(
                    objects=self.data["objects"],
                    traj=self.data["obs"],
                    goals=self.data["goal"],
                    save_path=fp,
                )
            elif self.env_name == "AntMaze":
                draw_antmaze(
                    context=self.data["context"],
                    traj=self.data["obs"],
                    goals=self.data["goal"],
                    save_path=fp,
                )
            elif self.env_name == "AntFourRooms":
                draw_antfourrooms(
                    context=self.data["context"],
                    traj=self.data["obs"],
                    goals=self.data["goal"],
                    save_path=fp,
                )
            else:
                raise NotImplementedError("Unknown environment: {}".format(
                    self.env_name))


def draw_antgather(objects, traj, goals, save_path):
    """Draw the trajectory when using the AntGather environment.

    Parameters
    ----------
    objects : list of (float, float, int)
        list of apples/bombs data-points, described as follows:
          1. the x-coordinate
          2. the y-coordinate
          3. 0 if apple, 1 if bomb
    traj : array_like
        the (x,y) coordinates of the agent at every step
    goals : array_like
        the desired (x,y) coordinates of the agent at every step
    save_path : str
        the path to save the plot to
    """
    traj = np.asarray(traj)

    # Create figure and axes.
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # Draw the boundaries.
    rect = patches.Rectangle(
        (-11, -11), 1, 22,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (-11, -11), 22, 1,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (-11, 10), 22, 1,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (10, -11), 1, 22,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (-10, -10), 20, 20,
        linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (-11, -11), 22, 22,
        linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect)

    # Draw the apples and bombs.
    for obj in objects:
        ox, oy, typ = obj
        obj_touched = any(
            np.sqrt((traj[:, 0] - ox) ** 2 + (traj[:, 1] - oy) ** 2) < 1)
        circle = plt.Circle(
            (ox + OBJECT_OFFSET, oy + OBJECT_OFFSET),
            0.8, color='limegreen' if typ == 0 else 'red',
            alpha=0.25 if obj_touched else 1,
        )
        ax.add_artist(circle)

    for obj in goals:
        ox, oy = obj
        ox = max(-10 - OBJECT_OFFSET, min(10 - OBJECT_OFFSET, ox))
        oy = max(-10 - OBJECT_OFFSET, min(10 - OBJECT_OFFSET, oy))
        circle = plt.Circle(
            (ox + OBJECT_OFFSET, oy + OBJECT_OFFSET),
            0.4, color='blue', alpha=0.05)
        ax.add_artist(circle)

    # Draw the trajectory of the agent.
    plt.plot(traj[:, 0] + OBJECT_OFFSET,
             traj[:, 1] + OBJECT_OFFSET, '--', lw=1.5, c='k')

    # Plot position if agent died.
    # if len(traj) < 500:
    #     mymarker = plt.scatter(
    #         traj[-1][0] + OBJECT_OFFSET,
    #         traj[-1][1] + OBJECT_OFFSET, c='k', marker='x')
    #     ax.add_artist(mymarker)

    plt.xlim([-11.2, 11.2])
    plt.ylim([-11.2, 11.2])

    plt.axis('off')

    plt.savefig("{}.pdf".format(save_path),
                bbox_inches='tight', transparent=True)


def draw_antfourrooms(context, traj, goals, save_path):
    """Draw the trajectory when using the AntFourRooms environment.

    Parameters
    ----------
    context : [float, float]
        the (x,y) coordinates of the overall goal for the agent
    traj : array_like
        the (x,y) coordinates of the agent at every step
    goals : array_like
        the desired (x,y) coordinates of the agent at every step
    save_path : str
        the path to save the plot to
    """
    traj = np.asarray(traj)

    # Create figure and axes.
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # Draw the boundaries.
    rect = patches.Rectangle(
        (-3, -3), 2, 26,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (-3, -3), 26, 2,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (-3, 21), 26, 2,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (21, -3), 2, 26,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)

    rect = patches.Rectangle(
        (-1, -1), 22, 22,
        linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (-3, -3), 26, 26,
        linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect)

    rect = patches.Rectangle(
        (-1.2, 9), 2.2, 2,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    plt.plot([-1, 1], [9, 9], lw=1, c='k')
    plt.plot([-1, 1], [11, 11], lw=1, c='k')
    plt.plot([1, 1], [9, 11], lw=1, c='k')

    rect = patches.Rectangle(
        (9, 5), 2, 10,
        linewidth=1, edgecolor='k', facecolor='lightgrey')
    ax.add_patch(rect)

    rect = patches.Rectangle(
        (5, 9), 5.8, 2,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    plt.plot([5, 9], [9, 9], lw=1, c='k')
    plt.plot([5, 9], [11, 11], lw=1, c='k')
    plt.plot([5, 5], [9, 11], lw=1, c='k')

    rect = patches.Rectangle(
        (9, 19), 2, 2.2,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    plt.plot([9, 9], [19, 21], lw=1, c='k')
    plt.plot([11, 11], [19, 21], lw=1, c='k')
    plt.plot([9, 11], [19, 19], lw=1, c='k')

    rect = patches.Rectangle(
        (9, -1.2), 2, 2.2,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    plt.plot([9, 9], [-1, 1], lw=1, c='k')
    plt.plot([11, 11], [-1, 1], lw=1, c='k')
    plt.plot([9, 11], [1, 1], lw=1, c='k')

    rect = patches.Rectangle(
        (10.8, 11), 4.2, 2,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    plt.plot([11, 15], [11, 11], lw=1, c='k')
    plt.plot([11, 15], [13, 13], lw=1, c='k')
    plt.plot([15, 15], [11, 13], lw=1, c='k')

    rect = patches.Rectangle(
        (19, 11), 2.2, 2,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    plt.plot([19, 21], [11, 11], lw=1, c='k')
    plt.plot([19, 21], [13, 13], lw=1, c='k')
    plt.plot([19, 19], [11, 13], lw=1, c='k')

    # Draw goal
    circle = plt.Circle(
        tuple(context),
        2.5, color='limegreen', alpha=0.5
    )
    ax.add_artist(circle)

    for obj in goals:
        ox, oy = obj
        ox = max(-1 - OBJECT_OFFSET, min(21 - OBJECT_OFFSET, ox))
        oy = max(-1 - OBJECT_OFFSET, min(21 - OBJECT_OFFSET, oy))
        circle = plt.Circle(
            (ox+OBJECT_OFFSET, oy+OBJECT_OFFSET),
            0.4, color='blue', alpha=0.05)
        ax.add_artist(circle)

    # Draw the trajectory of the agent.
    plt.plot(traj[:, 0] + OBJECT_OFFSET,
             traj[:, 1] + OBJECT_OFFSET, '--', lw=1.5, c='k')

    # Plot position if agent died.
    # if len(traj) < 500:
    #     mymarker = plt.scatter(
    #         traj[-1][0] + OBJECT_OFFSET,
    #         traj[-1][1] + OBJECT_OFFSET, c='k', marker='x')
    #     ax.add_artist(mymarker)

    plt.xlim([-3.2, 23.2])
    plt.ylim([-3.2, 23.2])

    plt.axis('off')

    plt.savefig("{}.pdf".format(save_path),
                bbox_inches='tight', transparent=True)

    plt.show()


def draw_antmaze(context, traj, goals, save_path):
    """Draw the trajectory when using the AntMaze environment.

    Parameters
    ----------
    context : [float, float]
        the (x,y) coordinates of the overall goal for the agent
    traj : array_like
        the (x,y) coordinates of the agent at every step
    goals : array_like
        the desired (x,y) coordinates of the agent at every step
    save_path : str
        the path to save the plot to
    """
    traj = np.asarray(traj)

    # Create figure and axes.
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # Draw the boundaries.
    rect = patches.Rectangle(
        (-6, -6), 2, 28,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (-6, -6), 28, 2,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (-6, 20), 28, 2,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (20, -6), 2, 28,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)

    rect = patches.Rectangle(
        (-4, -4), 24, 24,
        linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (-6, -6), 28, 28,
        linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect)

    rect = patches.Rectangle(
        (-5, 4), 17, 8,
        linewidth=1, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    plt.plot([-4, 12], [4, 4], lw=1, c='k')
    plt.plot([-4, 12], [12, 12], lw=1, c='k')
    plt.plot([12, 12], [4, 12], lw=1, c='k')

    # Draw goal
    circle = plt.Circle(
        context,
        3.5, color='limegreen', alpha=0.5
    )
    ax.add_artist(circle)

    for obj in goals:
        ox, oy = obj
        ox = max(-4 - OBJECT_OFFSET, min(20 - OBJECT_OFFSET, ox))
        oy = max(-4 - OBJECT_OFFSET, min(20 - OBJECT_OFFSET, oy))
        circle = plt.Circle(
            (ox + OBJECT_OFFSET, oy + OBJECT_OFFSET),
            0.4, color='blue', alpha=0.05)
        ax.add_artist(circle)

    # Draw the trajectory of the agent.
    plt.plot(traj[:, 0] + OBJECT_OFFSET,
             traj[:, 1] + OBJECT_OFFSET, '--', lw=1.5, c='k')

    # Plot position if agent died.
    # if len(traj) < 500:
    #     mymarker = plt.scatter(
    #         traj[-1][0] + OBJECT_OFFSET,
    #         traj[-1][1] + OBJECT_OFFSET, c='k', marker='x')
    #     ax.add_artist(mymarker)

    plt.xlim([-6.2, 22.2])
    plt.ylim([-6.2, 22.2])

    plt.axis('off')

    plt.savefig("{}.pdf".format(save_path),
                bbox_inches='tight', transparent=True)

    plt.show()
