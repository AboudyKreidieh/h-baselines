"""Script containing the UR5 and Pendulum environments."""
import numpy as np
import gym
from gym.spaces import Box
import os
from hbaselines.envs.hac.env_utils import check_validity
from hbaselines.utils.reward_fns import negative_distance

try:
    import mujoco_py
except ImportError:
    # for testing purposes
    import hbaselines.envs.hac.dummy_mujoco as mujoco_py


class Environment(gym.Env):
    """Base environment class.

    Supports the UR5 and Pendulum environments from:

    Levy, Andrew, et al. "Learning Multi-Level Hierarchies with Hindsight."
    (2018).

    Attributes
    ----------
    name : str
        name of the environment; adopted from the name of the model
    model : object
        the imported MuJoCo model
    sim : mujoco_py.MjSim
        the MuJoCo simulator object, used to interact with and advance the
        simulation
    end_goal_thresholds : array_like
        goal achievement thresholds. If the agent is within the threshold for
        each dimension, the end goal has been achieved and the reward of 0 is
        granted.
    initial_state_space : list of (float, float)
        bounds for the initial values for all elements in the state space.
        This is achieved during the reset procedure.
    max_actions : int
        maximum number of atomic actions. This will typically be
        flags.time_scale**(flags.layers).
    visualize : bool
        specifies whether to render the environment
    viewer : mujoco_py.MjViewer
        a display GUI showing the scene of an MjSim object
    num_frames_skip : int
        number of time steps per atomic action
    num_steps : int
        number of steps since the start of the current rollout
    """

    def __init__(self,
                 model_name,
                 project_state_to_end_goal,
                 end_goal_thresholds,
                 initial_state_space,
                 contextual_reward,
                 use_contexts=False,
                 random_contexts=False,
                 context_range=None,
                 max_actions=1200,
                 num_frames_skip=1,
                 show=False):
        """Instantiate the Environment object.

        Parameters
        ----------
        model_name : str
            name of the xml file in './mujoco_files/' that the model is
            generated from

        end_goal_thresholds : array_like
            goal achievement thresholds. If the agent is within the threshold
            for each dimension, the end goal has been achieved and the reward
            of 0 is granted.
        initial_state_space : list of (float, float)
            bounds for the initial values for all elements in the state space.
            This is achieved during the reset procedure.
        max_actions : int, optional
            maximum number of atomic actions. Defaults to 1200.
        num_frames_skip : int, optional
            number of time steps per atomic action. Defaults to 10.
        show : bool, optional
            specifies whether to render the environment. Defaults to False.
        """
        # Ensure environment customization have been properly entered.
        check_validity(model_name, initial_state_space, max_actions,
                       num_frames_skip)

        self.name = model_name
        self.project_state_to_end_goal = project_state_to_end_goal
        self.end_goal_thresholds = end_goal_thresholds
        self.initial_state_space = initial_state_space
        self.max_actions = max_actions

        # Create Mujoco Simulation
        mujoco_file_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'assets'))
        self.model = mujoco_py.load_model_from_path(
            os.path.join(mujoco_file_path, model_name))
        self.sim = mujoco_py.MjSim(self.model)

        # contextual variables
        self.use_contexts = use_contexts
        self.random_contexts = random_contexts
        self.context_range = context_range
        self.contextual_reward = contextual_reward
        self.current_context = None

        # Implement visualization if necessary
        self.visualize = show  # Visualization boolean
        if self.visualize:
            self.viewer = mujoco_py.MjViewer(self.sim)  # pragma: no cover
        else:
            self.viewer = None
        self.num_frames_skip = num_frames_skip

        self.num_steps = 0

    def get_state(self):
        """Get state, which concatenates joint positions and velocities."""
        raise NotImplementedError

    def reset(self):
        """Reset simulation to state within initial state specified by user.

        Returns
        -------
        array_like
            the initial observation
        """
        # Reset the time counter.
        self.num_steps = 0

        # Reset joint positions and velocities
        for i in range(len(self.sim.data.qpos)):
            self.sim.data.qpos[i] = np.random.uniform(
                self.initial_state_space[i][0], self.initial_state_space[i][1])

        for i in range(len(self.sim.data.qvel)):
            self.sim.data.qvel[i] = np.random.uniform(
                self.initial_state_space[len(self.sim.data.qpos) + i][0],
                self.initial_state_space[len(self.sim.data.qpos) + i][1])

        # Update the goal.
        if self.use_contexts:
            self.current_context = self.get_next_goal()

        # Return state
        return self.get_state()

    def step(self, action):
        """Advance the simulation by one step.

        This method executes the low-level action. This is done for number of
        frames specified by num_frames_skip.

        Parameters
        ----------
        action : array_like
            the low level primitive action

        Returns
        -------
        array_like
            the next observation
        float
            reward
        bool
            done mask
        dict
            extra info (set to an empty dictionary by default)
        """
        # Perform the requested action for a given number of steps.
        self.sim.data.ctrl[:] = action
        for _ in range(self.num_frames_skip):
            self.sim.step()
            self.num_steps += 1
            if self.visualize:
                self.render()  # pragma: no cover

        obs = self.get_state()

        # check whether the goal is reached.
        is_success = all(
            np.absolute(self.project_state_to_end_goal(self.sim, obs)
                        - self.current_context)
            < self.end_goal_thresholds)

        # Reward of 0 when the goal is reached, and -1 otherwise.
        reward = self.contextual_reward(obs, self.current_context, obs)

        # If the time horizon is met, set done to True.
        done = self.num_steps >= self.max_actions or is_success

        # Success is defined as getting within a distance threshold from the
        # target.
        info_dict = {'is_success': is_success}

        return obs, reward, done, info_dict

    def display_end_goal(self, end_goal):
        """Visualize end goal.

        The goal can be visualized by changing the location of the relevant
        site object.

        Parameters
        ----------
        end_goal : array_like
            the desired end goals to be displayed
        """
        raise NotImplementedError

    def get_next_goal(self):
        """Return an end goal.

        Returns
        -------
        array_like
            the end goal
        """
        raise NotImplementedError

    def render(self, mode='human'):
        """Render the environment."""
        self.viewer.render()  # pragma: no cover

    @property
    def horizon(self):
        """Return the environment horizon."""
        return self.max_actions

    @property
    def context_space(self):
        """Return the shape and bounds of the contextual term."""
        # Check if the environment is using contexts, and if not, return a None
        # value as the context space.
        if self.use_contexts:
            # If the context space is random, use the min and max values of
            # each context to specify the space range. Otherwise, the min and
            # max values are both the deterministic context value.
            if self.random_contexts:
                context_low = []
                context_high = []
                for context_i in self.context_range:
                    low, high = context_i
                    context_low.append(low)
                    context_high.append(high)
                return Box(low=np.asarray(context_low),
                           high=np.asarray(context_high),
                           dtype=np.float32)
            else:
                return Box(low=np.asarray(self.context_range),
                           high=np.asarray(self.context_range),
                           dtype=np.float32)
        else:
            return None


class UR5(Environment):
    """UR5 environment class.

    In this environment, a UR5 reacher object is tasked with reaching an end
    goal consisting of the desired joint positions for the 3 main joints.
    """

    def __init__(self,
                 use_contexts=False,
                 random_contexts=False,
                 context_range=None,
                 show=False):
        """Initialize the UR5 environment.

        Parameters
        ----------
        use_contexts : bool, optional
            specifies whether to add contexts to the observations and add the
            contextual rewards
        random_contexts : bool
            specifies whether the context is a single value, or a random set of
            values between some range
        context_range : list of float or list of (float, float)
            the desired context / goal, or the (lower, upper) bound tuple for
            each dimension of the goal
        show : bool
            specifies whether to render the environment

        Raises
        ------
        AssertionError
            If the context_range is not the right form based on whether
            contexts are a single value or random across a range.
        """
        # max number of atomic actions
        max_actions = 600

        # number of time steps per atomic action
        timesteps_per_action = 1  # 15

        # file name of Mujoco model. This file is stored in "assets" folder
        model_name = "ur5.xml"

        # initial state space consisting of the ranges for all joint angles and
        # velocities. In the UR5 Reacher task, we use a random initial shoulder
        # position and use fixed values for the remainder. Initial joint
        # velocities are set to 0.
        initial_joint_pos = [(-np.pi / 8, np.pi / 8),
                             (3.22757851e-03, 3.22757851e-03),
                             (-1.27944547e-01, -1.27944547e-01)]
        initial_joint_speed = [(0, 0) for _ in range(len(initial_joint_pos))]
        initial_state_space = initial_joint_pos + initial_joint_speed

        # Supplementary function that will ensure all angles are between
        # [-2*np.pi,2*np.pi]
        def bound_angle(angle):
            bounded_angle = np.absolute(angle) % (2 * np.pi)
            if angle < 0:
                bounded_angle = -bounded_angle
            return bounded_angle

        # function that maps from the state space to the end goal space
        def project_state_to_end_goal(sim, *_):
            return np.array([bound_angle(sim.data.qpos[i])
                             for i in range(len(sim.data.qpos))])

        # end goal achievement thresholds. If the agent is within the threshold
        # for each dimension, the end goal has been achieved.
        angle_threshold = np.deg2rad(10)
        end_goal_thresholds = np.array([angle_threshold for _ in range(3)])

        def contextual_reward(states, goals, next_states):
            return negative_distance(
                states=states,
                goals=goals,
                next_states=next_states,
                state_indices=[0, 1, 2],
                relative_context=False,
                offset=0.0,
                reward_scales=1.0
            )

        super(UR5, self).__init__(
            model_name=model_name,
            project_state_to_end_goal=project_state_to_end_goal,
            end_goal_thresholds=end_goal_thresholds,
            initial_state_space=initial_state_space,
            max_actions=max_actions,
            num_frames_skip=timesteps_per_action,
            show=show,
            contextual_reward=contextual_reward,
            use_contexts=use_contexts,
            random_contexts=random_contexts,
            context_range=context_range,
        )

    @property
    def observation_space(self):
        """Return the observation space."""
        return gym.spaces.Box(
            low=-1, high=1,  # TODO: bounds?
            shape=(len(self.sim.data.qpos) + len(self.sim.data.qvel),),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        """Return the action space."""
        return gym.spaces.Box(
            low=-self.sim.model.actuator_ctrlrange[:, 1],
            high=self.sim.model.actuator_ctrlrange[:, 1],
            dtype=np.float32,
        )

    def get_state(self):
        """See parent class."""
        return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))

    def get_next_goal(self):
        """See parent class."""
        end_goal = np.zeros(shape=(len(self.context_range),))
        goal_possible = False

        while not goal_possible:
            end_goal = np.zeros(shape=(len(self.context_range),))

            end_goal[0] = np.random.uniform(self.context_range[0][0],
                                            self.context_range[0][1])
            end_goal[1] = np.random.uniform(self.context_range[1][0],
                                            self.context_range[1][1])
            end_goal[2] = np.random.uniform(self.context_range[2][0],
                                            self.context_range[2][1])

            # Next need to ensure chosen joint angles result in achievable
            # task (i.e., desired end effector position is above ground)
            theta_1 = end_goal[0]
            theta_2 = end_goal[1]
            theta_3 = end_goal[2]

            # shoulder_pos_1 = np.array([0, 0, 0, 1])
            # upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
            forearm_pos_3 = np.array([0.425, 0, 0, 1])
            wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

            # Transformation matrix from shoulder to base reference frame
            t_1_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                              [0, 0, 1, 0.089159], [0, 0, 0, 1]])

            # Transformation matrix from upper arm to shoulder reference frame
            t_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],
                              [np.sin(theta_1), np.cos(theta_1), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

            # Transformation matrix from forearm to upper arm reference frame
            t_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0],
                              [0, 1, 0, 0.13585],
                              [-np.sin(theta_2), 0, np.cos(theta_2), 0],
                              [0, 0, 0, 1]])

            # Transformation matrix from wrist 1 to forearm reference frame
            t_4_3 = np.array([[np.cos(theta_3), 0, np.sin(theta_3), 0.425],
                              [0, 1, 0, 0],
                              [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                              [0, 0, 0, 1]])

            forearm_pos = t_1_0.dot(t_2_1).dot(t_3_2).dot(
                forearm_pos_3)[:3]
            wrist_1_pos = t_1_0.dot(t_2_1).dot(t_3_2).dot(t_4_3).dot(
                wrist_1_pos_4)[:3]

            # Make sure wrist 1 pos is above ground so can actually be reached
            if np.absolute(end_goal[0]) > np.pi / 4 \
                    and forearm_pos[2] > 0.05 and wrist_1_pos[2] > 0.15:
                goal_possible = True

        # Visualize End Goal
        self.display_end_goal(end_goal)

        return end_goal

    def display_end_goal(self, end_goal):
        """See parent class."""
        theta_1 = end_goal[0]
        theta_2 = end_goal[1]
        theta_3 = end_goal[2]

        # shoulder_pos_1 = np.array([0, 0, 0, 1])
        upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
        forearm_pos_3 = np.array([0.425, 0, 0, 1])
        wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

        # Transformation matrix from shoulder to base reference frame
        t_1_0 = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0.089159],
                          [0, 0, 0, 1]])

        # Transformation matrix from upper arm to shoulder reference frame
        t_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],
                          [np.sin(theta_1), np.cos(theta_1), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

        # Transformation matrix from forearm to upper arm reference frame
        t_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0],
                          [0, 1, 0, 0.13585],
                          [-np.sin(theta_2), 0, np.cos(theta_2), 0],
                          [0, 0, 0, 1]])

        # Transformation matrix from wrist 1 to forearm reference frame
        t_4_3 = np.array([[np.cos(theta_3), 0, np.sin(theta_3), 0.425],
                          [0, 1, 0, 0],
                          [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                          [0, 0, 0, 1]])

        # Determine joint position relative to original reference frame
        # shoulder_pos = T_1_0.dot(shoulder_pos_1)
        upper_arm_pos = t_1_0.dot(t_2_1).dot(upper_arm_pos_2)[:3]
        forearm_pos = t_1_0.dot(t_2_1).dot(t_3_2).dot(forearm_pos_3)[:3]
        wrist_1_pos = t_1_0.dot(t_2_1).dot(t_3_2).dot(t_4_3).dot(
            wrist_1_pos_4)[:3]

        joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

        for i in range(3):
            self.sim.data.mocap_pos[i] = joint_pos[i]


class Pendulum(Environment):
    """Pendulum environment class.

    In this environment, an inverted pendulum object is tasked with reaching an
    end goal consisting of the desired joint angle and joint velocity for the
    pendulum.
    """

    def __init__(self,
                 use_contexts=False,
                 random_contexts=False,
                 context_range=None,
                 show=False):
        """Initialize the Pendulum environment.

        Parameters
        ----------
        use_contexts : bool, optional
            specifies whether to add contexts to the observations and add the
            contextual rewards
        random_contexts : bool
            specifies whether the context is a single value, or a random set of
            values between some range
        context_range : list of float or list of (float, float)
            the desired context / goal, or the (lower, upper) bound tuple for
            each dimension of the goal
        show : bool, optional
            specifies whether to render the environment. Defaults to False.

        Raises
        ------
        AssertionError
            If the context_range is not the right form based on whether
            contexts are a single value or random across a range.
        """
        # max number of atomic actions
        max_actions = 1000

        # number of time steps per atomic action.
        timesteps_per_action = 1

        # file name of Mujoco model. This file is stored in the "assets" folder
        model_name = "pendulum.xml"

        # initial state space consisting of the ranges for all joint angles and
        # velocities. In the inverted pendulum task, we randomly sample from
        # the below initial joint position and joint velocity ranges. These
        # values are then converted to the actual state space, which is
        # [cos(pendulum angle), sin(pendulum angle), pendulum velocity].
        initial_state_space = [(np.pi / 4, 7 * np.pi / 4), (-0.05, 0.05)]

        # Supplemental function that converts angle to between [-pi,pi]
        def bound_angle(angle):
            bounded_angle = angle % (2 * np.pi)
            if np.absolute(bounded_angle) > np.pi:
                bounded_angle = -(np.pi - bounded_angle % np.pi)
            return bounded_angle

        # function that maps from the state space to the end goal space
        def project_state_to_end_goal(sim, state):
            return np.array([bound_angle(sim.data.qpos[0]), 15 if state[2] > 15
                             else -15 if state[2] < -15 else state[2]])

        # end goal achievement thresholds. If the agent is within the threshold
        # for each dimension, the end goal has been achieved.
        end_goal_thresholds = np.array([np.deg2rad(9.5), 0.6])

        def contextual_reward(states, goals, next_states):
            return negative_distance(
                states=states,
                goals=goals,
                next_states=next_states,
                state_indices=[0, 2],
                relative_context=False,
                offset=0.0,
                reward_scales=1.0
            )

        super(Pendulum, self).__init__(
            model_name=model_name,
            project_state_to_end_goal=project_state_to_end_goal,
            end_goal_thresholds=end_goal_thresholds,
            initial_state_space=initial_state_space,
            max_actions=max_actions,
            num_frames_skip=timesteps_per_action,
            show=show,
            contextual_reward=contextual_reward,
            use_contexts=use_contexts,
            random_contexts=random_contexts,
            context_range=context_range,
        )

    @property
    def observation_space(self):
        """Return the observation space."""
        # State will include (i) joint angles and (ii) joint velocities
        return gym.spaces.Box(
            low=0, high=1,  # TODO: bounds?
            shape=(2 * len(self.sim.data.qpos) + len(self.sim.data.qvel),),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        """Return the action space."""
        return gym.spaces.Box(
            low=-self.sim.model.actuator_ctrlrange[:, 1],
            high=self.sim.model.actuator_ctrlrange[:, 1],
            dtype=np.float32,
        )

    def get_state(self):
        """See parent class."""
        return np.concatenate(
            [np.cos(self.sim.data.qpos), np.sin(self.sim.data.qpos),
             self.sim.data.qvel]
        )

    def get_next_goal(self):
        """See parent class."""
        end_goal = np.zeros((len(self.context_range)))

        for i in range(len(self.context_range)):
            end_goal[i] = np.random.uniform(self.context_range[i][0],
                                            self.context_range[i][1])

        # Visualize End Goal
        self.display_end_goal(end_goal)

        return end_goal

    def display_end_goal(self, end_goal):
        """See parent class."""
        self.sim.data.mocap_pos[0] = np.array(
            [0.5 * np.sin(end_goal[0]), 0, 0.5 * np.cos(end_goal[0]) + 0.6])
