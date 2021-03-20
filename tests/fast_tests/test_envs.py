"""Contains tests for the contained environments."""
import unittest
import numpy as np
import random
import os
import json
from copy import deepcopy

from flow.core.params import EnvParams

from hbaselines.envs.efficient_hrl.maze_env_utils import line_intersect
from hbaselines.envs.efficient_hrl.maze_env_utils import point_distance
from hbaselines.envs.efficient_hrl.maze_env_utils import construct_maze
from hbaselines.envs.efficient_hrl.maze_env_utils import ray_segment_intersect
from hbaselines.envs.efficient_hrl.envs import AntMaze
from hbaselines.envs.efficient_hrl.envs import AntFall
from hbaselines.envs.efficient_hrl.envs import AntPush
from hbaselines.envs.efficient_hrl.envs import AntFourRooms
from hbaselines.envs.efficient_hrl.envs import HumanoidMaze

from hbaselines.envs.hac.env_utils import check_validity
from hbaselines.envs.hac.envs import UR5, Pendulum

from hbaselines.envs.mixed_autonomy.params.ring \
    import get_flow_params as ring
from hbaselines.envs.mixed_autonomy.params.highway \
    import get_flow_params as highway

from hbaselines.envs.mixed_autonomy.envs.av import AVEnv
from hbaselines.envs.mixed_autonomy.envs.av import AVClosedEnv
from hbaselines.envs.mixed_autonomy.envs.av import AVOpenEnv
from hbaselines.envs.mixed_autonomy.envs.av \
    import CLOSED_ENV_PARAMS as SA_CLOSED_ENV_PARAMS
from hbaselines.envs.mixed_autonomy.envs.av \
    import OPEN_ENV_PARAMS as SA_OPEN_ENV_PARAMS
from hbaselines.envs.mixed_autonomy.envs.av_multi import AVMultiAgentEnv
from hbaselines.envs.mixed_autonomy.envs.av_multi import AVClosedMultiAgentEnv
from hbaselines.envs.mixed_autonomy.envs.av_multi import AVOpenMultiAgentEnv
from hbaselines.envs.mixed_autonomy.envs.av_multi \
    import OPEN_ENV_PARAMS as MA_OPEN_ENV_PARAMS
from hbaselines.envs.mixed_autonomy.envs.av_multi \
    import CLOSED_ENV_PARAMS as MA_CLOSED_ENV_PARAMS
from hbaselines.envs.mixed_autonomy.envs.ring_nonflow import RingEnv
from hbaselines.envs.mixed_autonomy.envs.ring_nonflow import RingSingleAgentEnv
from hbaselines.envs.mixed_autonomy.envs.ring_nonflow import RingMultiAgentEnv

from hbaselines.envs.point2d import Point2DEnv
from hbaselines.utils.env_util import create_env

import hbaselines.config as hbaselines_config

os.environ["TEST_FLAG"] = "True"


class TestEfficientHRLAntEnvironments(unittest.TestCase):
    """Test the Ant* environments in envs/efficient_hrl/."""

    def test_maze_env_utils(self):
        """Test hbaselines/envs/efficient_hrl/maze_env_utils.py."""
        # test construct_maze
        for maze_id in ["Maze", "Push", "Fall", "Block", "BlockMaze",
                        "FourRooms"]:
            construct_maze(maze_id)
        self.assertRaises(NotImplementedError, construct_maze, maze_id="error")

        # test point_distance
        p1 = (0, 0)
        p2 = (2, 2)
        self.assertAlmostEqual(point_distance(p1, p2), np.sqrt(2**2 + 2**2))

        # test line_intersect
        p1 = (0, 0)
        p2 = (2, 2)
        p3 = (0, 2)
        p4 = (2, 0)
        x, y, *_ = line_intersect(p1, p2, p3, p4)
        self.assertAlmostEqual(x, 1)
        self.assertAlmostEqual(y, 1)

        # test ray_segment_intersect
        ray = ((0, 1), 2)
        segment = ((3, 4), (5, 6))
        self.assertIsNone(ray_segment_intersect(ray, segment))

    def test_contextual_reward(self):
        """Check the functionality of the context_space attribute.

        This method is tested for the following environments:

        1. AntMaze
        2. AntPush
        3. AntFall
        4. AntFourRooms
        """
        from hbaselines.envs.efficient_hrl.envs import REWARD_SCALE

        # test case 1
        env = AntMaze(use_contexts=True, context_range=[0, 0])
        self.assertAlmostEqual(
            env.contextual_reward(
                np.array([0, 0]), np.array([1, 1]), np.array([2, 2])),
            -1.4142135624084504 * REWARD_SCALE
        )

        # test case 2
        env = AntPush(use_contexts=True, context_range=[0, 0])
        self.assertAlmostEqual(
            env.contextual_reward(
                np.array([0, 0]), np.array([1, 1]), np.array([2, 2])),
            -1.4142135624084504 * REWARD_SCALE
        )

        # test case 3
        env = AntFall(use_contexts=True, context_range=[0, 0, 0])
        self.assertAlmostEqual(
            env.contextual_reward(
                np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2])),
            -1.7320508075977448 * REWARD_SCALE
        )

        # test case 4
        env = AntFourRooms(use_contexts=True, context_range=[0, 0])
        self.assertAlmostEqual(
            env.contextual_reward(
                np.array([0, 0]), np.array([1, 1]), np.array([2, 2])),
            -1.4142135624084504 * REWARD_SCALE
        )

    def test_context_space(self):
        """Check the functionality of the context_space attribute.

        This method is tested for the following cases:

        1. no context
        2. random contexts
        3. fixed single context
        4. fixed multiple contexts
        """
        # test case 1
        env = AntMaze(use_contexts=False)
        self.assertIsNone(env.context_space)

        # test case 2
        env = AntMaze(use_contexts=True, random_contexts=True,
                      context_range=[(-4, 5), (4, 20)])
        np.testing.assert_almost_equal(
            env.context_space.low, np.array([-4, 4]))
        np.testing.assert_almost_equal(
            env.context_space.high, np.array([5, 20]))

        # test case 3
        env = AntMaze(use_contexts=True, random_contexts=False,
                      context_range=[-4, 5])
        np.testing.assert_almost_equal(
            env.context_space.low, np.array([-4, 5]))
        np.testing.assert_almost_equal(
            env.context_space.high, np.array([-4, 5]))

        # test case 4
        env = AntMaze(use_contexts=True, random_contexts=False,
                      context_range=[[-4, 5], [-3, 10], [-2, 7]])
        np.testing.assert_almost_equal(
            env.context_space.low, np.array([-4, 5]))
        np.testing.assert_almost_equal(
            env.context_space.high, np.array([-2, 10]))

    def test_current_context(self):
        """Check the functionality of the current_context attribute.

        This method is tested for the following cases:

        1. no context
        2. random contexts
        3. fixed single context
        4. fixed multiple contexts
        """
        np.random.seed(0)
        random.seed(0)

        # test case 1
        env = AntMaze(use_contexts=False)
        env.reset()
        self.assertIsNone(env.current_context)

        # test case 2
        env = AntMaze(use_contexts=True, random_contexts=True,
                      context_range=[(-4, 5), (4, 20)])
        env.reset()
        np.testing.assert_almost_equal(
            env.current_context, np.array([3.5997967, 16.1272704]))

        # test case 3
        env = AntMaze(use_contexts=True, random_contexts=False,
                      context_range=[-4, 5])
        env.reset()
        np.testing.assert_almost_equal(
            env.current_context, np.array([-4, 5]))

        # test case 4
        env = AntMaze(use_contexts=True, random_contexts=False,
                      context_range=[[-4, 5], [-3, 6], [-2, 7]])
        env.reset()
        np.testing.assert_almost_equal(
            env.current_context, np.array([-3, 6]))
        env.reset()
        np.testing.assert_almost_equal(
            env.current_context, np.array([-4, 5]))


class TestEfficientHRLHumanoidEnvironments(unittest.TestCase):
    """Test the Humanoid* environments in envs/efficient_hrl/."""

    def test_contextual_reward(self):
        """Check the functionality of the context_space attribute.

        This method is tested for the following environments:

        1. HumanoidMaze
        """
        # test case 1
        env = HumanoidMaze(use_contexts=True, context_range=[0, 0])
        self.assertAlmostEqual(
            env.contextual_reward(
                np.array([0, 0]), np.array([1, 1]), np.array([2, 2])),
            0.8216682531742017
        )

    def test_context_space(self):
        """Check the functionality of the context_space attribute.

        This method is tested for the following cases:

        1. no context
        2. random contexts
        3. fixed single context
        4. fixed multiple contexts
        """
        # test case 1
        env = HumanoidMaze(use_contexts=False)
        self.assertIsNone(env.context_space)

        # test case 2
        env = HumanoidMaze(
            use_contexts=True,
            random_contexts=True,
            context_range=[(-4, 5), (4, 20)],
        )
        np.testing.assert_almost_equal(
            env.context_space.low, np.array([-4, 4]))
        np.testing.assert_almost_equal(
            env.context_space.high, np.array([5, 20]))

        # test case 3
        env = HumanoidMaze(
            use_contexts=True,
            random_contexts=False,
            context_range=[-4, 5],
        )
        np.testing.assert_almost_equal(
            env.context_space.low, np.array([-4, 5]))
        np.testing.assert_almost_equal(
            env.context_space.high, np.array([-4, 5]))

        # test case 4
        env = HumanoidMaze(
            use_contexts=True,
            random_contexts=False,
            context_range=[[-4, 5], [-3, 10], [-2, 7]],
        )
        np.testing.assert_almost_equal(
            env.context_space.low, np.array([-4, 5]))
        np.testing.assert_almost_equal(
            env.context_space.high, np.array([-2, 10]))

    def test_current_context(self):
        """Check the functionality of the current_context attribute.

        This method is tested for the following cases:

        1. no context
        2. random contexts
        3. fixed single context
        4. fixed multiple contexts
        """
        np.random.seed(0)
        random.seed(0)

        # test case 1
        env = HumanoidMaze(use_contexts=False)
        env.reset()
        self.assertIsNone(env.current_context)

        # test case 2
        env = HumanoidMaze(
            use_contexts=True,
            random_contexts=True,
            context_range=[(-4, 5), (4, 20)],
        )
        env.reset()
        np.testing.assert_almost_equal(
            env.current_context, np.array([3.5997967, 16.1272704]))

        # test case 3
        env = HumanoidMaze(
            use_contexts=True,
            random_contexts=False,
            context_range=[-4, 5],
        )
        env.reset()
        np.testing.assert_almost_equal(
            env.current_context, np.array([-4, 5]))

        # test case 4
        env = HumanoidMaze(
            use_contexts=True,
            random_contexts=False,
            context_range=[[-4, 5], [-3, 6], [-2, 7]],
        )
        env.reset()
        np.testing.assert_almost_equal(
            env.current_context, np.array([-3, 6]))
        env.reset()
        np.testing.assert_almost_equal(
            env.current_context, np.array([-4, 5]))


class TestHACEnvironments(unittest.TestCase):
    """Test the environments in envs/hac/."""

    def test_check_validity(self):
        bad_model_name = "bad"
        model_name = "good.xml"

        bad_initial_state_space = [(-1, 1), (2, 1)]
        initial_state_space = [(-1, 1), (1, 2)]

        bad_max_actions = -100
        max_actions = 100

        bad_timesteps_per_action = -100
        timesteps_per_action = 100

        self.assertRaises(
            AssertionError,
            check_validity,
            model_name=bad_model_name,
            initial_state_space=initial_state_space,
            max_actions=max_actions,
            timesteps_per_action=timesteps_per_action
        )

        self.assertRaises(
            AssertionError,
            check_validity,
            model_name=model_name,
            initial_state_space=bad_initial_state_space,
            max_actions=max_actions,
            timesteps_per_action=timesteps_per_action
        )

        self.assertRaises(
            AssertionError,
            check_validity,
            model_name=model_name,
            initial_state_space=initial_state_space,
            max_actions=bad_max_actions,
            timesteps_per_action=timesteps_per_action
        )

        self.assertRaises(
            AssertionError,
            check_validity,
            model_name=model_name,
            initial_state_space=initial_state_space,
            max_actions=max_actions,
            timesteps_per_action=bad_timesteps_per_action
        )


class TestUR5(unittest.TestCase):
    """Tests the UR5 environment class."""

    def setUp(self):
        """Create the UR5 environment.

        This follows the ur5.py example located in example_designs/.
        """
        self.env = UR5(
            use_contexts=True,
            random_contexts=True,
            context_range=[(-np.pi, np.pi),
                           (-np.pi / 4, 0),
                           (-np.pi / 4, np.pi / 4)]
        )
        self.env.reset()

    def tearDown(self):
        del self.env

    def test_init(self):
        """Ensure that all variables are being initialized properly."""
        self.assertEqual(self.env.name, 'ur5.xml')
        self.assertEqual(self.env.observation_space.shape[0], 6)
        self.assertEqual(self.env.action_space.shape[0], 3)
        np.testing.assert_array_almost_equal(
            (self.env.action_space.high - self.env.action_space.low) / 2,
            [3.15, 5.00, 3.15])
        np.testing.assert_array_almost_equal(
            (self.env.action_space.high + self.env.action_space.low) / 2,
            [0.00, 0.00, 0.00])
        self.assertEqual(len(self.env.context_range), 3)
        np.testing.assert_array_almost_equal(
            self.env.end_goal_thresholds, [0.17453293 for _ in range(3)])
        self.assertEqual(self.env.max_actions, 600)
        self.assertEqual(self.env.visualize, False)
        self.assertEqual(self.env.viewer, None)
        self.assertEqual(self.env.num_frames_skip, 1)
        np.testing.assert_array_almost_equal(
            self.env.context_space.low, [-3.141593, -0.785398, -0.785398])
        np.testing.assert_array_almost_equal(
            self.env.context_space.high, [3.141593, 0., 0.785398])

    def test_step(self):
        """Ensure the step method is functioning properly.

        This does the following tasks:
        * checks that the simulation control is set to the action.
        * Checks that a sufficient number of steps have passed.
        """
        action = [1, 2, 3]
        steps_before = self.env.num_steps
        self.env.step(action)
        steps_after = self.env.num_steps

        # check the number of steps that have passed
        self.assertEqual(steps_after - steps_before, self.env.num_frames_skip)
        # check the control method
        np.testing.assert_array_almost_equal(action, self.env.sim.data.ctrl[:])

    def test_reset(self):
        """Ensure the state initialization is within the expected range."""
        state = self.env.reset()
        for i in range(len(state)):
            self.assertTrue(state[i] >= self.env.initial_state_space[i][0])
            self.assertTrue(state[i] <= self.env.initial_state_space[i][1])


class TestPendulum(unittest.TestCase):
    """Tests the Pendulum environment class."""

    def setUp(self):
        """Create the UR5 environment.

        This follows the pendulum.py example located in example_designs/.
        """
        self.env = Pendulum(
            use_contexts=True,
            random_contexts=True,
            context_range=[(np.deg2rad(-16), np.deg2rad(16)), (-0.6, 0.6)]
        )
        self.env.reset()

    def tearDown(self):
        del self.env

    def test_init(self):
        """Ensure that all variables are being initialized properly."""
        self.assertEqual(self.env.name, 'pendulum.xml')
        self.assertEqual(self.env.observation_space.shape[0], 3)
        self.assertEqual(self.env.action_space.shape[0], 1)
        np.testing.assert_array_almost_equal(
            (self.env.action_space.high - self.env.action_space.low) / 2, [2])
        np.testing.assert_array_almost_equal(
            (self.env.action_space.high + self.env.action_space.low) / 2, [0])
        self.assertEqual(len(self.env.context_range), 2)
        np.testing.assert_array_almost_equal(
            self.env.end_goal_thresholds, [0.16580628, 0.6])
        self.assertEqual(
            self.env.context_range,
            [(-0.2792526803190927, 0.2792526803190927), (-0.6, 0.6)])
        self.assertEqual(self.env.max_actions, 1000)
        self.assertEqual(self.env.visualize, False)
        self.assertEqual(self.env.viewer, None)
        self.assertEqual(self.env.num_frames_skip, 1)
        np.testing.assert_array_almost_equal(
            self.env.context_space.low, [-0.279253, -0.6])
        np.testing.assert_array_almost_equal(
            self.env.context_space.high, [0.279253, 0.6])

    def test_step(self):
        """Ensure the step method is functioning properly.

        This does the following tasks:
        * checks that the simulation control is set to the action.
        * Checks that a sufficient number of steps have passed.
        """
        action = [1]
        steps_before = self.env.num_steps
        self.env.step(action)
        steps_after = self.env.num_steps

        # check the number of steps that have passed
        self.assertEqual(steps_after - steps_before, self.env.num_frames_skip)
        # check the control method
        np.testing.assert_array_almost_equal(action, self.env.sim.data.ctrl[:])

    def test_reset(self):
        """Ensure the state initialization is within the expected range."""
        state = self.env.reset()
        num_obj = len(state) // 3
        for i in range(num_obj):
            self.assertTrue(np.arccos(state[i])
                            >= self.env.initial_state_space[i][0])
            self.assertTrue(np.arccos(state[i])
                            <= self.env.initial_state_space[i][1])
            self.assertTrue(state[i + 2 * num_obj]
                            >= self.env.initial_state_space[i + num_obj][0])
            self.assertTrue(state[i + 2 * num_obj]
                            <= self.env.initial_state_space[i + num_obj][1])

    def test_display_end_goal(self):
        pass

    def test_get_next_goal(self):
        pass

    def test_display_subgoal(self):
        pass


class TestMixedAutonomyEnvs(unittest.TestCase):
    """Test the functionality of each of the trainable mixed-autonomy envs.

    Each of these environments are tests for the following cases:

    1. the observation space matches its expected values
    2. the action space matches its expected values

    For some of the the multi-agent environments, we also perform the following
    tests:

    3. the agent IDs match their expected values
    """

    def setUp(self):
        self.maxDiff = None

    # ======================================================================= #
    #                                 ring-v0                                 #
    # ======================================================================= #

    def test_single_agent_ring_v0(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("ring-v0")

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(15)]),
            expected_max=np.array([float("inf") for _ in range(15)]),
            expected_size=15,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1.0 for _ in range(1)]),
            expected_max=np.array([1.0 for _ in range(1)]),
            expected_size=1,
        )
        self.assertEqual(
            env.wrapped_env.env_params.additional_params["max_accel"], 0.5)

        # kill the environment
        env.wrapped_env.terminate()

    def test_multi_agent_ring_v0(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("multiagent-ring-v0")

        # test case 1
        test_space(
            env.observation_space["rl_0_0"],
            expected_min=np.array([-float("inf") for _ in range(15)]),
            expected_max=np.array([float("inf") for _ in range(15)]),
            expected_size=15,
        )

        # test case 2
        test_space(
            env.action_space["rl_0_0"],
            expected_min=np.array([-1.0]),
            expected_max=np.array([1.0]),
            expected_size=1,
        )
        self.assertEqual(
            env.wrapped_env.env_params.additional_params["max_accel"], 0.5)

        # test case 3
        self.assertListEqual(
            sorted(env.agents),
            ['rl_0_0']
        )

        # kill the environment
        env.wrapped_env.terminate()

    # ======================================================================= #
    #                                 ring-v0                                 #
    # ======================================================================= #

    def test_single_agent_ring_v0_fast(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("ring-v0-fast")

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(15)]),
            expected_max=np.array([float("inf") for _ in range(15)]),
            expected_size=15,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1.0 for _ in range(1)]),
            expected_max=np.array([1.0 for _ in range(1)]),
            expected_size=1,
        )
        self.assertEqual(env.max_accel, 0.5)

    def test_multi_agent_ring_v0_fast(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("multiagent-ring-v0-fast")

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(15)]),
            expected_max=np.array([float("inf") for _ in range(15)]),
            expected_size=15,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1.0 for _ in range(1)]),
            expected_max=np.array([1.0 for _ in range(1)]),
            expected_size=1,
        )
        self.assertEqual(env.max_accel, 0.5)

        # test case 3
        self.assertListEqual(
            sorted(env.rl_ids),
            [0]
        )

    # ======================================================================= #
    #                                merge-v0                                 #
    # ======================================================================= #

    def test_single_agent_merge_v0(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("merge-v0")

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([0 for _ in range(25)]),
            expected_max=np.array([1 for _ in range(25)]),
            expected_size=25,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1.5 for _ in range(5)]),
            expected_max=np.array([1.5 for _ in range(5)]),
            expected_size=5,
        )

        # kill the environment
        env.wrapped_env.terminate()

    # ======================================================================= #
    #                                merge-v1                                 #
    # ======================================================================= #

    def test_single_agent_merge_v1(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("merge-v1")

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([0 for _ in range(65)]),
            expected_max=np.array([1 for _ in range(65)]),
            expected_size=65,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1.5 for _ in range(13)]),
            expected_max=np.array([1.5 for _ in range(13)]),
            expected_size=13,
        )

        # kill the environment
        env.wrapped_env.terminate()

    # ======================================================================= #
    #                                merge-v2                                 #
    # ======================================================================= #

    def test_single_agent_merge_v2(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("merge-v2")

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([0 for _ in range(85)]),
            expected_max=np.array([1 for _ in range(85)]),
            expected_size=85,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1.5 for _ in range(17)]),
            expected_max=np.array([1.5 for _ in range(17)]),
            expected_size=17,
        )

        # kill the environment
        env.wrapped_env.terminate()

    # ======================================================================= #
    #                               highway-v0                                #
    # ======================================================================= #

    def test_single_agent_highway_v0(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("highway-v0")

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(75)]),
            expected_max=np.array([float("inf") for _ in range(75)]),
            expected_size=75,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1.0 for _ in range(5)]),
            expected_max=np.array([1.0 for _ in range(5)]),
            expected_size=5,
        )
        self.assertEqual(
            env.wrapped_env.env_params.additional_params["max_accel"], 0.5)

        # kill the environment
        env.wrapped_env.terminate()

    def test_multi_agent_highway_v0(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("multiagent-highway-v0", shared=True)

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(15)]),
            expected_max=np.array([float("inf") for _ in range(15)]),
            expected_size=15,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1.0 for _ in range(1)]),
            expected_max=np.array([1.0 for _ in range(1)]),
            expected_size=1,
        )
        self.assertEqual(
            env.wrapped_env.env_params.additional_params["max_accel"], 0.5)

        # kill the environment
        env.wrapped_env.terminate()

    # ======================================================================= #
    #                               highway-v1                                #
    # ======================================================================= #

    def test_single_agent_highway_v1(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("highway-v1")

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(75)]),
            expected_max=np.array([float("inf") for _ in range(75)]),
            expected_size=75,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1.0 for _ in range(5)]),
            expected_max=np.array([1.0 for _ in range(5)]),
            expected_size=5,
        )
        self.assertEqual(
            env.wrapped_env.env_params.additional_params["max_accel"], 0.5)

        # kill the environment
        env.wrapped_env.terminate()

    def test_multi_agent_highway_v1(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("multiagent-highway-v1", shared=True)

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(15)]),
            expected_max=np.array([float("inf") for _ in range(15)]),
            expected_size=15,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1.0 for _ in range(1)]),
            expected_max=np.array([1.0 for _ in range(1)]),
            expected_size=1,
        )
        self.assertEqual(
            env.wrapped_env.env_params.additional_params["max_accel"], 0.5)

        # kill the environment
        env.wrapped_env.terminate()

    # ======================================================================= #
    #                               highway-v2                                #
    # ======================================================================= #

    def test_single_agent_highway_v2(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("highway-v2")

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(75)]),
            expected_max=np.array([float("inf") for _ in range(75)]),
            expected_size=75,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1.0 for _ in range(5)]),
            expected_max=np.array([1.0 for _ in range(5)]),
            expected_size=5,
        )
        self.assertEqual(
            env.wrapped_env.env_params.additional_params["max_accel"], 0.5)

        # kill the environment
        env.wrapped_env.terminate()

    def test_multi_agent_highway_v2(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("multiagent-highway-v2", shared=True)

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(15)]),
            expected_max=np.array([float("inf") for _ in range(15)]),
            expected_size=15,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1.0 for _ in range(1)]),
            expected_max=np.array([1.0 for _ in range(1)]),
            expected_size=1,
        )
        self.assertEqual(
            env.wrapped_env.env_params.additional_params["max_accel"], 0.5)

        # kill the environment
        env.wrapped_env.terminate()

    # ======================================================================= #
    #                               highway-v3                                #
    # ======================================================================= #

    def test_single_agent_highway_v3(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("highway-v3")

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(75)]),
            expected_max=np.array([float("inf") for _ in range(75)]),
            expected_size=75,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([0 for _ in range(5)]),
            expected_max=np.array([15 for _ in range(5)]),
            expected_size=5,
        )

        # kill the environment
        env.wrapped_env.terminate()

    def test_multi_agent_highway_v3(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("multiagent-highway-v3", shared=True)

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(15)]),
            expected_max=np.array([float("inf") for _ in range(15)]),
            expected_size=15,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([0 for _ in range(1)]),
            expected_max=np.array([15 for _ in range(1)]),
            expected_size=1,
        )

        # kill the environment
        env.wrapped_env.terminate()

    # ======================================================================= #
    #                                 i210-v0                                 #
    # ======================================================================= #

    # FIXME
    # def test_single_agent_i210_v0(self):
    #     # set a random seed
    #     set_seed(0)
    #
    #     # create the environment
    #     env, _ = create_env("i210-v0")
    #
    #     # test case 1
    #     test_space(
    #         env.observation_space,
    #         expected_min=np.array([-float("inf") for _ in range(1250)]),
    #         expected_max=np.array([float("inf") for _ in range(1250)]),
    #         expected_size=1250,
    #     )
    #
    #     # test case 2
    #     test_space(
    #         env.action_space,
    #         expected_min=np.array([-1.0 for _ in range(50)]),
    #         expected_max=np.array([1.0 for _ in range(50)]),
    #         expected_size=50,
    #     )
    #     self.assertEqual(
    #         env.wrapped_env.env_params.additional_params["max_accel"], 0.5)
    #
    #     # kill the environment
    #     env.wrapped_env.terminate()
    #
    # def test_multi_agent_i210_v0(self):
    #     # set a random seed
    #     set_seed(0)
    #
    #     # create the environment
    #     env, _ = create_env("multiagent-i210-v0")
    #
    #     # test case 1
    #     test_space(
    #         env.observation_space["lane_0"],
    #         expected_min=np.array([-float("inf") for _ in range(250)]),
    #         expected_max=np.array([float("inf") for _ in range(250)]),
    #         expected_size=250,
    #     )
    #
    #     # test case 2
    #     test_space(
    #         env.action_space["lane_0"],
    #         expected_min=np.array([-1.0 for _ in range(10)]),
    #         expected_max=np.array([1.0 for _ in range(10)]),
    #         expected_size=10,
    #     )
    #     self.assertEqual(
    #         env.wrapped_env.env_params.additional_params["max_accel"], 0.5)
    #
    #     # test case 3
    #     self.assertListEqual(
    #         sorted(env.agents),
    #         ['lane_0', 'lane_1', 'lane_2', 'lane_3', 'lane_4']
    #     )
    #
    #     # kill the environment
    #     env.wrapped_env.terminate()

    # ======================================================================= #
    #                                 i210-v1                                 #
    # ======================================================================= #

    # FIXME
    # def test_single_agent_i210_v1(self):
    #     # set a random seed
    #     set_seed(0)
    #
    #     # create the environment
    #     env, _ = create_env("i210-v1")
    #
    #     # test case 1
    #     test_space(
    #         env.observation_space,
    #         expected_min=np.array([-float("inf") for _ in range(1250)]),
    #         expected_max=np.array([float("inf") for _ in range(1250)]),
    #         expected_size=1250,
    #     )
    #
    #     # test case 2
    #     test_space(
    #         env.action_space,
    #         expected_min=np.array([-1.0 for _ in range(50)]),
    #         expected_max=np.array([1.0 for _ in range(50)]),
    #         expected_size=50,
    #     )
    #     self.assertEqual(
    #         env.wrapped_env.env_params.additional_params["max_accel"], 0.5)
    #
    #     # kill the environment
    #     env.wrapped_env.terminate()
    #
    # def test_multi_agent_i210_v1(self):
    #     # set a random seed
    #     set_seed(0)
    #
    #     # create the environment
    #     env, _ = create_env("multiagent-i210-v1")
    #
    #     # test case 1
    #     test_space(
    #         env.observation_space["lane_0"],
    #         expected_min=np.array([-float("inf") for _ in range(250)]),
    #         expected_max=np.array([float("inf") for _ in range(250)]),
    #         expected_size=250,
    #     )
    #
    #     # test case 2
    #     test_space(
    #         env.action_space["lane_0"],
    #         expected_min=np.array([-1.0 for _ in range(10)]),
    #         expected_max=np.array([1.0 for _ in range(10)]),
    #         expected_size=10,
    #     )
    #     self.assertEqual(
    #         env.wrapped_env.env_params.additional_params["max_accel"], 0.5)
    #
    #     # test case 3
    #     self.assertListEqual(
    #         sorted(env.agents),
    #         ['lane_0', 'lane_1', 'lane_2', 'lane_3', 'lane_4']
    #     )
    #
    #     # kill the environment
    #     env.wrapped_env.terminate()

    # ======================================================================= #
    #                                 i210-v2                                 #
    # ======================================================================= #

    # FIXME
    # def test_single_agent_i210_v2(self):
    #     # set a random seed
    #     set_seed(0)
    #
    #     # create the environment
    #     env, _ = create_env("i210-v2")
    #
    #     # test case 1
    #     test_space(
    #         env.observation_space,
    #         expected_min=np.array([-float("inf") for _ in range(1250)]),
    #         expected_max=np.array([float("inf") for _ in range(1250)]),
    #         expected_size=1250,
    #     )
    #
    #     # test case 2
    #     test_space(
    #         env.action_space,
    #         expected_min=np.array([-1.0 for _ in range(50)]),
    #         expected_max=np.array([1.0 for _ in range(50)]),
    #         expected_size=50,
    #     )
    #     self.assertEqual(
    #         env.wrapped_env.env_params.additional_params["max_accel"], 0.5)
    #
    #     # kill the environment
    #     env.wrapped_env.terminate()
    #
    # def test_multi_agent_i210_v2(self):
    #     # set a random seed
    #     set_seed(0)
    #
    #     # create the environment
    #     env, _ = create_env("multiagent-i210-v2")
    #
    #     # test case 1
    #     test_space(
    #         env.observation_space["lane_0"],
    #         expected_min=np.array([-float("inf") for _ in range(250)]),
    #         expected_max=np.array([float("inf") for _ in range(250)]),
    #         expected_size=250,
    #     )
    #
    #     # test case 2
    #     test_space(
    #         env.action_space["lane_0"],
    #         expected_min=np.array([-1.0 for _ in range(10)]),
    #         expected_max=np.array([1.0 for _ in range(10)]),
    #         expected_size=10,
    #     )
    #     self.assertEqual(
    #         env.wrapped_env.env_params.additional_params["max_accel"], 0.5)
    #
    #     # test case 3
    #     self.assertListEqual(
    #         sorted(env.agents),
    #         ['lane_0', 'lane_1', 'lane_2', 'lane_3', 'lane_4']
    #     )
    #
    #     # kill the environment
    #     env.wrapped_env.terminate()


class TestAV(unittest.TestCase):
    """Tests the automated vehicles single agent environments."""

    def setUp(self):
        self.sim_params = deepcopy(ring(
            stopping_penalty=True,
            acceleration_penalty=True,
        ))["sim"]
        self.sim_params.render = False

        # for AVClosedEnv
        flow_params_closed = deepcopy(ring(
            stopping_penalty=True,
            acceleration_penalty=True,
        ))

        self.network_closed = flow_params_closed["network"](
            name="test_closed",
            vehicles=flow_params_closed["veh"],
            net_params=flow_params_closed["net"],
        )
        self.env_params_closed = flow_params_closed["env"]
        self.env_params_closed.warmup_steps = 0
        self.env_params_closed.additional_params = SA_CLOSED_ENV_PARAMS.copy()

        # for AVOpenEnv
        flow_params_open = deepcopy(highway(
            fixed_boundary=False,
            stopping_penalty=True,
            acceleration_penalty=True,
            use_follower_stopper=False,
        ))

        self.network_open = flow_params_open["network"](
            name="test_open",
            vehicles=flow_params_open["veh"],
            net_params=flow_params_open["net"],
        )
        self.env_params_open = flow_params_open["env"]
        self.env_params_open.warmup_steps = 0
        self.env_params_open.additional_params = SA_OPEN_ENV_PARAMS.copy()

    def test_base_env(self):
        """Validate the functionality of the AVEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2. that the observation space matches its expected values
        3. that the action space matches its expected values
        4. that the observed vehicle IDs after a reset matches its expected
           values
        """
        # test case 1
        self.assertTrue(
            test_additional_params(
                env_class=AVEnv,
                sim_params=self.sim_params,
                network=self.network_closed,
                additional_params={
                    "max_accel": 3,
                    "stopping_penalty": True,
                    "acceleration_penalty": True,
                    "use_follower_stopper": True,
                    "obs_frames": 5,
                },
            )
        )

        # Set a random seed.
        random.seed(0)
        np.random.seed(0)

        # Create a single lane environment.
        env_single = AVEnv(
            env_params=self.env_params_closed,
            sim_params=self.sim_params,
            network=self.network_closed
        )

        # test case 2
        test_space(
            gym_space=env_single.observation_space,
            expected_size=3 * env_single.initial_vehicles.num_rl_vehicles,
            expected_min=-float("inf"),
            expected_max=float("inf"),
        )

        # test case 3
        test_space(
            gym_space=env_single.action_space,
            expected_size=env_single.initial_vehicles.num_rl_vehicles,
            expected_min=-1,
            expected_max=1,
        )

        # test case 4
        self.assertTrue(
            test_observed(
                env_class=AVEnv,
                sim_params=self.sim_params,
                network=self.network_closed,
                env_params=self.env_params_closed,
                expected_observed=['human_0_0']
            )
        )

    def test_closed_env(self):
        """Validate the functionality of the AVClosedEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2. that the number of vehicles is properly modified in between resets
        """
        # test case 1
        self.assertTrue(
            test_additional_params(
                env_class=AVClosedEnv,
                sim_params=self.sim_params,
                network=self.network_closed,
                additional_params={
                    "max_accel": 3,
                    "stopping_penalty": True,
                    "acceleration_penalty": True,
                    "use_follower_stopper": True,
                    "obs_frames": 5,
                    "ring_length": [220, 270],
                },
            )
        )

        # set a random seed to ensure the network lengths are always the same
        # during testing
        random.seed(1)

        # test case 2
        env = AVClosedEnv(
            env_params=self.env_params_closed,
            sim_params=self.sim_params,
            network=self.network_closed
        )

        # reset the network several times and check its number of vehicle
        self.assertEqual(env.k.network.length(), 260.4)
        env.reset()
        self.assertEqual(env.k.network.length(), 228.4)
        env.reset()
        self.assertEqual(env.k.network.length(), 268.4)

    def test_open_env(self):
        """Validate the functionality of the AVOpenEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2. that the inflow rate of vehicles is properly modified in between
           resets
        """
        # test case 1
        self.assertTrue(
            test_additional_params(
                env_class=AVOpenEnv,
                sim_params=self.sim_params,
                network=self.network_open,
                additional_params={
                    "max_accel": 3,
                    "stopping_penalty": True,
                    "acceleration_penalty": True,
                    "use_follower_stopper": True,
                    "obs_frames": 5,
                    "inflows": [1000, 2000],
                    "rl_penetration": 0.1,
                    "num_rl": 5,
                    "control_range": [500, 2500],
                    "warmup_path": None,
                },
            )
        )

        # set a random seed to ensure the network lengths are always the same
        # during testing
        random.seed(1)

        # test case 2
        env = AVOpenEnv(
            env_params=self.env_params_open,
            sim_params=self.sim_params,
            network=self.network_open
        )

        # reset the network several times and check its inflow rate
        inflows = env.net_params.inflows.get()
        for inflow_i in inflows:
            veh_type = inflow_i["vtype"]
            expected_rate = 2114 if veh_type == "human" else 100
            self.assertAlmostEqual(inflow_i["vehsPerHour"], expected_rate)

        env.reset()
        inflows = env.net_params.inflows.get()
        for inflow_i in inflows:
            veh_type = inflow_i["vtype"]
            expected_rate = 1023.3 if veh_type == "human" else 113.7
            self.assertAlmostEqual(inflow_i["vehsPerHour"], expected_rate)

        env.reset()
        inflows = env.net_params.inflows.get()
        for inflow_i in inflows:
            veh_type = inflow_i["vtype"]
            expected_rate = 1680.3 if veh_type == "human" else 186.7
            self.assertAlmostEqual(inflow_i["vehsPerHour"], expected_rate)


class TestAVMulti(unittest.TestCase):
    """Tests the automated vehicles multi-agent environments."""

    def setUp(self):
        self.sim_params = deepcopy(ring(
            stopping_penalty=True,
            acceleration_penalty=True,
            multiagent=True,
        ))["sim"]
        self.sim_params.render = False

        # for AVClosedMultiAgentEnv
        flow_params_closed = deepcopy(ring(
            stopping_penalty=True,
            acceleration_penalty=True,
            multiagent=True,
        ))

        self.network_closed = flow_params_closed["network"](
            name="test_closed",
            vehicles=flow_params_closed["veh"],
            net_params=flow_params_closed["net"],
        )
        self.env_params_closed = flow_params_closed["env"]
        self.env_params_closed.warmup_steps = 0
        self.env_params_closed.additional_params = MA_CLOSED_ENV_PARAMS.copy()

        # for AVOpenMultiAgentEnv
        flow_params_open = deepcopy(highway(
            fixed_boundary=False,
            stopping_penalty=True,
            acceleration_penalty=True,
            multiagent=True,
            use_follower_stopper=False,
        ))

        self.network_open = flow_params_open["network"](
            name="test_open",
            vehicles=flow_params_open["veh"],
            net_params=flow_params_open["net"],
        )
        self.env_params_open = flow_params_open["env"]
        self.env_params_open.warmup_steps = 0
        self.env_params_open.additional_params = MA_OPEN_ENV_PARAMS.copy()

    def test_base_env(self):
        """Validate the functionality of the AVMultiAgentEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2. that the observation space matches its expected values
        3. that the action space matches its expected values
        4. that the observed vehicle IDs after a reset matches its expected
           values
        """
        # test case 1
        self.assertTrue(
            test_additional_params(
                env_class=AVMultiAgentEnv,
                sim_params=self.sim_params,
                network=self.network_closed,
                additional_params={
                    "max_accel": 3,
                    "stopping_penalty": True,
                    "acceleration_penalty": True,
                    "use_follower_stopper": True,
                    "obs_frames": 5,
                },
            )
        )

        # Set a random seed.
        random.seed(0)
        np.random.seed(0)

        # Create a single lane environment.
        env_single = AVMultiAgentEnv(
            env_params=self.env_params_closed,
            sim_params=self.sim_params,
            network=self.network_closed
        )

        # test case 2
        test_space(
            gym_space=env_single.observation_space,
            expected_size=3,
            expected_min=-float("inf"),
            expected_max=float("inf"),
        )

        # test case 3
        test_space(
            gym_space=env_single.action_space,
            expected_size=1,
            expected_min=-1,
            expected_max=1,
        )

        # test case 4
        self.assertTrue(
            test_observed(
                env_class=AVMultiAgentEnv,
                sim_params=self.sim_params,
                network=self.network_closed,
                env_params=self.env_params_closed,
                expected_observed=['human_0_0']
            )
        )

    def test_closed_env(self):
        """Validate the functionality of the AVClosedMultiAgentEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2. that the number of vehicles is properly modified in between resets
        """
        # test case 1
        self.assertTrue(
            test_additional_params(
                env_class=AVClosedMultiAgentEnv,
                sim_params=self.sim_params,
                network=self.network_closed,
                additional_params={
                    "max_accel": 3,
                    "stopping_penalty": True,
                    "use_follower_stopper": True,
                    "acceleration_penalty": True,
                    "obs_frames": 5,
                    "ring_length": [220, 270],
                },
            )
        )

        # set a random seed to ensure the network lengths are always the same
        # during testing
        random.seed(1)

        # test case 2
        env = AVClosedMultiAgentEnv(
            env_params=self.env_params_closed,
            sim_params=self.sim_params,
            network=self.network_closed
        )

        # reset the network several times and check its number of vehicles
        self.assertEqual(env.k.network.length(), 260.4)
        env.reset()
        self.assertEqual(env.k.network.length(), 228.4)
        env.reset()
        self.assertEqual(env.k.network.length(), 268.4)

    def test_open_env(self):
        """Validate the functionality of the AVOpenMultiAgentEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2. that the inflow rate of vehicles is properly modified in between
           resets
        """
        # test case 1
        self.assertTrue(
            test_additional_params(
                env_class=AVOpenMultiAgentEnv,
                sim_params=self.sim_params,
                network=self.network_open,
                additional_params={
                    "max_accel": 3,
                    "stopping_penalty": True,
                    "acceleration_penalty": True,
                    "use_follower_stopper": True,
                    "obs_frames": 5,
                    "inflows": [1000, 2000],
                    "rl_penetration": 0.1,
                    "num_rl": 5,
                    "control_range": [500, 2500],
                    "warmup_path": None,
                },
            )
        )

        # set a random seed to ensure the network lengths are always the same
        # during testing
        random.seed(1)

        # test case 2
        env = AVOpenMultiAgentEnv(
            env_params=self.env_params_open,
            sim_params=self.sim_params,
            network=self.network_open
        )

        # reset the network several times and check its inflow rate
        inflows = env.net_params.inflows.get()
        for inflow_i in inflows:
            veh_type = inflow_i["vtype"]
            expected_rate = 2114 if veh_type == "human" else 100
            self.assertAlmostEqual(inflow_i["vehsPerHour"], expected_rate)

        env.reset()
        inflows = env.net_params.inflows.get()
        for inflow_i in inflows:
            veh_type = inflow_i["vtype"]
            expected_rate = 1023.3 if veh_type == "human" else 113.7
            self.assertAlmostEqual(inflow_i["vehsPerHour"], expected_rate)

        env.reset()
        inflows = env.net_params.inflows.get()
        for inflow_i in inflows:
            veh_type = inflow_i["vtype"]
            expected_rate = 1680.3 if veh_type == "human" else 186.7
            self.assertAlmostEqual(inflow_i["vehsPerHour"], expected_rate)


class TestPoint2D(unittest.TestCase):
    """Test the functionality of features in envs/point2d.py."""

    def setUp(self):
        self.env_cls = Point2DEnv
        self.env_params = {
            'render_dt_msec': 0,
            'action_l2norm_penalty': 0,
            'render_onscreen': False,
            'render_size': 32,
            'reward_type': "dense",
            'action_scale': 1.0,
            'target_radius': 0.60,
            'boundary_dist': 4,
            'ball_radius': 0.50,
            'walls': None,
            'fixed_goal': None,
            'randomize_position_on_reset': True,
            'images_are_rgb': False,
            'show_goal': True,
            'images_in_obs': True,
        }

    def test_reset(self):
        """Validate the functionality of the current_context method.

        This also tests the current_context, sample_position, and sample_goals
        methods.

        This test attempts to reset the environment and read the current
        context term and initial positions. This is done for two cases:

        1. fixed_goal = None
        2. fixed_goal = [0, 1]
        """
        np.random.seed(0)

        # test case 1
        params = deepcopy(self.env_params)
        params['fixed_goal'] = None
        env = self.env_cls(**params)

        self.assertEqual(env.current_context, None)
        np.testing.assert_almost_equal(env._position, np.array([0, 0]))
        env.reset()
        np.testing.assert_almost_equal(env.current_context,
                                       np.array([0.390508, 1.7215149]))
        np.testing.assert_almost_equal(env._position,
                                       np.array([0.822107, 0.3590655]))

        # test case 2
        params = deepcopy(self.env_params)
        params['fixed_goal'] = [0, 1]
        env = self.env_cls(**params)

        self.assertEqual(env.current_context, None)
        np.testing.assert_almost_equal(env._position, np.array([0, 0]))
        env.reset()
        np.testing.assert_almost_equal(env.current_context,
                                       np.array([0, 1]))
        np.testing.assert_almost_equal(env._position,
                                       np.array([-0.6107616,  1.1671529]))

    def test_step(self):
        """Validate the functionality of the step method.

        This also tests the get_obs and compute_rewards methods.

        The step method is used and the ouput is evaluated for the following
        cases:

        1. not using images
        2. using images
        """
        # test case 1
        params = deepcopy(self.env_params)
        params['images_in_obs'] = False
        env = self.env_cls(**params)

        obs = env.reset()

        np.testing.assert_almost_equal(
            obs,
            np.array([3.7093021, -0.9324678])
        )

        obs, reward, done, _ = env.step(np.array([1, 1]))

        np.testing.assert_almost_equal(
            obs,
            np.array([4., 0.0675322])
        )

        self.assertAlmostEqual(reward, -5.445004580312284)
        self.assertEqual(done, False)

        # test case 2
        params = deepcopy(self.env_params)
        params['images_in_obs'] = True
        env = self.env_cls(**params)

        obs = env.reset()

        self.assertEqual(obs.shape[0], 1026)
        np.testing.assert_almost_equal(
            obs[-2:],
            np.array([0.54435649, 3.40477311])
        )

        obs, reward, done, _ = env.step(np.array([1, 1]))

        self.assertEqual(obs.shape[0], 1026)
        np.testing.assert_almost_equal(
            obs[-2:],
            np.array([1.5443565, 4.])
        )

        self.assertAlmostEqual(reward, -3.850633885880888)
        self.assertEqual(done, False)

    def test_get_goal(self):
        """Validate the functionality of the get_goal method."""
        np.random.seed(0)

        # Initialize the environment.
        env = self.env_cls(**deepcopy(self.env_params))

        # After first reset.
        env.reset()
        np.testing.assert_almost_equal(env.get_goal(), [0.390508, 1.7215149])

        # After second reset.
        env.reset()
        np.testing.assert_almost_equal(env.get_goal(), [-0.6107616, 1.1671529])

    def test_true_model(self):
        """Validate the functionality of the true_model method."""
        # Initialize the environment.
        env = self.env_cls(**deepcopy(self.env_params))

        # Test the method.
        s_t = np.array([0, 1, 2, 3])
        a_t = np.array([0, -10, -1, 5])
        np.testing.assert_almost_equal(env.true_model(s_t, a_t), [0, 0, 1, 4])

    def test_true_states(self):
        """Validate the functionality of the true_states method."""
        # Initialize the environment.
        env = self.env_cls(**deepcopy(self.env_params))

        # Test the method.
        s_t = np.array([0, 1, 2, 3])
        a_t = [np.array([1, 1, 1, 1]), np.array([0, -10, -1, 5])]
        np.testing.assert_almost_equal(
            env.true_states(s_t, a_t),
            [[0, 1, 2, 3], [1, 2, 3, 4], [1, 1, 2, 4]])


class TestRingNonFlow(unittest.TestCase):
    """Test the functionality of features in ring_nonflow.py."""

    def setUp(self):
        self._init_parameters = dict(
            length=260,
            num_vehicles=22,
            dt=0.2,
            horizon=1500,
            gen_emission=False,
            rl_ids=[0],
            warmup_steps=0,
            initial_state=None,
            sims_per_step=1,
            maddpg=False,
        )

        self._initial_state_path = os.path.join(
            hbaselines_config.PROJECT_PATH,
            "hbaselines/envs/mixed_autonomy/envs/initial_states/ring-v0.json"
        )
        with open(self._initial_state_path, "r") as fp:
            self._initial_state = json.load(fp)

    def test_base_env(self):
        """Validate the functionality of the RingEnv class.

        This tests checks that expected outputs are returned for the following
        methods:

        1. action_space
        2. observation_space
        3. get_state
        4. compute_reward
        """
        # Create the environment.
        env = RingEnv(**self._init_parameters)

        # test case 3
        self.assertEqual(env.get_state(), [])

        # test case 4
        self.assertEqual(env.compute_reward(action=None), 0)

    def test_single_agent_env(self):
        """Validate the functionality of the RingSingleAgentEnv class.

        This tests checks that expected outputs are returned for the following
        methods:

        1. action_space
        2. observation_space
        3. get_state
        4. compute_reward
        """
        # Create the environment.
        init_parameters = deepcopy(self._init_parameters)
        init_parameters["rl_ids"] = [0, 11]
        env = RingSingleAgentEnv(**init_parameters)

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(30)]),
            expected_max=np.array([float("inf") for _ in range(30)]),
            expected_size=30,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1.0 for _ in range(2)]),
            expected_max=np.array([1.0 for _ in range(2)]),
            expected_size=2,
        )

        # test case 3
        env.headways = [5 * i for i in range(22)]
        env.speeds = [i for i in range(22)]
        np.testing.assert_almost_equal(
            env.get_state(),
            [0., 0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.1,
             1.2, 0.55, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        )

        # test case 4
        env.speeds = [10 for _ in range(22)]
        self.assertEqual(env.compute_reward(action=None), 10.0)
        env.speeds = [i for i in range(22)]
        self.assertEqual(env.compute_reward(action=None), 11.025)

    def test_multi_agent_env(self):
        """Validate the functionality of the RingMultiAgentEnv class.

        This tests checks that expected outputs are returned for the following
        methods:

        1. action_space
        2. observation_space
        3. get_state
        4. compute_reward
        5. obs after reset and step when maddpg=True
        """
        set_seed(0)

        # Create the environment.
        init_parameters = deepcopy(self._init_parameters)
        init_parameters["rl_ids"] = [0, 11]
        env = RingMultiAgentEnv(**init_parameters)

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(15)]),
            expected_max=np.array([float("inf") for _ in range(15)]),
            expected_size=15,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1. for _ in range(1)]),
            expected_max=np.array([1. for _ in range(1)]),
            expected_size=1,
        )

        # test case 3
        env.headways = [5 * i for i in range(22)]
        env.speeds = [i for i in range(22)]
        state = env.get_state()
        self.assertEqual(list(state.keys()), [0, 11])
        np.testing.assert_almost_equal(
            state[0],
            [0., 0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        )
        np.testing.assert_almost_equal(
            state[11],
            [1.1, 1.2, 0.55, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        )

        # test case 4
        env.speeds = [10 for _ in range(22)]
        self.assertDictEqual(
            env.compute_reward(action=None), {0: -0.5, 11: -0.5})
        env.speeds = [i for i in range(22)]
        self.assertDictEqual(
            env.compute_reward(action=None), {0: 0.0, 11: -0.605})

        # Create the environment.
        init_parameters = deepcopy(self._init_parameters)
        init_parameters["rl_ids"] = [0, 11]
        init_parameters["maddpg"] = True
        env = RingMultiAgentEnv(**init_parameters)

        # test case 5
        obs = env.reset()
        np.testing.assert_almost_equal(
            obs["obs"][0],
            [0., 0., 0.0681818, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        )
        np.testing.assert_almost_equal(
            obs["obs"][11],
            [0., 0., 0.0681818, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        )
        np.testing.assert_almost_equal(
            obs["all_obs"],
            [0., 0., 0.0681818, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0.0681818, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        )

        obs, _, _, _ = env.step({0: [0], 11: [1]})
        np.testing.assert_almost_equal(
            obs["obs"][0],
            [0.01, 0.0253635, 0.0683355, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0.]
        )
        np.testing.assert_almost_equal(
            obs["obs"][11],
            [0.02, 0.026807, 0.0682499, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0.]
        )
        np.testing.assert_almost_equal(
            obs["all_obs"],
            [0.01, 0.0253635, 0.0683355, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0.02, 0.026807, 0.0682499, 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0.]
        )

    def test_set_length(self):
        """Validates the functionality of the _set_length method.

        This is done for the following cases

        1. length = 260
        2. length = [260, 270]
        3. length = [260, 265, 270]
        """
        # Set a random seed.
        set_seed(0)

        # Create the environment.
        env = RingEnv(**self._init_parameters)

        # test case 1
        self.assertEqual(env._set_length(260), 260)

        # test case 2
        self.assertEqual(env._set_length([260, 270]), 266)

        # test case 3
        self.assertEqual(env._set_length([260, 265, 270]), 265)

    def test_set_initial_state(self):
        """Validates the functionality of the _set_initial_state method.

        This is done for the following cases

        1. initial_state = None
        2. initial_state = "random"
        3. initial_state = < some appropriate list >
        """
        # Set a random seed.
        set_seed(0)

        # Create the environment.
        env = RingEnv(**self._init_parameters)

        # test case 1
        pos, vel = env._set_initial_state(
            length=260,
            num_vehicles=22,
            initial_state=None,
            min_gap=0.5
        )
        np.testing.assert_almost_equal(pos, [260 / 22 * i for i in range(22)])
        np.testing.assert_almost_equal(vel, [0 for _ in range(22)])

        # test case 2
        pos, vel = env._set_initial_state(
            length=260,
            num_vehicles=22,
            initial_state="random",
            min_gap=0.5
        )
        np.testing.assert_almost_equal(
            pos,
            [2.81035724, 15.37401209, 23.11097266, 69.79837112, 80.88801711,
             88.32462237, 106.51639385, 114.23876244, 120.28507705,
             128.45819399, 138.78410927, 150.27928172, 165.41132193,
             179.66378838, 187.04978029, 193.58304043, 203.73415853,
             214.43168861, 222.95644711, 233.15793272, 243.94912371,
             251.52794957]
        )
        np.testing.assert_almost_equal(vel, [0 for _ in range(22)])

        # test case 3
        pos, vel = env._set_initial_state(
            length=260,
            num_vehicles=22,
            initial_state=self._initial_state,
            min_gap=0.5
        )
        np.testing.assert_almost_equal(
            pos,
            [4.26916339, 26.26884054, 45.00283694, 59.21748779, 69.62140292,
             77.50811521, 84.4383123, 91.32026691, 98.20029183, 105.08299618,
             111.96267019, 118.84624248, 125.72564607, 132.72740364,
             140.53168409, 149.69677999, 160.49310896, 173.06216771,
             187.49368115, 203.83776384, 222.12540662, 242.34888505]
        )
        np.testing.assert_almost_equal(
            vel,
            [11.21739964, 10.07556411, 6.87465934, 3.77013913, 1.44293824,
             0.1846424, 0., 0., 0., 0., 0., 0., 0., 0.27799439, 1.10959298,
             2.20639645, 3.44200324, 4.75554959, 6.11640064, 7.50645556,
             8.90890994, 10.27611265]
        )

    def test_update_state(self):
        """Validates the functionality of the _update_state method.

        An initial state and action is passed to the method, and the output is
        checked to match expected values.
        """
        # Create the environment.
        env = RingEnv(**self._init_parameters)

        new_pos, new_vel = env._update_state(
            pos=np.array([0, 5, 10]),
            vel=np.array([0, 1, 2]),
            accel=np.array([1., 1., -1.])
        )

        np.testing.assert_almost_equal(new_pos, [0.02, 5.22, 10.38])
        np.testing.assert_almost_equal(new_vel, [0.2, 1.2, 1.8])

    def test_compute_headway(self):
        """Validates the functionality of the _compute_headway method.

        Positions are passed to the vehicles and the output is checked to match
        expected values.
        """
        # Create the environment.
        env = RingEnv(**self._init_parameters)

        env.positions = np.array([6 * i for i in range(22)])

        np.testing.assert_almost_equal(
            env._compute_headway(),
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 129.]
        )

    def test_reset(self):
        """Validates the functionality of the reset method.

        The positions, speeds, and network length are checked after the resets
        for the following cases:

        1. initial_state = None,      length = 260
        2. initial_state = None,      length = [260, 270]
        3. initial_state = some file, length = ...
        """
        # Set a random seed.
        set_seed(0)

        # test case 1
        init_parameters = deepcopy(self._init_parameters)
        init_parameters["length"] = 260
        env = RingEnv(**init_parameters)
        env.reset()

        self.assertEqual(env.length, 260)
        np.testing.assert_almost_equal(
            env.positions,
            [0.0, 11.818181818181818, 23.636363636363637, 35.45454545454545,
             47.27272727272727, 59.09090909090909, 70.9090909090909,
             82.72727272727273, 94.54545454545455, 106.36363636363636,
             118.18181818181819, 130.0, 141.8181818181818, 153.63636363636363,
             165.45454545454547, 177.27272727272728, 189.0909090909091,
             200.9090909090909, 212.72727272727272, 224.54545454545456,
             236.36363636363637, 248.1818181818182]
        )

        # test case 2
        init_parameters = deepcopy(self._init_parameters)
        init_parameters["length"] = [260, 270]
        env = RingEnv(**init_parameters)
        env.reset()

        self.assertEqual(env.length, 266)
        np.testing.assert_almost_equal(
            env.positions,
            [0.0, 12.090909090909092, 24.181818181818183, 36.27272727272727,
             48.36363636363637, 60.45454545454546, 72.54545454545455,
             84.63636363636364, 96.72727272727273, 108.81818181818183,
             120.90909090909092, 133.0, 145.0909090909091, 157.1818181818182,
             169.27272727272728, 181.36363636363637, 193.45454545454547,
             205.54545454545456, 217.63636363636365, 229.72727272727275,
             241.81818181818184, 253.90909090909093]
        )

        # test case 3
        init_parameters = deepcopy(self._init_parameters)
        init_parameters["initial_state"] = self._initial_state_path
        env = RingEnv(**init_parameters)
        env.reset()

        self.assertEqual(env.length, 315)
        np.testing.assert_almost_equal(
            env.positions,
            [1.456621610141724, 22.862776568370673, 46.21101268634284,
             71.39880838706915, 97.71478456842716, 122.44440175240561,
             142.69988721415794, 158.10207484483797, 169.40700574488392,
             177.813765780532, 184.85316440616074, 191.73418780156828,
             198.61487854549048, 205.49650594236655, 212.39333987008553,
             219.78140927565013, 228.35065388253238, 238.46151307933877,
             250.2987552109015, 263.96146567591495, 279.5196555147506,
             297.01011465470043]
        )


###############################################################################
#                              Utility methods                                #
###############################################################################

def test_additional_params(env_class,
                           sim_params,
                           network,
                           additional_params):
    """Test that the environment raises an Error in any param is missing.

    Parameters
    ----------
    env_class : flow.envs.Env
        the environment class. Used to try to instantiate the environment.
    sim_params : flow.core.params.SumoParams
        sumo-specific parameters
    network : flow.networks.Network
        network that works for the environment
    additional_params : dict
        the valid and required additional parameters for the environment in
        EnvParams

    Returns
    -------
    bool
        True if the test passed, False otherwise
    """
    for key in additional_params.keys():
        # remove one param from the additional_params dict
        new_add = additional_params.copy()
        del new_add[key]

        try:
            env_class(
                sim_params=sim_params,
                network=network,
                env_params=EnvParams(additional_params=new_add)
            )
            # if no KeyError is raised, the test has failed, so return False
            return False  # pragma: no cover
        except KeyError:
            # if a KeyError is raised, test the next param
            pass

    # make sure that add all params does not lead to an error
    try:
        env_class(
            sim_params=sim_params,
            network=network,
            env_params=EnvParams(additional_params=additional_params.copy())
        )
    except KeyError:  # pragma: no cover
        # if a KeyError is raised, the test has failed, so return False
        return False  # pragma: no cover

    # if removing all additional params led to KeyErrors, the test has passed,
    # so return True
    return True


def test_space(gym_space, expected_size, expected_min, expected_max):
    """Test that an action or observation space is the correct size and bounds.

    Parameters
    ----------
    gym_space : gym.spaces.Box
        gym space object to be tested
    expected_size : int
        expected size
    expected_min : float or array_like
        expected minimum value(s)
    expected_max : float or array_like
        expected maximum value(s)

    Returns
    -------
    bool
        True if the test passed, False otherwise
    """
    assert gym_space.shape[0] == expected_size, \
        "{}, {}".format(gym_space.shape[0], expected_size)
    np.testing.assert_almost_equal(gym_space.high, expected_max, decimal=4)
    np.testing.assert_almost_equal(gym_space.low, expected_min, decimal=4)


def test_observed(env_class,
                  sim_params,
                  network,
                  env_params,
                  expected_observed):
    """Test that the observed vehicles in the environment are as expected.

    Parameters
    ----------
    env_class : flow.envs.Env class
        the environment class. Used to instantiate the environment.
    sim_params : flow.core.params.SumoParams
        sumo-specific parameters
    network : flow.networks.Network
        network that works for the environment
    env_params : flow.core.params.EnvParams
        environment-specific parameters
    expected_observed : array_like
        expected list of observed vehicles

    Returns
    -------
    bool
        True if the test passed, False otherwise
    """
    env = env_class(sim_params=sim_params,
                    network=network,
                    env_params=env_params)
    env.reset()
    env.step(None)
    env.additional_command()
    test_mask = np.all(
        np.array(env.k.vehicle.get_observed_ids()) ==
        np.array(expected_observed)
    )
    env.terminate()

    return test_mask


def set_seed(seed):
    """Set the random seed for testing purposes."""
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    unittest.main()
