"""Contains tests for the contained environments."""
import unittest
import numpy as np
import random
from copy import deepcopy

from flow.core.params import EnvParams
from flow.controllers import IDMController

from hbaselines.envs.efficient_hrl.maze_env_utils import line_intersect, \
    point_distance, construct_maze
from hbaselines.envs.efficient_hrl.envs import AntMaze
from hbaselines.envs.efficient_hrl.envs import AntFall
from hbaselines.envs.efficient_hrl.envs import AntPush
from hbaselines.envs.efficient_hrl.envs import AntFourRooms

from hbaselines.envs.hac.env_utils import check_validity
from hbaselines.envs.hac.envs import UR5, Pendulum

from hbaselines.envs.mixed_autonomy.params.ring \
    import get_flow_params as ring
from hbaselines.envs.mixed_autonomy.params.highway \
    import get_flow_params as highway

from hbaselines.envs.mixed_autonomy.envs.av import AVEnv
from hbaselines.envs.mixed_autonomy.envs.av import AVClosedEnv
from hbaselines.envs.mixed_autonomy.envs.av \
    import CLOSED_ENV_PARAMS as SA_CLOSED_ENV_PARAMS
from hbaselines.envs.mixed_autonomy.envs.av \
    import OPEN_ENV_PARAMS as SA_OPEN_ENV_PARAMS
from hbaselines.envs.mixed_autonomy.envs.av_multi import AVMultiAgentEnv
from hbaselines.envs.mixed_autonomy.envs.av_multi \
    import CLOSED_ENV_PARAMS as MA_CLOSED_ENV_PARAMS
from hbaselines.envs.mixed_autonomy.envs.imitation import AVImitationEnv
from hbaselines.envs.mixed_autonomy.envs.imitation import AVClosedImitationEnv
from hbaselines.envs.mixed_autonomy.envs.imitation import AVOpenImitationEnv
from hbaselines.envs.point2d import Point2DEnv
from hbaselines.utils.env_util import create_env


class TestEfficientHRLEnvironments(unittest.TestCase):
    """Test the environments in envs/efficient_hrl/."""

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
    3. the reward function returns its expected value

    For the multi-agent environments, we also perform the following tests:

    4. the agent IDs match their expected values
    """

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
            expected_min=np.array([-float("inf") for _ in range(25)]),
            expected_max=np.array([float("inf") for _ in range(25)]),
            expected_size=25,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1 for _ in range(5)]),
            expected_max=np.array([1 for _ in range(5)]),
            expected_size=5,
        )

        # test case 3
        self.assertAlmostEqual(
            env.wrapped_env.compute_reward(np.array([1] * 5), fail=False),
            8.760423525122517
        )

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
            expected_min=np.array([-float("inf") for _ in range(5)]),
            expected_max=np.array([float("inf") for _ in range(5)]),
            expected_size=5,
        )

        # test case 2
        test_space(
            env.action_space["rl_0_0"],
            expected_min=np.array([-1]),
            expected_max=np.array([1]),
            expected_size=1,
        )

        # test case 3
        self.assertDictEqual(
            env.wrapped_env.compute_reward(
                {key: np.array([1]) for key in env.agents},
                fail=False,
            ),
            {'rl_0_0': 8.484502989441593,
             'rl_0_1': 8.484502989441593,
             'rl_0_2': 8.484502989441593,
             'rl_0_3': 8.484502989441593,
             'rl_0_4': 8.484502989441593}
        )

        # test case 4
        self.assertListEqual(
            sorted(env.agents),
            ['rl_0_0', 'rl_0_1', 'rl_0_2', 'rl_0_3', 'rl_0_4']
        )

        # kill the environment
        env.wrapped_env.terminate()

    # ======================================================================= #
    #                                 ring-v1                                 #
    # ======================================================================= #

    def test_single_agent_ring_v1(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("ring-v1")

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(25)]),
            expected_max=np.array([float("inf") for _ in range(25)]),
            expected_size=25,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1 for _ in range(5)]),
            expected_max=np.array([1 for _ in range(5)]),
            expected_size=5,
        )

        # test case 3
        self.assertAlmostEqual(
            env.wrapped_env.compute_reward(np.array([1] * 5), fail=False),
            8.760423525122517
        )

        # kill the environment
        env.wrapped_env.terminate()

    def test_multi_agent_ring_v1(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("multiagent-ring-v1")

        # test case 1
        test_space(
            env.observation_space["rl_0_0"],
            expected_min=np.array([-float("inf") for _ in range(5)]),
            expected_max=np.array([float("inf") for _ in range(5)]),
            expected_size=5,
        )

        # test case 2
        test_space(
            env.action_space["rl_0_0"],
            expected_min=np.array([-1]),
            expected_max=np.array([1]),
            expected_size=1,
        )

        # test case 3
        self.assertDictEqual(
            env.wrapped_env.compute_reward(
                {key: np.array([1]) for key in env.agents},
                fail=False,
            ),
            {'rl_0_0': 8.484502989441593,
             'rl_0_1': 8.484502989441593,
             'rl_0_2': 8.484502989441593,
             'rl_0_3': 8.484502989441593,
             'rl_0_4': 8.484502989441593}
        )

        # test case 4
        self.assertListEqual(
            sorted(env.agents),
            ['rl_0_0', 'rl_0_1', 'rl_0_2', 'rl_0_3', 'rl_0_4']
        )

        # kill the environment
        env.wrapped_env.terminate()

    # ======================================================================= #
    #                                 ring-v2                                 #
    # ======================================================================= #

    def test_single_agent_ring_v2(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("ring-v2")

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(25)]),
            expected_max=np.array([float("inf") for _ in range(25)]),
            expected_size=25,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1 for _ in range(5)]),
            expected_max=np.array([1 for _ in range(5)]),
            expected_size=5,
        )

        # test case 3
        self.assertAlmostEqual(
            env.wrapped_env.compute_reward(np.array([1] * 5), fail=False),
            13.760423525122517
        )

        # kill the environment
        env.wrapped_env.terminate()

    def test_multi_agent_ring_v2(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("multiagent-ring-v2")

        # test case 1
        test_space(
            env.observation_space["rl_0_0"],
            expected_min=np.array([-float("inf") for _ in range(5)]),
            expected_max=np.array([float("inf") for _ in range(5)]),
            expected_size=5,
        )

        # test case 2
        test_space(
            env.action_space["rl_0_0"],
            expected_min=np.array([-1]),
            expected_max=np.array([1]),
            expected_size=1,
        )

        # test case 3
        self.assertDictEqual(
            env.wrapped_env.compute_reward(
                {key: np.array([1]) for key in env.agents},
                fail=False,
            ),
            {'rl_0_0': 13.484502989441593,
             'rl_0_1': 13.484502989441593,
             'rl_0_2': 13.484502989441593,
             'rl_0_3': 13.484502989441593,
             'rl_0_4': 13.484502989441593}
        )

        # test case 4
        self.assertListEqual(
            sorted(env.agents),
            ['rl_0_0', 'rl_0_1', 'rl_0_2', 'rl_0_3', 'rl_0_4']
        )

        # kill the environment
        env.wrapped_env.terminate()

    # ======================================================================= #
    #                             ring-imitation                              #
    # ======================================================================= #

    def test_single_agent_ring_imitation(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("ring-imitation")

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(25)]),
            expected_max=np.array([float("inf") for _ in range(25)]),
            expected_size=25,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-1 for _ in range(5)]),
            expected_max=np.array([1 for _ in range(5)]),
            expected_size=5,
        )

        # test case 3
        self.assertAlmostEqual(
            env.wrapped_env.compute_reward(np.array([1] * 5), fail=False),
            13.760423525122517
        )

        # kill the environment
        env.wrapped_env.terminate()

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

        # test case 3
        self.assertAlmostEqual(
            env.wrapped_env.compute_reward(np.array([1.5] * 5), fail=False),
            0
        )

        # kill the environment
        env.wrapped_env.terminate()

    def test_multi_agent_merge_v0(self):
        # set a random seed
        set_seed(0)

        # create the environment
        # env, _ = create_env("multiagent-merge-v0")

        # test case 1
        pass  # TODO

        # test case 2
        pass  # TODO

        # test case 3
        pass  # TODO

        # test case 4
        pass  # TODO

        # kill the environment
        # env.wrapped_env.terminate()

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

        # test case 3
        self.assertAlmostEqual(
            env.wrapped_env.compute_reward(np.array([1.5] * 13), fail=False),
            0
        )

        # kill the environment
        env.wrapped_env.terminate()

    def test_multi_agent_merge_v1(self):
        # set a random seed
        set_seed(0)

        # create the environment
        # env, _ = create_env("multiagent-merge-v1")

        # test case 1
        pass  # TODO

        # test case 2
        pass  # TODO

        # test case 3
        pass  # TODO

        # test case 4
        pass  # TODO

        # kill the environment
        # env.wrapped_env.terminate()

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

        # test case 3
        self.assertAlmostEqual(
            env.wrapped_env.compute_reward(np.array([1.5] * 17), fail=False),
            0
        )

        # kill the environment
        env.wrapped_env.terminate()

    def test_multi_agent_merge_v2(self):
        # set a random seed
        set_seed(0)

        # create the environment
        # env, _ = create_env("multiagent-merge-v2")

        # test case 1
        pass  # TODO

        # test case 2
        pass  # TODO

        # test case 3
        pass  # TODO

        # test case 4
        pass  # TODO

        # kill the environment
        # env.wrapped_env.terminate()

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
            expected_min=np.array([-float("inf") for _ in range(50)]),
            expected_max=np.array([float("inf") for _ in range(50)]),
            expected_size=50,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-0.5 for _ in range(10)]),
            expected_max=np.array([0.5 for _ in range(10)]),
            expected_size=10,
        )

        # test case 3
        self.assertAlmostEqual(
            env.wrapped_env.compute_reward(np.array([0.5] * 10), fail=False),
            -8.019758424524031
        )

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
            expected_min=np.array([-float("inf") for _ in range(5)]),
            expected_max=np.array([float("inf") for _ in range(5)]),
            expected_size=5,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-0.5 for _ in range(1)]),
            expected_max=np.array([0.5 for _ in range(1)]),
            expected_size=1,
        )

        # test case 3
        self.assertDictEqual(
            env.wrapped_env.compute_reward(
                {key: np.array([0.5]) for key in env.agents},
                fail=False,
            ),
            {'rl_highway_inflow_10.18': -8.378975313435493,
             'rl_highway_inflow_10.19': -8.378975313435493,
             'rl_highway_inflow_10.20': -8.378975313435493,
             'rl_highway_inflow_10.21': -8.378975313435493,
             'rl_highway_inflow_10.22': -8.378975313435493,
             'rl_highway_inflow_10.23': -8.378975313435493,
             'rl_highway_inflow_10.24': -8.378975313435493,
             'rl_highway_inflow_10.25': -8.378975313435493,
             'rl_highway_inflow_10.26': -8.378975313435493,
             'rl_highway_inflow_10.27': -8.378975313435493}
        )

        # test case 4
        self.assertListEqual(
            sorted(env.agents),
            ['rl_highway_inflow_10.18',
             'rl_highway_inflow_10.19',
             'rl_highway_inflow_10.20',
             'rl_highway_inflow_10.21',
             'rl_highway_inflow_10.22',
             'rl_highway_inflow_10.23',
             'rl_highway_inflow_10.24',
             'rl_highway_inflow_10.25',
             'rl_highway_inflow_10.26',
             'rl_highway_inflow_10.27']
        )

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
            expected_min=np.array([-float("inf") for _ in range(50)]),
            expected_max=np.array([float("inf") for _ in range(50)]),
            expected_size=50,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-0.5 for _ in range(10)]),
            expected_max=np.array([0.5 for _ in range(10)]),
            expected_size=10,
        )

        # test case 3
        self.assertAlmostEqual(
            env.wrapped_env.compute_reward(np.array([0.5] * 10), fail=False),
            1.9802415754759695
        )

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
            expected_min=np.array([-float("inf") for _ in range(5)]),
            expected_max=np.array([float("inf") for _ in range(5)]),
            expected_size=5,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-0.5 for _ in range(1)]),
            expected_max=np.array([0.5 for _ in range(1)]),
            expected_size=1,
        )

        # test case 3
        self.assertDictEqual(
            env.wrapped_env.compute_reward(
                {key: np.array([0.5]) for key in env.agents},
                fail=False,
            ),
            {'rl_highway_inflow_10.18': 1.6210246865645077,
             'rl_highway_inflow_10.19': 1.6210246865645077,
             'rl_highway_inflow_10.20': 1.6210246865645077,
             'rl_highway_inflow_10.21': 1.6210246865645077,
             'rl_highway_inflow_10.22': 1.6210246865645077,
             'rl_highway_inflow_10.23': 1.6210246865645077,
             'rl_highway_inflow_10.24': 1.6210246865645077,
             'rl_highway_inflow_10.25': 1.6210246865645077,
             'rl_highway_inflow_10.26': 1.6210246865645077,
             'rl_highway_inflow_10.27': 1.6210246865645077}
        )

        # test case 4
        self.assertListEqual(
            sorted(env.agents),
            ['rl_highway_inflow_10.18',
             'rl_highway_inflow_10.19',
             'rl_highway_inflow_10.20',
             'rl_highway_inflow_10.21',
             'rl_highway_inflow_10.22',
             'rl_highway_inflow_10.23',
             'rl_highway_inflow_10.24',
             'rl_highway_inflow_10.25',
             'rl_highway_inflow_10.26',
             'rl_highway_inflow_10.27']
        )

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
            expected_min=np.array([-float("inf") for _ in range(50)]),
            expected_max=np.array([float("inf") for _ in range(50)]),
            expected_size=50,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-0.5 for _ in range(10)]),
            expected_max=np.array([0.5 for _ in range(10)]),
            expected_size=10,
        )

        # test case 3
        self.assertAlmostEqual(
            env.wrapped_env.compute_reward(np.array([0.5] * 10), fail=False),
            4.4802415754759695
        )

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
            expected_min=np.array([-float("inf") for _ in range(5)]),
            expected_max=np.array([float("inf") for _ in range(5)]),
            expected_size=5,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-0.5 for _ in range(1)]),
            expected_max=np.array([0.5 for _ in range(1)]),
            expected_size=1,
        )

        # test case 3
        self.assertDictEqual(
            env.wrapped_env.compute_reward(
                {key: np.array([0.5]) for key in env.agents},
                fail=False,
            ),
            {'rl_highway_inflow_10.18': 4.121024686564508,
             'rl_highway_inflow_10.19': 4.121024686564508,
             'rl_highway_inflow_10.20': 4.121024686564508,
             'rl_highway_inflow_10.21': 4.121024686564508,
             'rl_highway_inflow_10.22': 4.121024686564508,
             'rl_highway_inflow_10.23': 4.121024686564508,
             'rl_highway_inflow_10.24': 4.121024686564508,
             'rl_highway_inflow_10.25': 4.121024686564508,
             'rl_highway_inflow_10.26': 4.121024686564508,
             'rl_highway_inflow_10.27': 4.121024686564508}
        )

        # test case 4
        self.assertListEqual(
            sorted(env.agents),
            ['rl_highway_inflow_10.18',
             'rl_highway_inflow_10.19',
             'rl_highway_inflow_10.20',
             'rl_highway_inflow_10.21',
             'rl_highway_inflow_10.22',
             'rl_highway_inflow_10.23',
             'rl_highway_inflow_10.24',
             'rl_highway_inflow_10.25',
             'rl_highway_inflow_10.26',
             'rl_highway_inflow_10.27']
        )

        # kill the environment
        env.wrapped_env.terminate()

    # ======================================================================= #
    #                            highway-imitation                            #
    # ======================================================================= #

    def test_single_agent_highway_imitation(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("highway-imitation")

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(50)]),
            expected_max=np.array([float("inf") for _ in range(50)]),
            expected_size=50,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-0.5 for _ in range(10)]),
            expected_max=np.array([0.5 for _ in range(10)]),
            expected_size=10,
        )

        # test case 3
        self.assertAlmostEqual(
            env.wrapped_env.compute_reward(np.array([0.5] * 10), fail=False),
            4.4802415754759695
        )

        # kill the environment
        env.wrapped_env.terminate()

    # ======================================================================= #
    #                                 i210-v0                                 #
    # ======================================================================= #

    def test_single_agent_i210_v0(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("i210-v0")

        # test case 1
        test_space(
            env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(250)]),
            expected_max=np.array([float("inf") for _ in range(250)]),
            expected_size=250,
        )

        # test case 2
        test_space(
            env.action_space,
            expected_min=np.array([-0.5 for _ in range(50)]),
            expected_max=np.array([0.5 for _ in range(50)]),
            expected_size=50,
        )

        # test case 3
        self.assertAlmostEqual(
            env.wrapped_env.compute_reward(np.array([0.5] * 50), fail=False),
            -21.939782231314958
        )

        # kill the environment
        env.wrapped_env.terminate()

    def test_multi_agent_i210_v0(self):
        # set a random seed
        set_seed(0)

        # create the environment
        env, _ = create_env("multiagent-i210-v0")

        # test case 1
        test_space(
            env.observation_space["lane_0"],
            expected_min=np.array([-float("inf") for _ in range(50)]),
            expected_max=np.array([float("inf") for _ in range(50)]),
            expected_size=50,
        )

        # test case 2
        test_space(
            env.action_space["lane_0"],
            expected_min=np.array([-0.5 for _ in range(10)]),
            expected_max=np.array([0.5 for _ in range(10)]),
            expected_size=10,
        )

        # test case 3
        self.assertDictEqual(
            env.wrapped_env.compute_reward(
                {key: np.array([0.5] * 10) for key in env.agents},
                fail=False,
            ),
            {'lane_0': 4.986116706600313,
             'lane_1': 5.065488111440597,
             'lane_2': 4.646797630457543,
             'lane_3': 5.0177871084707295,
             'lane_4': 4.954239592286896}
        )

        # test case 4
        self.assertListEqual(
            sorted(env.agents),
            ['lane_0', 'lane_1', 'lane_2', 'lane_3', 'lane_4']
        )

        # kill the environment
        env.wrapped_env.terminate()

    # ======================================================================= #
    #                                 i210-v1                                 #
    # ======================================================================= #

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
    #         expected_min=np.array([-float("inf") for _ in range(250)]),
    #         expected_max=np.array([float("inf") for _ in range(250)]),
    #         expected_size=250,
    #     )
    #
    #     # test case 2
    #     test_space(
    #         env.action_space,
    #         expected_min=np.array([-0.5 for _ in range(50)]),
    #         expected_max=np.array([0.5 for _ in range(50)]),
    #         expected_size=50,
    #     )
    #
    #     # test case 3
    #     self.assertAlmostEqual(
    #         env.wrapped_env.compute_reward(np.array([0.5] * 50), fail=False),
    #         6.791321863445345
    #     )
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
    #         expected_min=np.array([-float("inf") for _ in range(50)]),
    #         expected_max=np.array([float("inf") for _ in range(50)]),
    #         expected_size=50,
    #     )
    #
    #     # test case 2
    #     test_space(
    #         env.action_space["lane_0"],
    #         expected_min=np.array([-0.5 for _ in range(10)]),
    #         expected_max=np.array([0.5 for _ in range(10)]),
    #         expected_size=10,
    #     )
    #
    #     # test case 3
    #     self.assertDictEqual(
    #         env.wrapped_env.compute_reward(
    #             {key: np.array([0.5] * 10) for key in env.agents},
    #             fail=False,
    #         ),
    #         {}
    #     )
    #
    #     # test case 4
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
    #         expected_min=np.array([-float("inf") for _ in range(250)]),
    #         expected_max=np.array([float("inf") for _ in range(250)]),
    #         expected_size=250,
    #     )
    #
    #     # test case 2
    #     test_space(
    #         env.action_space,
    #         expected_min=np.array([-0.5 for _ in range(50)]),
    #         expected_max=np.array([0.5 for _ in range(50)]),
    #         expected_size=50,
    #     )
    #
    #     # test case 3
    #     self.assertAlmostEqual(
    #         env.wrapped_env.compute_reward(np.array([0.5] * 50), fail=False),
    #         11.005385206055989
    #     )
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
    #         expected_min=np.array([-float("inf") for _ in range(50)]),
    #         expected_max=np.array([float("inf") for _ in range(50)]),
    #         expected_size=50,
    #     )
    #
    #     # test case 2
    #     test_space(
    #         env.action_space["lane_0"],
    #         expected_min=np.array([-0.5 for _ in range(10)]),
    #         expected_max=np.array([0.5 for _ in range(10)]),
    #         expected_size=10,
    #     )
    #
    #     # test case 3
    #     self.assertDictEqual(
    #         env.wrapped_env.compute_reward(
    #             {key: np.array([0.5] * 10) for key in env.agents},
    #             fail=False,
    #         ),
    #         {}
    #     )
    #
    #     # test case 4
    #     self.assertListEqual(
    #         sorted(env.agents),
    #         ['lane_0', 'lane_1', 'lane_2', 'lane_3', 'lane_4']
    #     )

        # kill the environment
        env.wrapped_env.terminate()


class TestAV(unittest.TestCase):
    """Tests the automated vehicles single agent environments."""

    def setUp(self):
        self.sim_params = deepcopy(ring(
            fixed_density=True,
            stopping_penalty=True,
            acceleration_penalty=True,
        ))["sim"]
        self.sim_params.render = False

        # for AVClosedEnv
        flow_params_closed = deepcopy(ring(
            fixed_density=True,
            stopping_penalty=True,
            acceleration_penalty=True,
        ))

        self.network_closed = flow_params_closed["network"](
            name="test_closed",
            vehicles=flow_params_closed["veh"],
            net_params=flow_params_closed["net"],
        )
        self.env_params_closed = flow_params_closed["env"]
        self.env_params_closed.additional_params = SA_CLOSED_ENV_PARAMS.copy()

        # for AVOpenEnv
        pass  # TODO

    def test_base_env(self):
        """Validate the functionality of the AVEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2. that the observation space matches its expected values
           a. for the single lane case
           b. for the multi-lane case
        3. that the action space matches its expected values
           a. for the single lane case
           b. for the multi-lane case
        4. that the observed vehicle IDs after a reset matches its expected
           values
           a. for the single lane case
           b. for the multi-lane case
        """
        # test case 1
        self.assertTrue(
            test_additional_params(
                env_class=AVEnv,
                sim_params=self.sim_params,
                network=self.network_closed,
                additional_params={
                    "max_accel": 3,
                    "max_decel": 3,
                    "target_velocity": 30,
                    "stopping_penalty": True,
                    "acceleration_penalty": True,
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

        # Create a multi-lane environment.
        env_multi = None  # TODO
        del env_multi

        # test case 2.a
        test_space(
            gym_space=env_single.observation_space,
            expected_size=5 * env_single.initial_vehicles.num_rl_vehicles,
            expected_min=-float("inf"),
            expected_max=float("inf"),
        )

        # test case 2.b
        pass  # TODO

        # test case 3.a
        test_space(
            gym_space=env_single.action_space,
            expected_size=env_single.initial_vehicles.num_rl_vehicles,
            expected_min=-1,
            expected_max=1,
        )

        # test case 3.b
        pass  # TODO

        # test case 4.a
        self.assertTrue(
            test_observed(
                env_class=AVEnv,
                sim_params=self.sim_params,
                network=self.network_closed,
                env_params=self.env_params_closed,
                expected_observed=['rl_0_1', 'rl_0_2', 'rl_0_3', 'rl_0_4',
                                   'human_0_0', 'human_0_44', 'rl_0_0']
            )
        )

        # test case 4.b
        pass  # TODO

    def test_closed_env(self):
        """Validate the functionality of the AVClosedEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2, that the number of vehicles is properly modified in between resets
        """
        # test case 1
        self.assertTrue(
            test_additional_params(
                env_class=AVClosedEnv,
                sim_params=self.sim_params,
                network=self.network_closed,
                additional_params={
                    "max_accel": 3,
                    "max_decel": 3,
                    "target_velocity": 30,
                    "stopping_penalty": True,
                    "acceleration_penalty": True,
                    "num_vehicles": [50, 75],
                    "even_distribution": False,
                    "sort_vehicles": True,
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

        # reset the network several times and check its length
        self.assertEqual(env.k.vehicle.num_vehicles, 50)
        self.assertEqual(env.k.vehicle.num_rl_vehicles, 5)
        env.reset()
        self.assertEqual(env.k.vehicle.num_vehicles, 54)
        self.assertEqual(env.k.vehicle.num_rl_vehicles, 5)
        env.reset()
        self.assertEqual(env.k.vehicle.num_vehicles, 58)
        self.assertEqual(env.k.vehicle.num_rl_vehicles, 5)

    def test_open_env(self):
        """Validate the functionality of the AVOpenEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2, that the inflow rate of vehicles is properly modified in between
           resets
        """
        # test case 1
        pass  # TODO

        # test case 2
        pass  # TODO


class TestAVMulti(unittest.TestCase):
    """Tests the automated vehicles multi-agent environments."""

    def setUp(self):
        self.sim_params = deepcopy(ring(
            fixed_density=True,
            stopping_penalty=True,
            acceleration_penalty=True,
            multiagent=True,
        ))["sim"]
        self.sim_params.render = False

        # for AVClosedEnv
        flow_params_closed = deepcopy(ring(
            fixed_density=True,
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
        self.env_params_closed.additional_params = MA_CLOSED_ENV_PARAMS.copy()

        # for AVOpenEnv
        pass  # TODO

    def test_base_env(self):
        """Validate the functionality of the AVMultiAgentEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2. that the observation space matches its expected values
           a. for the single lane case
           b. for the multi-lane case
        3. that the action space matches its expected values
           a. for the single lane case
           b. for the multi-lane case
        4. that the observed vehicle IDs after a reset matches its expected
           values
           a. for the single lane case
           b. for the multi-lane case
        """
        # test case 1
        self.assertTrue(
            test_additional_params(
                env_class=AVMultiAgentEnv,
                sim_params=self.sim_params,
                network=self.network_closed,
                additional_params={
                    "max_accel": 3,
                    "max_decel": 3,
                    "target_velocity": 30,
                    "stopping_penalty": True,
                    "acceleration_penalty": True,
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

        # Create a multi-lane environment.
        env_multi = None  # TODO
        del env_multi

        # test case 2.a
        test_space(
            gym_space=env_single.observation_space,
            expected_size=env_single.initial_vehicles.num_rl_vehicles,
            expected_min=-float("inf"),
            expected_max=float("inf"),
        )

        # test case 2.b
        pass  # TODO

        # test case 3.a
        test_space(
            gym_space=env_single.action_space,
            expected_size=1,
            expected_min=-1,
            expected_max=1,
        )

        # test case 3.b
        pass  # TODO

        # test case 4.a
        self.assertTrue(
            test_observed(
                env_class=AVMultiAgentEnv,
                sim_params=self.sim_params,
                network=self.network_closed,
                env_params=self.env_params_closed,
                expected_observed=['rl_0_1', 'rl_0_2', 'rl_0_3', 'rl_0_4',
                                   'human_0_0', 'human_0_44', 'rl_0_0']
            )
        )

        # test case 4.b
        pass  # TODO

    def test_closed_env(self):
        """Validate the functionality of the AVClosedMultiAgentEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2, that the number of vehicles is properly modified in between resets
        """
        # test case 1
        pass  # TODO

        # test case 2
        pass  # TODO

    def test_open_env(self):
        """Validate the functionality of the AVOpenMultiAgentEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2, that the inflow rate of vehicles is properly modified in between
           resets
        """
        # test case 1
        pass  # TODO

        # test case 2
        pass  # TODO


class TestAVImitation(unittest.TestCase):
    """Tests the automated vehicles single agent imitation environments."""

    def setUp(self):
        self.sim_params = deepcopy(ring(
            fixed_density=True,
            stopping_penalty=True,
            acceleration_penalty=True,
            imitation=True,
        ))["sim"]
        self.sim_params.render = False

        # for AVClosedEnv
        flow_params_closed = deepcopy(ring(
            fixed_density=True,
            stopping_penalty=True,
            acceleration_penalty=True,
            imitation=True,
        ))

        self.network_closed = flow_params_closed["network"](
            name="test_closed",
            vehicles=flow_params_closed["veh"],
            net_params=flow_params_closed["net"],
        )
        self.env_params_closed = flow_params_closed["env"]
        self.env_params_closed.additional_params = SA_CLOSED_ENV_PARAMS.copy()

        # for AVOpenEnv
        flow_params_open = deepcopy(highway(
            fixed_boundary=True,
            stopping_penalty=True,
            acceleration_penalty=True,
            imitation=True,
        ))

        self.network_open = flow_params_open["network"](
            name="test_open",
            vehicles=flow_params_open["veh"],
            net_params=flow_params_open["net"],
        )
        self.env_params_open = flow_params_open["env"]
        self.env_params_open.additional_params = SA_OPEN_ENV_PARAMS.copy()

    def test_base_env(self):
        """Validate the functionality of the AVImitationEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2. that the observation space matches its expected values
           a. for the single lane case
           b. for the multi-lane case
        3. that the action space matches its expected values
           a. for the single lane case
           b. for the multi-lane case
        4. that the observed vehicle IDs after a reset matches its expected
           values
           a. for the single lane case
           b. for the multi-lane case
        5. the the query_expert method returns the expected values
        """
        env_params = deepcopy(self.env_params_closed)
        env_params.additional_params["expert_model"] = (IDMController, {
            "a": 0.3,
            "b": 2.0,
        })

        # test case 1
        self.assertTrue(
            test_additional_params(
                env_class=AVImitationEnv,
                sim_params=self.sim_params,
                network=self.network_closed,
                additional_params={
                    "max_accel": 3,
                    "max_decel": 3,
                    "target_velocity": 30,
                    "stopping_penalty": True,
                    "acceleration_penalty": True,
                    "expert_model": (IDMController, {
                        "a": 0.3,
                        "b": 2.0,
                    }),
                },
            )
        )

        # Set a random seed.
        random.seed(0)
        np.random.seed(0)

        # Create a single lane environment.
        env_single = AVImitationEnv(
            env_params=env_params,
            sim_params=self.sim_params,
            network=self.network_closed
        )

        # Create a multi-lane environment.
        env_multi = None  # TODO
        del env_multi

        # test case 2.a
        test_space(
            gym_space=env_single.observation_space,
            expected_size=25,
            expected_min=-float("inf"),
            expected_max=float("inf"),
        )

        # test case 2.b
        pass  # TODO

        # test case 3.a
        test_space(
            gym_space=env_single.action_space,
            expected_size=5,
            expected_min=-1,
            expected_max=1,
        )

        # test case 3.b
        pass  # TODO

        # test case 4.a
        self.assertTrue(
            test_observed(
                env_class=AVImitationEnv,
                sim_params=self.sim_params,
                network=self.network_closed,
                env_params=env_params,
                expected_observed=['rl_0_1', 'rl_0_2', 'rl_0_3', 'rl_0_4',
                                   'human_0_0', 'human_0_44', 'rl_0_0']
            )
        )

        # test case 4.b
        pass  # TODO

        # test case 5
        env = AVImitationEnv(
            sim_params=self.sim_params,
            network=self.network_closed,
            env_params=env_params,
        )
        env.reset()

        np.testing.assert_almost_equal(
            env.query_expert(None),
            [-0.0041661, -0.0080797, 0.0279598, 0.0662519, 0.1447145]
        )

    def test_closed_env(self):
        """Validate the functionality of the AVClosedImitationEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2. that the number of vehicles is properly modified in between resets
        3. the the query_expert method returns the expected values
        """
        env_params = deepcopy(self.env_params_closed)
        env_params.additional_params["expert_model"] = (IDMController, {
            "a": 0.3,
            "b": 2.0,
        })

        # test case 1
        self.assertTrue(
            test_additional_params(
                env_class=AVClosedImitationEnv,
                sim_params=self.sim_params,
                network=self.network_closed,
                additional_params={
                    "max_accel": 3,
                    "max_decel": 3,
                    "target_velocity": 30,
                    "stopping_penalty": True,
                    "acceleration_penalty": True,
                    "num_vehicles": [50, 75],
                    "even_distribution": False,
                    "sort_vehicles": True,
                    "expert_model": (IDMController, {
                        "a": 0.3,
                        "b": 2.0,
                    }),
                },
            )
        )

        # set a random seed to ensure the network lengths are always the same
        # during testing
        set_seed(1)

        # test case 2
        env = AVClosedImitationEnv(
            env_params=env_params,
            sim_params=self.sim_params,
            network=self.network_closed
        )

        # reset the network several times and check its length
        self.assertEqual(env.k.vehicle.num_vehicles, 50)
        self.assertEqual(env.num_rl, 5)
        env.reset()
        self.assertEqual(env.k.vehicle.num_vehicles, 54)
        self.assertEqual(env.num_rl, 5)
        env.reset()
        self.assertEqual(env.k.vehicle.num_vehicles, 58)
        self.assertEqual(env.num_rl, 5)

        # test case 3
        env = AVClosedImitationEnv(
            sim_params=self.sim_params,
            network=self.network_closed,
            env_params=env_params,
        )
        env.reset()

        np.testing.assert_almost_equal(
            env.query_expert(None),
            [-0.0021488, -0.0365337, -0.0196091, -0.0057063, 0.0409672]
        )

    def test_open_env(self):
        """Validate the functionality of the AVOpenImitationEnv class.

        This tests checks for the following cases:

        1. that additional_env_params cause an Exception to be raised if not
           properly passed
        2. that the inflow rate is properly modified in between resets
        3. the the query_expert method returns the expected values
        """
        env_params = deepcopy(self.env_params_open)
        env_params.additional_params["expert_model"] = (IDMController, {
            "a": 0.3,
            "b": 2.0,
        })

        # test case 1
        self.assertTrue(
            test_additional_params(
                env_class=AVOpenImitationEnv,
                sim_params=self.sim_params,
                network=self.network_open,
                additional_params={
                    "max_accel": 3,
                    "max_decel": 3,
                    "target_velocity": 30,
                    "stopping_penalty": True,
                    "acceleration_penalty": True,
                    "inflows": [1000, 2000],
                    "rl_penetration": 0.1,
                    "num_rl": 5,
                    "control_range": [500, 700],
                    "expert_model": (IDMController, {
                        "a": 0.3,
                        "b": 2.0,
                    }),
                },
            )
        )

        # set a random seed to ensure the network lengths are always the same
        # during testing
        random.seed(1)

        # test case 2
        pass  # TODO

        # test case 3
        env = AVOpenImitationEnv(
            sim_params=self.sim_params,
            network=self.network_open,
            env_params=env_params,
        )
        env.reset()

        np.testing.assert_almost_equal(
            env.query_expert(None),
            [0.0604367, 0.1412521, 0.1402381, 0.1775216, 0.0628473]
        )


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

    def test_position_inside_wall(self):
        """Validate the functionality of the position_inside_wall method.

        TODO
        """
        pass  # TODO

    def test_get_goal(self):
        """Validate the functionality of the get_goal method.

        TODO
        """
        pass  # TODO

    def test_get_image(self):
        """Validate the functionality of the get_image method.

        TODO
        """
        pass  # TODO

    def test_draw(self):
        """Validate the functionality of the draw method.

        TODO
        """
        pass  # TODO

    def test_true_model(self):
        """Validate the functionality of the true_model method.

        TODO
        """
        pass  # TODO

    def test_true_states(self):
        """Validate the functionality of the true_states method.

        TODO
        """
        pass  # TODO

    def test_plot_trajectory(self):
        """Validate the functionality of the plot_trajectory method.

        TODO
        """
        pass  # TODO


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
    env_class : flow.envs.Env type
        blank
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
        blank
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
