"""Contains tests for the contained environments."""
import unittest
import numpy as np
import random

from hbaselines.envs.efficient_hrl.maze_env_utils import line_intersect, \
    point_distance, construct_maze
from hbaselines.envs.efficient_hrl.envs import AntMaze
from hbaselines.envs.efficient_hrl.envs import AntFall
from hbaselines.envs.efficient_hrl.envs import AntPush
from hbaselines.envs.efficient_hrl.envs import AntFourRooms
from hbaselines.envs.hac.env_utils import check_validity
from hbaselines.envs.hac.envs import UR5, Pendulum
from hbaselines.envs.mixed_autonomy import FlowEnv


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


class TestMixedAutonomy(unittest.TestCase):
    """Test the functionality of features in envs/mixed_autonomy."""

    def test_single_agent_ring(self):
        # create the base environment
        env = FlowEnv(
            env_name="ring",
            env_params={
                "num_automated": 1,
                "horizon": 1500,
                "simulator": "traci",
                "multiagent": False
            },
            version=0
        )
        env.reset()

        # test observation space
        test_space(
            env.observation_space,
            expected_min=np.array([-np.inf for _ in range(3)]),
            expected_max=np.array([np.inf for _ in range(3)]),
            expected_size=3,
        )

        # test action space
        test_space(
            env.action_space,
            expected_min=np.array([-1]),
            expected_max=np.array([1]),
            expected_size=1,
        )

        # kill the environment
        env.wrapped_env.terminate()

    def test_multi_agent_ring(self):
        # create the base environment
        env = FlowEnv(
            env_name="ring",
            env_params={
                "num_automated": 1,
                "horizon": 1500,
                "simulator": "traci",
                "multiagent": True
            },
            multiagent=True,
            shared=False,
            version=1
        )
        env.reset()

        # test the agent IDs.
        self.assertListEqual(env.agents, ["rl_0_0"])

        # test observation space
        test_space(
            env.observation_space["rl_0_0"],
            expected_min=np.array([-5 for _ in range(3)]),
            expected_max=np.array([5 for _ in range(3)]),
            expected_size=3,
        )

        # test action space
        test_space(
            env.action_space["rl_0_0"],
            expected_min=np.array([-1]),
            expected_max=np.array([1]),
            expected_size=1,
        )

        # kill the environment
        env.wrapped_env.terminate()

        # create the environment with multiple automated vehicles
        env = FlowEnv(
            env_name="ring",
            env_params={
                "num_automated": 4,
                "horizon": 1500,
                "simulator": "traci",
                "multiagent": True
            },
            multiagent=True,
            shared=True,
        )
        env.reset()

        # test the agent IDs.
        self.assertListEqual(
            env.agents, ["rl_0_0", "rl_1_0", "rl_2_0", "rl_3_0"])

        # test observation space
        test_space(
            env.observation_space,
            expected_min=np.array([-5 for _ in range(3)]),
            expected_max=np.array([5 for _ in range(3)]),
            expected_size=3,
        )

        # test action space
        test_space(
            env.action_space,
            expected_min=np.array([-1]),
            expected_max=np.array([1]),
            expected_size=1,
        )

        # kill the environment
        env.wrapped_env.terminate()

    def test_single_agent_figure_eight(self):
        # create the base environment
        env = FlowEnv(
            env_name="figure_eight",
            env_params={
                "num_automated": 1,
                "horizon": 1500,
                "simulator": "traci",
                "multiagent": False
            },
            version=0
        )
        env.reset()

        # test observation space
        test_space(
            env.observation_space,
            expected_min=np.array([0 for _ in range(28)]),
            expected_max=np.array([1 for _ in range(28)]),
            expected_size=28,
        )

        # test action space
        test_space(
            env.action_space,
            expected_min=np.array([-3]),
            expected_max=np.array([3]),
            expected_size=1,
        )

        # kill the environment
        env.wrapped_env.terminate()

        # create the environment with multiple automated vehicles
        env = FlowEnv(
            env_name="figure_eight",
            env_params={
                "num_automated": 14,
                "horizon": 1500,
                "simulator": "traci",
                "multiagent": False
            },
            version=1
        )
        env.reset()

        # test observation space
        test_space(
            env.observation_space,
            expected_min=np.array([0 for _ in range(28)]),
            expected_max=np.array([1 for _ in range(28)]),
            expected_size=28,
        )

        # test action space
        test_space(
            env.action_space,
            expected_min=np.array([-3 for _ in range(14)]),
            expected_max=np.array([3 for _ in range(14)]),
            expected_size=14,
        )

        # kill the environment
        env.wrapped_env.terminate()

    def test_multi_agent_figure_eight(self):
        # create the base environment
        env = FlowEnv(
            env_name="figure_eight",
            env_params={
                "num_automated": 1,
                "horizon": 1500,
                "simulator": "traci",
                "multiagent": True
            },
            version=0
        )
        env.reset()

        # test observation space
        pass  # TODO

        # test action space
        pass  # TODO

        # kill the environment
        env.wrapped_env.terminate()

        # create the environment with multiple automated vehicles
        env = FlowEnv(
            env_name="figure_eight",
            env_params={
                "num_automated": 14,
                "horizon": 1500,
                "simulator": "traci",
                "multiagent": True
            },
            version=1
        )
        env.reset()

        # test observation space
        pass  # TODO

        # test action space
        pass  # TODO

        # kill the environment
        env.wrapped_env.terminate()

    def test_single_agent_merge(self):
        # create version 0 of the environment
        env = FlowEnv(
            env_name="merge",
            env_params={
                "exp_num": 0,
                "horizon": 6000,
                "simulator": "traci",
                "multiagent": False
            },
            version=0
        )
        env.reset()

        # test observation space
        test_space(
            env.observation_space,
            expected_min=np.array([0 for _ in range(25)]),
            expected_max=np.array([1 for _ in range(25)]),
            expected_size=25,
        )

        # test action space
        test_space(
            env.action_space,
            expected_min=np.array([-1.5 for _ in range(5)]),
            expected_max=np.array([1.5 for _ in range(5)]),
            expected_size=5,
        )

        # kill the environment
        env.wrapped_env.terminate()

        # create version 1 of the environment
        env = FlowEnv(
            env_name="merge",
            env_params={
                "exp_num": 1,
                "horizon": 6000,
                "simulator": "traci",
                "multiagent": False
            },
            version=1
        )
        env.reset()

        # test observation space
        test_space(
            env.observation_space,
            expected_min=np.array([0 for _ in range(65)]),
            expected_max=np.array([1 for _ in range(65)]),
            expected_size=65,
        )

        # test action space
        test_space(
            env.action_space,
            expected_min=np.array([-1.5 for _ in range(13)]),
            expected_max=np.array([1.5 for _ in range(13)]),
            expected_size=13,
        )

        # kill the environment
        env.wrapped_env.terminate()

        # create version 2 of the environment
        env = FlowEnv(
            env_name="merge",
            env_params={
                "exp_num": 2,
                "horizon": 6000,
                "simulator": "traci",
                "multiagent": False
            },
            version=2
        )
        env.reset()

        # test observation space
        test_space(
            env.observation_space,
            expected_min=np.array([0 for _ in range(85)]),
            expected_max=np.array([1 for _ in range(85)]),
            expected_size=85,
        )

        # test action space
        test_space(
            env.action_space,
            expected_min=np.array([-1.5 for _ in range(17)]),
            expected_max=np.array([1.5 for _ in range(17)]),
            expected_size=17,
        )

        # kill the environment
        env.wrapped_env.terminate()

    # def test_multi_agent_merge(self):
    #     # create version 0 of the environment
    #     env = FlowEnv(
    #         env_name="merge",
    #         env_params={
    #             "exp_num": 0,
    #             "horizon": 6000,
    #             "simulator": "traci",
    #             "multiagent": True
    #         },
    #         version=0
    #     )
    #     env.reset()
    #
    #     # test observation space
    #     pass  # TODO
    #
    #     # test action space
    #     pass  # TODO
    #
    #     # kill the environment
    #     env.wrapped_env.terminate()
    #
    #     # create version 1 of the environment
    #     env = FlowEnv(
    #         env_name="merge",
    #         env_params={
    #             "exp_num": 1,
    #             "horizon": 6000,
    #             "simulator": "traci",
    #             "multiagent": True
    #         },
    #         version=1
    #     )
    #     env.reset()
    #
    #     # test observation space
    #     pass  # TODO
    #
    #     # test action space
    #     pass  # TODO
    #
    #     # kill the environment
    #     env.wrapped_env.terminate()
    #
    #     # create version 2 of the environment
    #     env = FlowEnv(
    #         env_name="merge",
    #         env_params={
    #             "exp_num": 2,
    #             "horizon": 6000,
    #             "simulator": "traci",
    #             "multiagent": True
    #         },
    #         version=2
    #     )
    #     env.reset()
    #
    #     # test observation space
    #     pass  # TODO
    #
    #     # test action space
    #     pass  # TODO
    #
    #     # kill the environment
    #     env.wrapped_env.terminate()


def test_space(gym_space, expected_size, expected_min, expected_max):
    """Test the shape and bounds of an action or observation space.

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
    """
    assert gym_space.shape[0] == expected_size, \
        "{}, {}".format(gym_space.shape[0], expected_size)
    np.testing.assert_almost_equal(gym_space.high, expected_max, decimal=4)
    np.testing.assert_almost_equal(gym_space.low, expected_min, decimal=4)


if __name__ == '__main__':
    unittest.main()
