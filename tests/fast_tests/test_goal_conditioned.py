"""Tests for the policies in the hbaselines/goal_conditioned subdirectory."""
import unittest
import numpy as np
import tensorflow as tf
from gym.spaces import Box

from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy as \
    TD3GoalConditionedPolicy
from hbaselines.goal_conditioned.sac import GoalConditionedPolicy as \
    SACGoalConditionedPolicy
from hbaselines.algorithms.off_policy import SAC_PARAMS, TD3_PARAMS
from hbaselines.algorithms.off_policy import GOAL_CONDITIONED_PARAMS


class TestBaseGoalConditionedPolicy(unittest.TestCase):
    """Test GoalConditionedPolicy in hbaselines/goal_conditioned/base.py."""

    def setUp(self):
        self.policy_params = {
            'sess': tf.compat.v1.Session(),
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'ob_space': Box(low=-2, high=2, shape=(2,)),
            'co_space': Box(low=-3, high=3, shape=(2,)),
            'layers': None,
            'verbose': 0,
        }
        self.policy_params.update(TD3_PARAMS.copy())
        self.policy_params.update(GOAL_CONDITIONED_PARAMS.copy())

    def tearDown(self):
        self.policy_params['sess'].close()
        del self.policy_params

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_store_transition(self):
        """Check the functionality of the store_transition() method.

        This method is tested for the following cases:

        1. hindsight = False, relative_goals = False
        2. hindsight = False, relative_goals = True
        3. hindsight = True,  relative_goals = False
        4. hindsight = True,  relative_goals = True
        """
        # =================================================================== #
        #                             test case 1                             #
        # =================================================================== #

        policy_params = self.policy_params.copy()
        policy_params['relative_goals'] = False
        policy_params['hindsight'] = False
        policy_params['subgoal_testing_rate'] = 1
        policy_params['meta_period'] = 4
        policy_params['batch_size'] = 2
        policy = TD3GoalConditionedPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        policy._meta_action = [np.array([5, 5])]

        for i in range(4):
            obs0 = np.array([i for _ in range(2)])
            context0 = np.array([i for _ in range(3)])
            action = np.array([i for _ in range(1)])
            reward = i
            obs1 = np.array([i+1 for _ in range(2)])
            context1 = np.array([i for _ in range(3)])
            done, is_final_step, evaluate = False, False, False

            policy.store_transition(
                obs0=obs0,
                context0=context0,
                action=action,
                reward=reward,
                obs1=obs1,
                context1=context1,
                done=done,
                is_final_step=is_final_step,
                evaluate=evaluate
            )

        obs_t = policy.replay_buffer._obs_t[0]
        action_t = policy.replay_buffer._action_t[0]
        reward = policy.replay_buffer._reward_t[0]
        done = policy.replay_buffer._done_t[0]

        # check the various attributes
        self.assertTrue(
            all(all(obs_t[i] ==
                    [np.array([0, 0]),
                     np.array([1, 1]),
                     np.array([2, 2]),
                     np.array([3, 3]),
                     np.array([4, 4])][i])
                for i in range(len(obs_t)))
        )

        for i in range(len(action_t)):
            self.assertTrue(
                all(all(action_t[i][j] ==
                        [[np.array([5, 5]),
                          np.array([5, 5]),
                          np.array([5, 5]),
                          np.array([5, 5]),
                          np.array([5, 5])],
                         [np.array([0]),
                          np.array([1]),
                          np.array([2]),
                          np.array([3])]][i][j])
                    for j in range(len(action_t[i])))
            )

        self.assertEqual(reward,
                         [[6], [-5.656854249501219, -4.24264068713107,
                                -2.8284271247638677, -1.4142135624084504]])

        self.assertEqual(done,
                         [False, False, False, False])

    def test_store_transition_2(self):
        policy_params = self.policy_params.copy()
        policy_params['relative_goals'] = True
        policy_params['hindsight'] = False
        policy_params['subgoal_testing_rate'] = 1
        policy_params['meta_period'] = 4
        policy_params['batch_size'] = 2
        policy = TD3GoalConditionedPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        policy._meta_action = [np.array([5, 5])]

        for i in range(4):
            obs0 = np.array([i for _ in range(2)])
            context0 = np.array([i for _ in range(3)])
            action = np.array([i for _ in range(1)])
            reward = i
            obs1 = np.array([i+1 for _ in range(2)])
            context1 = np.array([i for _ in range(3)])
            done, is_final_step, evaluate = False, False, False

            policy.store_transition(
                obs0=obs0,
                context0=context0,
                action=action,
                reward=reward,
                obs1=obs1,
                context1=context1,
                done=done,
                is_final_step=is_final_step,
                evaluate=evaluate
            )

        obs_t = policy.replay_buffer._obs_t[0]
        action_t = policy.replay_buffer._action_t[0]
        reward = policy.replay_buffer._reward_t[0]
        done = policy.replay_buffer._done_t[0]

        # check the various attributes
        self.assertTrue(
            all(all(obs_t[i] ==
                    [np.array([0, 0]), np.array([1, 1]), np.array([2, 2]),
                     np.array([3, 3]), np.array([4, 4])][i])
                for i in range(len(obs_t)))
        )

        for i in range(len(action_t)):
            self.assertTrue(
                all(all(action_t[i][j] ==
                        [[np.array([5, 5]), np.array([5, 5]), np.array([5, 5]),
                          np.array([5, 5]), np.array([4, 4])],
                         [np.array([0]), np.array([1]), np.array([2]),
                          np.array([3])]][i][j])
                    for j in range(len(action_t[i])))
            )

        self.assertEqual(reward,
                         [[6], [-5.656854249501219, -5.656854249501219,
                                -5.656854249501219, -5.656854249501219]])

        self.assertEqual(done, [False, False, False, False])

    def test_store_transition_3(self):
        policy_params = self.policy_params.copy()
        policy_params['relative_goals'] = False
        policy_params['hindsight'] = True
        policy_params['subgoal_testing_rate'] = 1
        policy_params['meta_period'] = 4
        policy_params['batch_size'] = 2
        policy = TD3GoalConditionedPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        policy._meta_action = [np.array([5, 5])]

        for i in range(4):
            obs0 = np.array([i for _ in range(2)])
            context0 = np.array([i for _ in range(3)])
            action = np.array([i for _ in range(1)])
            reward = i
            obs1 = np.array([i+1 for _ in range(2)])
            context1 = np.array([i for _ in range(3)])
            done, is_final_step, evaluate = False, False, False

            policy.store_transition(
                obs0=obs0,
                context0=context0,
                action=action,
                reward=reward,
                obs1=obs1,
                context1=context1,
                done=done,
                is_final_step=is_final_step,
                evaluate=evaluate
            )

        # unchanged sample
        obs_t = policy.replay_buffer._obs_t[0]
        action_t = policy.replay_buffer._action_t[0]
        reward_t = policy.replay_buffer._reward_t[0]
        done_t = policy.replay_buffer._done_t[0]

        # check the various attributes
        self.assertTrue(
            all(all(obs_t[i] ==
                    [np.array([0, 0]),
                     np.array([1, 1]),
                     np.array([2, 2]),
                     np.array([3, 3]),
                     np.array([4, 4])][i])
                for i in range(len(obs_t)))
        )

        for i in range(len(action_t)):
            self.assertTrue(
                all(all(action_t[i][j] ==
                        [[np.array([5, 5]),
                          np.array([5, 5]),
                          np.array([5, 5]),
                          np.array([5, 5]),
                          np.array([5, 5])],
                         [np.array([0]),
                          np.array([1]),
                          np.array([2]),
                          np.array([3])]][i][j])
                    for j in range(len(action_t[i])))
            )

        self.assertEqual(reward_t,
                         [[6], [-5.656854249501219, -4.24264068713107,
                                -2.8284271247638677, -1.4142135624084504]])

        self.assertEqual(done_t, [False, False, False, False])

        # hindsight sample
        obs_t = policy.replay_buffer._obs_t[1]
        action_t = policy.replay_buffer._action_t[1]
        reward_t = policy.replay_buffer._reward_t[1]
        done_t = policy.replay_buffer._done_t[1]

        # check the various attributes
        self.assertTrue(
            all(all(obs_t[i] ==
                    [np.array([0, 0]),
                     np.array([1, 1]),
                     np.array([2, 2]),
                     np.array([3, 3]),
                     np.array([4, 4])][i])
                for i in range(len(obs_t)))
        )

        for i in range(len(action_t)):
            self.assertTrue(
                all(all(action_t[i][j] ==
                        [[np.array([4, 4]),
                          np.array([4, 4]),
                          np.array([4, 4]),
                          np.array([4, 4]),
                          np.array([4, 4])],
                         [np.array([0]),
                          np.array([1]),
                          np.array([2]),
                          np.array([3])]][i][j])
                    for j in range(len(action_t[i])))
            )

        self.assertEqual(reward_t,
                         [[6], [-4.24264068713107, -2.8284271247638677,
                                -1.4142135624084504, -1e-05]])

        self.assertEqual(done_t, [False, False, False, False])

    def test_store_transition_4(self):
        policy_params = self.policy_params.copy()
        policy_params['relative_goals'] = True
        policy_params['hindsight'] = True
        policy_params['subgoal_testing_rate'] = 1
        policy_params['meta_period'] = 4
        policy_params['batch_size'] = 2
        policy = TD3GoalConditionedPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        policy._meta_action = [np.array([5, 5])]

        for i in range(4):
            obs0 = np.array([i for _ in range(2)])
            context0 = np.array([i for _ in range(3)])
            action = np.array([i for _ in range(1)])
            reward = i
            obs1 = np.array([i+1 for _ in range(2)])
            context1 = np.array([i for _ in range(3)])
            done, is_final_step, evaluate = False, False, False

            policy.store_transition(
                obs0=obs0,
                context0=context0,
                action=action,
                reward=reward,
                obs1=obs1,
                context1=context1,
                done=done,
                is_final_step=is_final_step,
                evaluate=evaluate
            )

        # unchanged sample
        obs_t = policy.replay_buffer._obs_t[0]
        action_t = policy.replay_buffer._action_t[0]
        reward = policy.replay_buffer._reward_t[0]
        done = policy.replay_buffer._done_t[0]

        # check the various attributes
        self.assertTrue(
            all(all(obs_t[i] ==
                    [np.array([0, 0]),
                     np.array([1, 1]),
                     np.array([2, 2]),
                     np.array([3, 3]),
                     np.array([4, 4])][i])
                for i in range(len(obs_t)))
        )

        for i in range(len(action_t)):
            self.assertTrue(
                all(all(action_t[i][j] ==
                        [[np.array([5, 5]),
                          np.array([5, 5]),
                          np.array([5, 5]),
                          np.array([5, 5]),
                          np.array([4, 4])],
                         [np.array([0]),
                          np.array([1]),
                          np.array([2]),
                          np.array([3])]][i][j])
                    for j in range(len(action_t[i])))
            )

        self.assertEqual(reward,
                         [[6], [-5.656854249501219, -5.656854249501219,
                                -5.656854249501219, -5.656854249501219]])

        self.assertEqual(done, [False, False, False, False])

        # hindsight sample
        obs_t = policy.replay_buffer._obs_t[1]
        action_t = policy.replay_buffer._action_t[1]
        reward_t = policy.replay_buffer._reward_t[1]
        done_t = policy.replay_buffer._done_t[1]

        # check the various attributes
        self.assertTrue(
            all(all(obs_t[i] ==
                    [np.array([0, 0]),
                     np.array([1, 1]),
                     np.array([2, 2]),
                     np.array([3, 3]),
                     np.array([4, 4])][i])
                for i in range(len(obs_t)))
        )

        for i in range(len(action_t)):
            self.assertTrue(
                all(all(action_t[i][j] ==
                        [[np.array([4, 4]),
                          np.array([3, 3]),
                          np.array([2, 2]),
                          np.array([1, 1]),
                          np.array([0, 0])],
                         [np.array([0]),
                          np.array([1]),
                          np.array([2]),
                          np.array([3])]][i][j])
                    for j in range(len(action_t[i])))
            )

        self.assertEqual(reward_t,
                         [[6], [-4.24264068713107, -2.8284271247638677,
                                -1.4142135624084504, -1e-05]])

        self.assertEqual(done_t, [False, False, False, False])

    def test_update_meta(self):
        """Validate the functionality of the _update_meta function.

        This is tested for two cases:
        1. level = 0 after 0 steps --> True
        2. level = 1 after 0 steps --> True
        3. level = 0 after 2 steps --> False
        4. level = 1 after 2 steps --> False
        5. level = 0 after 5 steps --> False
        6. level = 1 after 5 steps --> True
        7. level = 0 after 10 steps --> False
        8. level = 1 after 10 steps --> True
        """
        policy_params = self.policy_params.copy()
        policy_params['meta_period'] = 5
        policy_params['num_levels'] = 3
        policy = TD3GoalConditionedPolicy(**policy_params)

        # test case 1
        policy._observations = []
        self.assertEqual(policy._update_meta(0), True)

        # test case 2
        policy._observations = []
        self.assertEqual(policy._update_meta(1), True)

        # test case 3
        policy._observations = [0 for _ in range(2)]
        self.assertEqual(policy._update_meta(0), False)

        # test case 4
        policy._observations = [0 for _ in range(2)]
        self.assertEqual(policy._update_meta(1), False)

        # test case 5
        policy._observations = [0 for _ in range(5)]
        self.assertEqual(policy._update_meta(0), False)

        # test case 6
        policy._observations = [0 for _ in range(5)]
        self.assertEqual(policy._update_meta(1), True)

        # test case 7
        policy._observations = [0 for _ in range(10)]
        self.assertEqual(policy._update_meta(0), False)

        # test case 8
        policy._observations = [0 for _ in range(10)]
        self.assertEqual(policy._update_meta(1), True)

    def test_intrinsic_rewards(self):
        """Validate the functionality of the intrinsic rewards.

        This is done for the following cases:

        1. intrinsic_reward_type = "negative_distance"
        2. intrinsic_reward_type = "scaled_negative_distance"
        3. intrinsic_reward_type = "non_negative_distance"
        4. intrinsic_reward_type = "scaled_non_negative_distance"
        5. intrinsic_reward_type = "exp_negative_distance"
        6. intrinsic_reward_type = "scaled_exp_negative_distance"
        7. intrinsic_reward_type = "error" -> raises ValueError
        """
        # test case 1
        policy_params = self.policy_params.copy()
        policy_params["intrinsic_reward_type"] = "negative_distance"
        policy = TD3GoalConditionedPolicy(**policy_params)

        self.assertAlmostEqual(
            policy.intrinsic_reward_fn(
                states=np.array([1, 2]),
                goals=np.array([3, 2]),
                next_states=np.array([0, 0])
            ),
            -3.6055512754778567
        )

        # Clear the graph.
        del policy
        tf.compat.v1.reset_default_graph()

        # test case 2
        policy_params = self.policy_params.copy()
        policy_params["intrinsic_reward_type"] = "scaled_negative_distance"
        policy = TD3GoalConditionedPolicy(**policy_params)

        self.assertAlmostEqual(
            policy.intrinsic_reward_fn(
                states=np.array([1, 2]),
                goals=np.array([3, 2]),
                next_states=np.array([0, 0])
            ),
            -1.8027756377597297
        )

        # Clear the graph.
        del policy
        tf.compat.v1.reset_default_graph()

        # test case 3
        policy_params = self.policy_params.copy()
        policy_params["intrinsic_reward_type"] = "non_negative_distance"
        policy = TD3GoalConditionedPolicy(**policy_params)

        self.assertAlmostEqual(
            policy.intrinsic_reward_fn(
                states=np.array([1, 2]),
                goals=np.array([3, 2]),
                next_states=np.array([0, 0])
            ),
            2.0513028772015867
        )

        # Clear the graph.
        del policy
        tf.compat.v1.reset_default_graph()

        # test case 4
        policy_params = self.policy_params.copy()
        policy_params["intrinsic_reward_type"] = "scaled_non_negative_distance"
        policy = TD3GoalConditionedPolicy(**policy_params)

        self.assertAlmostEqual(
            policy.intrinsic_reward_fn(
                states=np.array([1, 2]),
                goals=np.array([3, 2]),
                next_states=np.array([0, 0])
            ),
            3.8540785149197134
        )

        # Clear the graph.
        del policy
        tf.compat.v1.reset_default_graph()

        # test case 5
        policy_params = self.policy_params.copy()
        policy_params["intrinsic_reward_type"] = "exp_negative_distance"
        policy = TD3GoalConditionedPolicy(**policy_params)

        self.assertAlmostEqual(
            policy.intrinsic_reward_fn(
                states=np.array([1, 2]),
                goals=np.array([3, 2]),
                next_states=np.array([0, 0])
            ),
            2.2603294067550214e-06
        )

        # Clear the graph.
        del policy
        tf.compat.v1.reset_default_graph()

        # test case 6
        policy_params = self.policy_params.copy()
        policy_params["intrinsic_reward_type"] = "scaled_exp_negative_distance"
        policy = TD3GoalConditionedPolicy(**policy_params)

        self.assertAlmostEqual(
            policy.intrinsic_reward_fn(
                states=np.array([1, 2]),
                goals=np.array([3, 2]),
                next_states=np.array([0, 0])
            ),
            0.03877420782784459
        )

        # Clear the graph.
        del policy
        tf.compat.v1.reset_default_graph()

        # test case 7
        policy_params = self.policy_params.copy()
        policy_params["intrinsic_reward_type"] = "error"
        self.assertRaises(
            ValueError, TD3GoalConditionedPolicy, **policy_params)

    def test_relative_goals(self):
        """Validate the functionality of relative goals.

        This should affect the intrinsic reward function as well as
        transformation from relative goals to absolute goals.
        """
        policy_params = self.policy_params.copy()
        policy_params["relative_goals"] = True
        policy = TD3GoalConditionedPolicy(**policy_params)

        # Test the updated reward function.
        states = np.array([1, 2])
        goals = np.array([4, 5])
        next_states = np.array([7, 8])
        self.assertAlmostEqual(
            policy.intrinsic_reward_fn(states, goals, next_states),
            -2.2360679775221506
        )

    def test_sample_best_meta_action(self):
        """Check the functionality of the _sample_best_meta_action() method."""
        pass  # TODO

    def test_sample(self):
        """Check the functionality of the _sample() method.

        This test checks for the following features:

        1. that the shape of the output candidate goals is correct
        2. that the last few elements are the deterministic components that
           they are expected to be (see method's docstring)
        """
        policy = TD3GoalConditionedPolicy(**self.policy_params)

        # some variables to try on
        states = np.array(
            [[1, 2],
             [3, 4],
             [5, 6],
             [7, 8],
             [9, 10],
             [11, 12],
             [13, 14],
             [15, 16],
             [17, 18],
             [19, 20]]
        )
        next_states = -states
        num_samples = 10
        orig_goals = np.array(
            [[1, 1],
             [1, 1],
             [0, 0],
             [1, 1],
             [1, 1],
             [0, 0],
             [1, 1],
             [1, 1],
             [0, 0],
             [1, 1]]
        )
        samples = policy._sample(states, next_states, orig_goals, num_samples)

        # test case 1
        self.assertTupleEqual(
            samples.shape, (states.shape[0], states.shape[1], num_samples))

        # test case 2
        np.testing.assert_array_almost_equal(
            samples[:, :, -2:].reshape(states.shape[0] * states.shape[1], 2).T,
            np.vstack(
                [np.array([-2] * states.shape[0] * states.shape[1]),
                 orig_goals.flatten()]
            )
        )


class TestTD3GoalConditionedPolicy(unittest.TestCase):
    """Test GoalConditionedPolicy in hbaselines/goal_conditioned/td3.py."""

    def setUp(self):
        self.policy_params = {
            'sess': tf.compat.v1.Session(),
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'ob_space': Box(low=-2, high=2, shape=(2,)),
            'co_space': Box(low=-3, high=3, shape=(2,)),
            'layers': None,
            'verbose': 0,
        }
        self.policy_params.update(TD3_PARAMS.copy())
        self.policy_params.update(GOAL_CONDITIONED_PARAMS.copy())

    def tearDown(self):
        self.policy_params['sess'].close()
        del self.policy_params

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_init_2_levels(self):
        """Validate that the graph and variables are initialized properly."""
        policy_params = self.policy_params.copy()
        policy_params['num_levels'] = 2
        policy = TD3GoalConditionedPolicy(**policy_params)

        # Check that the abstract class has all the required attributes.
        self.assertEqual(policy.meta_period,
                         self.policy_params['meta_period'])
        self.assertEqual(policy.relative_goals,
                         self.policy_params['relative_goals'])
        self.assertEqual(policy.off_policy_corrections,
                         self.policy_params['off_policy_corrections'])
        self.assertEqual(policy.use_fingerprints,
                         self.policy_params['use_fingerprints'])
        self.assertEqual(policy.centralized_value_functions,
                         self.policy_params['centralized_value_functions'])
        self.assertEqual(policy.connected_gradients,
                         self.policy_params['connected_gradients'])
        self.assertEqual(policy.cg_weights,
                         self.policy_params['cg_weights'])

        # Check that all trainable variables have been created in the
        # TensorFlow graph.
        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['level_0/model/pi/fc0/bias:0',
             'level_0/model/pi/fc0/kernel:0',
             'level_0/model/pi/fc1/bias:0',
             'level_0/model/pi/fc1/kernel:0',
             'level_0/model/pi/output/bias:0',
             'level_0/model/pi/output/kernel:0',
             'level_0/model/qf_0/fc0/bias:0',
             'level_0/model/qf_0/fc0/kernel:0',
             'level_0/model/qf_0/fc1/bias:0',
             'level_0/model/qf_0/fc1/kernel:0',
             'level_0/model/qf_0/qf_output/bias:0',
             'level_0/model/qf_0/qf_output/kernel:0',
             'level_0/model/qf_1/fc0/bias:0',
             'level_0/model/qf_1/fc0/kernel:0',
             'level_0/model/qf_1/fc1/bias:0',
             'level_0/model/qf_1/fc1/kernel:0',
             'level_0/model/qf_1/qf_output/bias:0',
             'level_0/model/qf_1/qf_output/kernel:0',
             'level_0/target/pi/fc0/bias:0',
             'level_0/target/pi/fc0/kernel:0',
             'level_0/target/pi/fc1/bias:0',
             'level_0/target/pi/fc1/kernel:0',
             'level_0/target/pi/output/bias:0',
             'level_0/target/pi/output/kernel:0',
             'level_0/target/qf_0/fc0/bias:0',
             'level_0/target/qf_0/fc0/kernel:0',
             'level_0/target/qf_0/fc1/bias:0',
             'level_0/target/qf_0/fc1/kernel:0',
             'level_0/target/qf_0/qf_output/bias:0',
             'level_0/target/qf_0/qf_output/kernel:0',
             'level_0/target/qf_1/fc0/bias:0',
             'level_0/target/qf_1/fc0/kernel:0',
             'level_0/target/qf_1/fc1/bias:0',
             'level_0/target/qf_1/fc1/kernel:0',
             'level_0/target/qf_1/qf_output/bias:0',
             'level_0/target/qf_1/qf_output/kernel:0',
             'level_1/model/pi/fc0/bias:0',
             'level_1/model/pi/fc0/kernel:0',
             'level_1/model/pi/fc1/bias:0',
             'level_1/model/pi/fc1/kernel:0',
             'level_1/model/pi/output/bias:0',
             'level_1/model/pi/output/kernel:0',
             'level_1/model/qf_0/fc0/bias:0',
             'level_1/model/qf_0/fc0/kernel:0',
             'level_1/model/qf_0/fc1/bias:0',
             'level_1/model/qf_0/fc1/kernel:0',
             'level_1/model/qf_0/qf_output/bias:0',
             'level_1/model/qf_0/qf_output/kernel:0',
             'level_1/model/qf_1/fc0/bias:0',
             'level_1/model/qf_1/fc0/kernel:0',
             'level_1/model/qf_1/fc1/bias:0',
             'level_1/model/qf_1/fc1/kernel:0',
             'level_1/model/qf_1/qf_output/bias:0',
             'level_1/model/qf_1/qf_output/kernel:0',
             'level_1/target/pi/fc0/bias:0',
             'level_1/target/pi/fc0/kernel:0',
             'level_1/target/pi/fc1/bias:0',
             'level_1/target/pi/fc1/kernel:0',
             'level_1/target/pi/output/bias:0',
             'level_1/target/pi/output/kernel:0',
             'level_1/target/qf_0/fc0/bias:0',
             'level_1/target/qf_0/fc0/kernel:0',
             'level_1/target/qf_0/fc1/bias:0',
             'level_1/target/qf_0/fc1/kernel:0',
             'level_1/target/qf_0/qf_output/bias:0',
             'level_1/target/qf_0/qf_output/kernel:0',
             'level_1/target/qf_1/fc0/bias:0',
             'level_1/target/qf_1/fc0/kernel:0',
             'level_1/target/qf_1/fc1/bias:0',
             'level_1/target/qf_1/fc1/kernel:0',
             'level_1/target/qf_1/qf_output/bias:0',
             'level_1/target/qf_1/qf_output/kernel:0']
        )

    def test_init_3_levels(self):
        """Validate that the graph and variables are initialized properly."""
        policy_params = self.policy_params.copy()
        policy_params['num_levels'] = 3
        policy = TD3GoalConditionedPolicy(**policy_params)

        # Check that the abstract class has all the required attributes.
        self.assertEqual(policy.meta_period,
                         self.policy_params['meta_period'])
        self.assertEqual(policy.relative_goals,
                         self.policy_params['relative_goals'])
        self.assertEqual(policy.off_policy_corrections,
                         self.policy_params['off_policy_corrections'])
        self.assertEqual(policy.use_fingerprints,
                         self.policy_params['use_fingerprints'])
        self.assertEqual(policy.centralized_value_functions,
                         self.policy_params['centralized_value_functions'])
        self.assertEqual(policy.connected_gradients,
                         self.policy_params['connected_gradients'])
        self.assertEqual(policy.cg_weights,
                         self.policy_params['cg_weights'])

        # Check that all trainable variables have been created in the
        # TensorFlow graph.
        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['level_0/model/pi/fc0/bias:0',
             'level_0/model/pi/fc0/kernel:0',
             'level_0/model/pi/fc1/bias:0',
             'level_0/model/pi/fc1/kernel:0',
             'level_0/model/pi/output/bias:0',
             'level_0/model/pi/output/kernel:0',
             'level_0/model/qf_0/fc0/bias:0',
             'level_0/model/qf_0/fc0/kernel:0',
             'level_0/model/qf_0/fc1/bias:0',
             'level_0/model/qf_0/fc1/kernel:0',
             'level_0/model/qf_0/qf_output/bias:0',
             'level_0/model/qf_0/qf_output/kernel:0',
             'level_0/model/qf_1/fc0/bias:0',
             'level_0/model/qf_1/fc0/kernel:0',
             'level_0/model/qf_1/fc1/bias:0',
             'level_0/model/qf_1/fc1/kernel:0',
             'level_0/model/qf_1/qf_output/bias:0',
             'level_0/model/qf_1/qf_output/kernel:0',
             'level_0/target/pi/fc0/bias:0',
             'level_0/target/pi/fc0/kernel:0',
             'level_0/target/pi/fc1/bias:0',
             'level_0/target/pi/fc1/kernel:0',
             'level_0/target/pi/output/bias:0',
             'level_0/target/pi/output/kernel:0',
             'level_0/target/qf_0/fc0/bias:0',
             'level_0/target/qf_0/fc0/kernel:0',
             'level_0/target/qf_0/fc1/bias:0',
             'level_0/target/qf_0/fc1/kernel:0',
             'level_0/target/qf_0/qf_output/bias:0',
             'level_0/target/qf_0/qf_output/kernel:0',
             'level_0/target/qf_1/fc0/bias:0',
             'level_0/target/qf_1/fc0/kernel:0',
             'level_0/target/qf_1/fc1/bias:0',
             'level_0/target/qf_1/fc1/kernel:0',
             'level_0/target/qf_1/qf_output/bias:0',
             'level_0/target/qf_1/qf_output/kernel:0',
             'level_1/model/pi/fc0/bias:0',
             'level_1/model/pi/fc0/kernel:0',
             'level_1/model/pi/fc1/bias:0',
             'level_1/model/pi/fc1/kernel:0',
             'level_1/model/pi/output/bias:0',
             'level_1/model/pi/output/kernel:0',
             'level_1/model/qf_0/fc0/bias:0',
             'level_1/model/qf_0/fc0/kernel:0',
             'level_1/model/qf_0/fc1/bias:0',
             'level_1/model/qf_0/fc1/kernel:0',
             'level_1/model/qf_0/qf_output/bias:0',
             'level_1/model/qf_0/qf_output/kernel:0',
             'level_1/model/qf_1/fc0/bias:0',
             'level_1/model/qf_1/fc0/kernel:0',
             'level_1/model/qf_1/fc1/bias:0',
             'level_1/model/qf_1/fc1/kernel:0',
             'level_1/model/qf_1/qf_output/bias:0',
             'level_1/model/qf_1/qf_output/kernel:0',
             'level_1/target/pi/fc0/bias:0',
             'level_1/target/pi/fc0/kernel:0',
             'level_1/target/pi/fc1/bias:0',
             'level_1/target/pi/fc1/kernel:0',
             'level_1/target/pi/output/bias:0',
             'level_1/target/pi/output/kernel:0',
             'level_1/target/qf_0/fc0/bias:0',
             'level_1/target/qf_0/fc0/kernel:0',
             'level_1/target/qf_0/fc1/bias:0',
             'level_1/target/qf_0/fc1/kernel:0',
             'level_1/target/qf_0/qf_output/bias:0',
             'level_1/target/qf_0/qf_output/kernel:0',
             'level_1/target/qf_1/fc0/bias:0',
             'level_1/target/qf_1/fc0/kernel:0',
             'level_1/target/qf_1/fc1/bias:0',
             'level_1/target/qf_1/fc1/kernel:0',
             'level_1/target/qf_1/qf_output/bias:0',
             'level_1/target/qf_1/qf_output/kernel:0',
             'level_2/model/pi/fc0/bias:0',
             'level_2/model/pi/fc0/kernel:0',
             'level_2/model/pi/fc1/bias:0',
             'level_2/model/pi/fc1/kernel:0',
             'level_2/model/pi/output/bias:0',
             'level_2/model/pi/output/kernel:0',
             'level_2/model/qf_0/fc0/bias:0',
             'level_2/model/qf_0/fc0/kernel:0',
             'level_2/model/qf_0/fc1/bias:0',
             'level_2/model/qf_0/fc1/kernel:0',
             'level_2/model/qf_0/qf_output/bias:0',
             'level_2/model/qf_0/qf_output/kernel:0',
             'level_2/model/qf_1/fc0/bias:0',
             'level_2/model/qf_1/fc0/kernel:0',
             'level_2/model/qf_1/fc1/bias:0',
             'level_2/model/qf_1/fc1/kernel:0',
             'level_2/model/qf_1/qf_output/bias:0',
             'level_2/model/qf_1/qf_output/kernel:0',
             'level_2/target/pi/fc0/bias:0',
             'level_2/target/pi/fc0/kernel:0',
             'level_2/target/pi/fc1/bias:0',
             'level_2/target/pi/fc1/kernel:0',
             'level_2/target/pi/output/bias:0',
             'level_2/target/pi/output/kernel:0',
             'level_2/target/qf_0/fc0/bias:0',
             'level_2/target/qf_0/fc0/kernel:0',
             'level_2/target/qf_0/fc1/bias:0',
             'level_2/target/qf_0/fc1/kernel:0',
             'level_2/target/qf_0/qf_output/bias:0',
             'level_2/target/qf_0/qf_output/kernel:0',
             'level_2/target/qf_1/fc0/bias:0',
             'level_2/target/qf_1/fc0/kernel:0',
             'level_2/target/qf_1/fc1/bias:0',
             'level_2/target/qf_1/fc1/kernel:0',
             'level_2/target/qf_1/qf_output/bias:0',
             'level_2/target/qf_1/qf_output/kernel:0']
        )

    def test_initialize(self):
        """Check the functionality of the initialize() method.

        This test validates that the target variables are properly initialized
        when initialize is called.
        """
        policy = TD3GoalConditionedPolicy(**self.policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        model_var_list = [
            'level_0/model/pi/fc0/bias:0',
            'level_0/model/pi/fc0/kernel:0',
            'level_0/model/pi/fc1/bias:0',
            'level_0/model/pi/fc1/kernel:0',
            'level_0/model/pi/output/bias:0',
            'level_0/model/pi/output/kernel:0',
            'level_0/model/qf_0/fc0/bias:0',
            'level_0/model/qf_0/fc0/kernel:0',
            'level_0/model/qf_0/fc1/bias:0',
            'level_0/model/qf_0/fc1/kernel:0',
            'level_0/model/qf_0/qf_output/bias:0',
            'level_0/model/qf_0/qf_output/kernel:0',
            'level_0/model/qf_1/fc0/bias:0',
            'level_0/model/qf_1/fc0/kernel:0',
            'level_0/model/qf_1/fc1/bias:0',
            'level_0/model/qf_1/fc1/kernel:0',
            'level_0/model/qf_1/qf_output/bias:0',
            'level_0/model/qf_1/qf_output/kernel:0',

            'level_1/model/pi/fc0/bias:0',
            'level_1/model/pi/fc0/kernel:0',
            'level_1/model/pi/fc1/bias:0',
            'level_1/model/pi/fc1/kernel:0',
            'level_1/model/pi/output/bias:0',
            'level_1/model/pi/output/kernel:0',
            'level_1/model/qf_0/fc0/bias:0',
            'level_1/model/qf_0/fc0/kernel:0',
            'level_1/model/qf_0/fc1/bias:0',
            'level_1/model/qf_0/fc1/kernel:0',
            'level_1/model/qf_0/qf_output/bias:0',
            'level_1/model/qf_0/qf_output/kernel:0',
            'level_1/model/qf_1/fc0/bias:0',
            'level_1/model/qf_1/fc0/kernel:0',
            'level_1/model/qf_1/fc1/bias:0',
            'level_1/model/qf_1/fc1/kernel:0',
            'level_1/model/qf_1/qf_output/bias:0',
            'level_1/model/qf_1/qf_output/kernel:0',
        ]

        target_var_list = [
            'level_0/target/pi/fc0/bias:0',
            'level_0/target/pi/fc0/kernel:0',
            'level_0/target/pi/fc1/bias:0',
            'level_0/target/pi/fc1/kernel:0',
            'level_0/target/pi/output/bias:0',
            'level_0/target/pi/output/kernel:0',
            'level_0/target/qf_0/fc0/bias:0',
            'level_0/target/qf_0/fc0/kernel:0',
            'level_0/target/qf_0/fc1/bias:0',
            'level_0/target/qf_0/fc1/kernel:0',
            'level_0/target/qf_0/qf_output/bias:0',
            'level_0/target/qf_0/qf_output/kernel:0',
            'level_0/target/qf_1/fc0/bias:0',
            'level_0/target/qf_1/fc0/kernel:0',
            'level_0/target/qf_1/fc1/bias:0',
            'level_0/target/qf_1/fc1/kernel:0',
            'level_0/target/qf_1/qf_output/bias:0',
            'level_0/target/qf_1/qf_output/kernel:0',

            'level_1/target/pi/fc0/bias:0',
            'level_1/target/pi/fc0/kernel:0',
            'level_1/target/pi/fc1/bias:0',
            'level_1/target/pi/fc1/kernel:0',
            'level_1/target/pi/output/bias:0',
            'level_1/target/pi/output/kernel:0',
            'level_1/target/qf_0/fc0/bias:0',
            'level_1/target/qf_0/fc0/kernel:0',
            'level_1/target/qf_0/fc1/bias:0',
            'level_1/target/qf_0/fc1/kernel:0',
            'level_1/target/qf_0/qf_output/bias:0',
            'level_1/target/qf_0/qf_output/kernel:0',
            'level_1/target/qf_1/fc0/bias:0',
            'level_1/target/qf_1/fc0/kernel:0',
            'level_1/target/qf_1/fc1/bias:0',
            'level_1/target/qf_1/fc1/kernel:0',
            'level_1/target/qf_1/qf_output/bias:0',
            'level_1/target/qf_1/qf_output/kernel:0'
        ]

        for model, target in zip(model_var_list, target_var_list):
            with tf.compat.v1.variable_scope(
                    tf.compat.v1.get_variable_scope(), reuse=True):
                model_val = policy.sess.run(model)
                target_val = policy.sess.run(target)
            np.testing.assert_almost_equal(model_val, target_val)

    def test_log_probs(self):
        """Check the functionality of the log_probs() method."""
        pass  # TODO

    def test_connected_gradients(self):
        """Check the functionality of the connected-gradients feature."""
        pass  # TODO


class TestSACGoalConditionedPolicy(unittest.TestCase):
    """Test GoalConditionedPolicy in hbaselines/goal_conditioned/sac.py."""

    def setUp(self):
        self.policy_params = {
            'sess': tf.compat.v1.Session(),
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'ob_space': Box(low=-2, high=2, shape=(2,)),
            'co_space': Box(low=-3, high=3, shape=(2,)),
            'layers': None,
            'verbose': 0,
        }
        self.policy_params.update(SAC_PARAMS.copy())
        self.policy_params.update(GOAL_CONDITIONED_PARAMS.copy())

    def tearDown(self):
        self.policy_params['sess'].close()
        del self.policy_params

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_init_2_levels(self):
        """Validate that the graph and variables are initialized properly."""
        policy_params = self.policy_params.copy()
        policy_params['num_levels'] = 2
        policy = SACGoalConditionedPolicy(**policy_params)

        # Check that the abstract class has all the required attributes.
        self.assertEqual(policy.meta_period,
                         self.policy_params['meta_period'])
        self.assertEqual(policy.relative_goals,
                         self.policy_params['relative_goals'])
        self.assertEqual(policy.off_policy_corrections,
                         self.policy_params['off_policy_corrections'])
        self.assertEqual(policy.use_fingerprints,
                         self.policy_params['use_fingerprints'])
        self.assertEqual(policy.centralized_value_functions,
                         self.policy_params['centralized_value_functions'])
        self.assertEqual(policy.connected_gradients,
                         self.policy_params['connected_gradients'])
        self.assertEqual(policy.cg_weights,
                         self.policy_params['cg_weights'])

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['level_0/model/log_alpha:0',
             'level_0/model/pi/fc0/bias:0',
             'level_0/model/pi/fc0/kernel:0',
             'level_0/model/pi/fc1/bias:0',
             'level_0/model/pi/fc1/kernel:0',
             'level_0/model/pi/log_std/bias:0',
             'level_0/model/pi/log_std/kernel:0',
             'level_0/model/pi/mean/bias:0',
             'level_0/model/pi/mean/kernel:0',
             'level_0/model/value_fns/qf1/fc0/bias:0',
             'level_0/model/value_fns/qf1/fc0/kernel:0',
             'level_0/model/value_fns/qf1/fc1/bias:0',
             'level_0/model/value_fns/qf1/fc1/kernel:0',
             'level_0/model/value_fns/qf1/qf_output/bias:0',
             'level_0/model/value_fns/qf1/qf_output/kernel:0',
             'level_0/model/value_fns/qf2/fc0/bias:0',
             'level_0/model/value_fns/qf2/fc0/kernel:0',
             'level_0/model/value_fns/qf2/fc1/bias:0',
             'level_0/model/value_fns/qf2/fc1/kernel:0',
             'level_0/model/value_fns/qf2/qf_output/bias:0',
             'level_0/model/value_fns/qf2/qf_output/kernel:0',
             'level_0/model/value_fns/vf/fc0/bias:0',
             'level_0/model/value_fns/vf/fc0/kernel:0',
             'level_0/model/value_fns/vf/fc1/bias:0',
             'level_0/model/value_fns/vf/fc1/kernel:0',
             'level_0/model/value_fns/vf/vf_output/bias:0',
             'level_0/model/value_fns/vf/vf_output/kernel:0',
             'level_0/target/value_fns/vf/fc0/bias:0',
             'level_0/target/value_fns/vf/fc0/kernel:0',
             'level_0/target/value_fns/vf/fc1/bias:0',
             'level_0/target/value_fns/vf/fc1/kernel:0',
             'level_0/target/value_fns/vf/vf_output/bias:0',
             'level_0/target/value_fns/vf/vf_output/kernel:0',
             'level_1/model/log_alpha:0',
             'level_1/model/pi/fc0/bias:0',
             'level_1/model/pi/fc0/kernel:0',
             'level_1/model/pi/fc1/bias:0',
             'level_1/model/pi/fc1/kernel:0',
             'level_1/model/pi/log_std/bias:0',
             'level_1/model/pi/log_std/kernel:0',
             'level_1/model/pi/mean/bias:0',
             'level_1/model/pi/mean/kernel:0',
             'level_1/model/value_fns/qf1/fc0/bias:0',
             'level_1/model/value_fns/qf1/fc0/kernel:0',
             'level_1/model/value_fns/qf1/fc1/bias:0',
             'level_1/model/value_fns/qf1/fc1/kernel:0',
             'level_1/model/value_fns/qf1/qf_output/bias:0',
             'level_1/model/value_fns/qf1/qf_output/kernel:0',
             'level_1/model/value_fns/qf2/fc0/bias:0',
             'level_1/model/value_fns/qf2/fc0/kernel:0',
             'level_1/model/value_fns/qf2/fc1/bias:0',
             'level_1/model/value_fns/qf2/fc1/kernel:0',
             'level_1/model/value_fns/qf2/qf_output/bias:0',
             'level_1/model/value_fns/qf2/qf_output/kernel:0',
             'level_1/model/value_fns/vf/fc0/bias:0',
             'level_1/model/value_fns/vf/fc0/kernel:0',
             'level_1/model/value_fns/vf/fc1/bias:0',
             'level_1/model/value_fns/vf/fc1/kernel:0',
             'level_1/model/value_fns/vf/vf_output/bias:0',
             'level_1/model/value_fns/vf/vf_output/kernel:0',
             'level_1/target/value_fns/vf/fc0/bias:0',
             'level_1/target/value_fns/vf/fc0/kernel:0',
             'level_1/target/value_fns/vf/fc1/bias:0',
             'level_1/target/value_fns/vf/fc1/kernel:0',
             'level_1/target/value_fns/vf/vf_output/bias:0',
             'level_1/target/value_fns/vf/vf_output/kernel:0',
             ]
        )

    def test_init_3_levels(self):
        """Validate that the graph and variables are initialized properly."""
        policy_params = self.policy_params.copy()
        policy_params['num_levels'] = 3
        policy = SACGoalConditionedPolicy(**policy_params)

        # Check that the abstract class has all the required attributes.
        self.assertEqual(policy.meta_period,
                         self.policy_params['meta_period'])
        self.assertEqual(policy.relative_goals,
                         self.policy_params['relative_goals'])
        self.assertEqual(policy.off_policy_corrections,
                         self.policy_params['off_policy_corrections'])
        self.assertEqual(policy.use_fingerprints,
                         self.policy_params['use_fingerprints'])
        self.assertEqual(policy.centralized_value_functions,
                         self.policy_params['centralized_value_functions'])
        self.assertEqual(policy.connected_gradients,
                         self.policy_params['connected_gradients'])
        self.assertEqual(policy.cg_weights,
                         self.policy_params['cg_weights'])

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['level_0/model/log_alpha:0',
             'level_0/model/pi/fc0/bias:0',
             'level_0/model/pi/fc0/kernel:0',
             'level_0/model/pi/fc1/bias:0',
             'level_0/model/pi/fc1/kernel:0',
             'level_0/model/pi/log_std/bias:0',
             'level_0/model/pi/log_std/kernel:0',
             'level_0/model/pi/mean/bias:0',
             'level_0/model/pi/mean/kernel:0',
             'level_0/model/value_fns/qf1/fc0/bias:0',
             'level_0/model/value_fns/qf1/fc0/kernel:0',
             'level_0/model/value_fns/qf1/fc1/bias:0',
             'level_0/model/value_fns/qf1/fc1/kernel:0',
             'level_0/model/value_fns/qf1/qf_output/bias:0',
             'level_0/model/value_fns/qf1/qf_output/kernel:0',
             'level_0/model/value_fns/qf2/fc0/bias:0',
             'level_0/model/value_fns/qf2/fc0/kernel:0',
             'level_0/model/value_fns/qf2/fc1/bias:0',
             'level_0/model/value_fns/qf2/fc1/kernel:0',
             'level_0/model/value_fns/qf2/qf_output/bias:0',
             'level_0/model/value_fns/qf2/qf_output/kernel:0',
             'level_0/model/value_fns/vf/fc0/bias:0',
             'level_0/model/value_fns/vf/fc0/kernel:0',
             'level_0/model/value_fns/vf/fc1/bias:0',
             'level_0/model/value_fns/vf/fc1/kernel:0',
             'level_0/model/value_fns/vf/vf_output/bias:0',
             'level_0/model/value_fns/vf/vf_output/kernel:0',
             'level_0/target/value_fns/vf/fc0/bias:0',
             'level_0/target/value_fns/vf/fc0/kernel:0',
             'level_0/target/value_fns/vf/fc1/bias:0',
             'level_0/target/value_fns/vf/fc1/kernel:0',
             'level_0/target/value_fns/vf/vf_output/bias:0',
             'level_0/target/value_fns/vf/vf_output/kernel:0',
             'level_1/model/log_alpha:0',
             'level_1/model/pi/fc0/bias:0',
             'level_1/model/pi/fc0/kernel:0',
             'level_1/model/pi/fc1/bias:0',
             'level_1/model/pi/fc1/kernel:0',
             'level_1/model/pi/log_std/bias:0',
             'level_1/model/pi/log_std/kernel:0',
             'level_1/model/pi/mean/bias:0',
             'level_1/model/pi/mean/kernel:0',
             'level_1/model/value_fns/qf1/fc0/bias:0',
             'level_1/model/value_fns/qf1/fc0/kernel:0',
             'level_1/model/value_fns/qf1/fc1/bias:0',
             'level_1/model/value_fns/qf1/fc1/kernel:0',
             'level_1/model/value_fns/qf1/qf_output/bias:0',
             'level_1/model/value_fns/qf1/qf_output/kernel:0',
             'level_1/model/value_fns/qf2/fc0/bias:0',
             'level_1/model/value_fns/qf2/fc0/kernel:0',
             'level_1/model/value_fns/qf2/fc1/bias:0',
             'level_1/model/value_fns/qf2/fc1/kernel:0',
             'level_1/model/value_fns/qf2/qf_output/bias:0',
             'level_1/model/value_fns/qf2/qf_output/kernel:0',
             'level_1/model/value_fns/vf/fc0/bias:0',
             'level_1/model/value_fns/vf/fc0/kernel:0',
             'level_1/model/value_fns/vf/fc1/bias:0',
             'level_1/model/value_fns/vf/fc1/kernel:0',
             'level_1/model/value_fns/vf/vf_output/bias:0',
             'level_1/model/value_fns/vf/vf_output/kernel:0',
             'level_1/target/value_fns/vf/fc0/bias:0',
             'level_1/target/value_fns/vf/fc0/kernel:0',
             'level_1/target/value_fns/vf/fc1/bias:0',
             'level_1/target/value_fns/vf/fc1/kernel:0',
             'level_1/target/value_fns/vf/vf_output/bias:0',
             'level_1/target/value_fns/vf/vf_output/kernel:0',
             'level_2/model/log_alpha:0',
             'level_2/model/pi/fc0/bias:0',
             'level_2/model/pi/fc0/kernel:0',
             'level_2/model/pi/fc1/bias:0',
             'level_2/model/pi/fc1/kernel:0',
             'level_2/model/pi/log_std/bias:0',
             'level_2/model/pi/log_std/kernel:0',
             'level_2/model/pi/mean/bias:0',
             'level_2/model/pi/mean/kernel:0',
             'level_2/model/value_fns/qf1/fc0/bias:0',
             'level_2/model/value_fns/qf1/fc0/kernel:0',
             'level_2/model/value_fns/qf1/fc1/bias:0',
             'level_2/model/value_fns/qf1/fc1/kernel:0',
             'level_2/model/value_fns/qf1/qf_output/bias:0',
             'level_2/model/value_fns/qf1/qf_output/kernel:0',
             'level_2/model/value_fns/qf2/fc0/bias:0',
             'level_2/model/value_fns/qf2/fc0/kernel:0',
             'level_2/model/value_fns/qf2/fc1/bias:0',
             'level_2/model/value_fns/qf2/fc1/kernel:0',
             'level_2/model/value_fns/qf2/qf_output/bias:0',
             'level_2/model/value_fns/qf2/qf_output/kernel:0',
             'level_2/model/value_fns/vf/fc0/bias:0',
             'level_2/model/value_fns/vf/fc0/kernel:0',
             'level_2/model/value_fns/vf/fc1/bias:0',
             'level_2/model/value_fns/vf/fc1/kernel:0',
             'level_2/model/value_fns/vf/vf_output/bias:0',
             'level_2/model/value_fns/vf/vf_output/kernel:0',
             'level_2/target/value_fns/vf/fc0/bias:0',
             'level_2/target/value_fns/vf/fc0/kernel:0',
             'level_2/target/value_fns/vf/fc1/bias:0',
             'level_2/target/value_fns/vf/fc1/kernel:0',
             'level_2/target/value_fns/vf/vf_output/bias:0',
             'level_2/target/value_fns/vf/vf_output/kernel:0']
        )

    def test_initialize(self):
        """Check the functionality of the initialize() method.

        This test validates that the target variables are properly initialized
        when initialize is called.
        """
        policy = SACGoalConditionedPolicy(**self.policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        model_var_list = [
            'level_0/model/value_fns/vf/fc0/kernel:0',
            'level_0/model/value_fns/vf/fc0/bias:0',
            'level_0/model/value_fns/vf/fc1/kernel:0',
            'level_0/model/value_fns/vf/fc1/bias:0',
            'level_0/model/value_fns/vf/vf_output/kernel:0',
            'level_0/model/value_fns/vf/vf_output/bias:0',

            'level_1/model/value_fns/vf/fc0/kernel:0',
            'level_1/model/value_fns/vf/fc0/bias:0',
            'level_1/model/value_fns/vf/fc1/kernel:0',
            'level_1/model/value_fns/vf/fc1/bias:0',
            'level_1/model/value_fns/vf/vf_output/kernel:0',
            'level_1/model/value_fns/vf/vf_output/bias:0',
        ]

        target_var_list = [
            'level_0/target/value_fns/vf/fc0/kernel:0',
            'level_0/target/value_fns/vf/fc0/bias:0',
            'level_0/target/value_fns/vf/fc1/kernel:0',
            'level_0/target/value_fns/vf/fc1/bias:0',
            'level_0/target/value_fns/vf/vf_output/kernel:0',
            'level_0/target/value_fns/vf/vf_output/bias:0',

            'level_1/target/value_fns/vf/fc0/kernel:0',
            'level_1/target/value_fns/vf/fc0/bias:0',
            'level_1/target/value_fns/vf/fc1/kernel:0',
            'level_1/target/value_fns/vf/fc1/bias:0',
            'level_1/target/value_fns/vf/vf_output/kernel:0',
            'level_1/target/value_fns/vf/vf_output/bias:0',
        ]

        for model, target in zip(model_var_list, target_var_list):
            with tf.compat.v1.variable_scope(
                    tf.compat.v1.get_variable_scope(), reuse=True):
                model_val = policy.sess.run(model)
                target_val = policy.sess.run(target)
            np.testing.assert_almost_equal(model_val, target_val)

    def test_log_probs(self):
        """Check the functionality of the log_probs() method."""
        pass  # TODO

    def test_connected_gradients(self):
        """Check the functionality of the connected-gradients feature."""
        pass  # TODO


if __name__ == '__main__':
    unittest.main()
