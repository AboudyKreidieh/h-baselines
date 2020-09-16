"""Tests for the policies in the hbaselines/goal_conditioned subdirectory."""
import unittest
import numpy as np
import tensorflow as tf
import os
from gym.spaces import Box

from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy as \
    TD3GoalConditionedPolicy
from hbaselines.goal_conditioned.sac import GoalConditionedPolicy as \
    SACGoalConditionedPolicy
from hbaselines.algorithms.rl_algorithm import SAC_PARAMS, TD3_PARAMS
from hbaselines.algorithms.rl_algorithm import GOAL_CONDITIONED_PARAMS


class TestBaseGoalConditionedPolicy(unittest.TestCase):
    """Test GoalConditionedPolicy in hbaselines/goal_conditioned/base.py."""

    def setUp(self):
        self.policy_params = {
            'sess': tf.compat.v1.Session(),
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'ob_space': Box(low=-2, high=2, shape=(2,)),
            'co_space': Box(low=-3, high=3, shape=(2,)),
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
                evaluate=evaluate,
                env_num=0,
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
                evaluate=evaluate,
                env_num=0,
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
                evaluate=evaluate,
                env_num=0,
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
                evaluate=evaluate,
                env_num=0,
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
        policy._observations = [[] for _ in range(1)]
        self.assertEqual(policy._update_meta(0, env_num=0), True)

        # test case 2
        policy._observations = [[] for _ in range(1)]
        self.assertEqual(policy._update_meta(1, env_num=0), True)

        # test case 3
        policy._observations = [[0 for _ in range(2)] for _ in range(1)]
        self.assertEqual(policy._update_meta(0, env_num=0), False)

        # test case 4
        policy._observations = [[0 for _ in range(2)] for _ in range(1)]
        self.assertEqual(policy._update_meta(1, env_num=0), False)

        # test case 5
        policy._observations = [[0 for _ in range(5)] for _ in range(1)]
        self.assertEqual(policy._update_meta(0, env_num=0), False)

        # test case 6
        policy._observations = [[0 for _ in range(5)] for _ in range(1)]
        self.assertEqual(policy._update_meta(1, env_num=0), True)

        # test case 7
        policy._observations = [[0 for _ in range(10)] for _ in range(1)]
        self.assertEqual(policy._update_meta(0, env_num=0), False)

        # test case 8
        policy._observations = [[0 for _ in range(10)] for _ in range(1)]
        self.assertEqual(policy._update_meta(1, env_num=0), True)

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

        # test case 5  TODO: temporarily removed
        # policy_params = self.policy_params.copy()
        # policy_params["intrinsic_reward_type"] = "exp_negative_distance"
        # policy = TD3GoalConditionedPolicy(**policy_params)
        #
        # self.assertAlmostEqual(
        #     policy.intrinsic_reward_fn(
        #         states=np.array([1, 2]),
        #         goals=np.array([3, 2]),
        #         next_states=np.array([0, 0])
        #     ),
        #     2.2603294067550214e-06
        # )
        #
        # # Clear the graph.
        # del policy
        # tf.compat.v1.reset_default_graph()
        #
        # # test case 6
        # policy_params = self.policy_params.copy()
        # policy_params["intrinsic_reward_type"] =
        # "scaled_exp_negative_distance"
        # policy = TD3GoalConditionedPolicy(**policy_params)
        #
        # self.assertAlmostEqual(
        #     policy.intrinsic_reward_fn(
        #         states=np.array([1, 2]),
        #         goals=np.array([3, 2]),
        #         next_states=np.array([0, 0])
        #     ),
        #     0.03877420782784459
        # )
        #
        # # Clear the graph.
        # del policy
        # tf.compat.v1.reset_default_graph()

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

    def test_pretrain_train(self):
        pass  # TODO

    def test_pretrain_load(self):
        """Validates the functionality of the pre-trained loading operation.

        This method performs the following tests:

        1. Check the parameters that are loaded match the expected values for
           the following cases:
             a. no ckpt_num
             b. specified ckpt_num
        2. Check the num-levels assertions.
        3. Check the name mismatch assertions.
        4. Check the shape mismatch assertions.
        """
        # =================================================================== #
        # test case 1.a                                                       #
        # =================================================================== #

        policy_params = self.policy_params.copy()
        policy_params['env_name'] = "AntGather"
        policy_params['ac_space'] = Box(low=-1, high=1, shape=(8,))
        policy_params['ob_space'] = Box(low=-2, high=2, shape=(145,))
        policy_params['co_space'] = None
        policy_params["pretrain_ckpt"] = None
        policy_params["pretrain_path"] = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "supp_files/antgather_pretrained")

        # Initialize the policy.
        policy = TD3GoalConditionedPolicy(**policy_params)
        policy_params["sess"].run(tf.compat.v1.global_variables_initializer())
        policy.initialize()

        # Check that a specific variable was properly imported.
        val = policy_params['sess'].run(
            [v for v in get_trainable_vars()
             if v.name == 'level_1/model/qf_0/fc0/bias:0'][0])
        np.testing.assert_almost_equal(val, [
            -1.48894560e+00, 3.58899146e-01, -3.12390327e-01, 1.05936837e+00,
            1.38732374e+00, -7.56121755e-01, 2.48271704e+00, -2.05165744e+00,
            -1.61468133e-01, 1.25600290e+00, -9.43906605e-01, -2.03393340e+00,
            -2.29675269e+00, -1.25849569e+00, -1.04694414e+00, -1.83898121e-01,
            -9.34544444e-01, -9.39514577e-01, -8.24286878e-01, -3.24028790e-01,
            -1.26432347e+00, -8.07699203e-01, -2.03145698e-01, -9.32382643e-01,
            2.27273297e+00, 2.91315496e-01, -8.49827111e-01, -1.19624889e+00,
            -7.36967742e-01, 3.80735010e-01, -1.04197454e+00, 1.18836582e+00,
            -1.83718574e+00, -1.24244237e+00, 2.45089218e-01, -4.17167872e-01,
            -1.41953543e-01, -8.37748587e-01, -2.18121910e+00, 5.56767821e-01,
            -1.29594386e+00, -9.27014410e-01, -1.68718779e+00, 6.69142008e-01,
            -1.39908814e+00, -1.95475662e+00, 7.18701899e-01, -1.32534623e+00,
            -1.14043212e+00, -1.22170222e+00, 5.62444143e-03, 2.68185925e+00,
            -1.56831956e+00, -1.04305923e+00, -2.31012130e+00, 5.11528671e-01,
            4.30034250e-02, 6.72668815e-01, 4.98563275e-02, 1.89185989e+00,
            4.75367278e-01, 1.31618962e-01, -4.30353314e-01, -1.38826180e+00,
            7.58701682e-01, -2.08867580e-01, 1.46650529e+00, -1.34794402e+00,
            -1.28479147e+00, 5.54868519e-01, -1.35296082e+00, -5.64228296e-01,
            -1.64990091e+00, 1.70074478e-01, -7.73855269e-01, 9.44840729e-01,
            -1.61343896e+00, 2.19100404e+00, -1.79324031e+00, -4.74657029e-01,
            -1.66948462e+00, -1.44102037e+00, -1.34983373e+00, -3.05905676e+00,
            -1.38599861e+00, 5.20873368e-01, -8.71101394e-02, -5.01250863e-01,
            -1.26061702e+00, -3.74994457e-01, -1.68697524e+00, 1.99624926e-01,
            -7.37098455e-01, -8.95464718e-01, 8.42310011e-01, 2.96877623e-01,
            6.30660236e-01, -2.50717555e-03, -1.66991621e-01, -1.14282012e+00,
            -5.67260504e-01, -1.92494416e+00, 2.28663180e-02, -1.65851915e+00,
            -2.95281500e-01, 3.20799381e-01, -7.72721350e-01, 1.25514364e+00,
            -2.04449749e+00, -8.87898579e-02, -6.51960552e-01, -1.77445424e+00,
            -3.86437088e-01, -1.11224163e+00, -6.70421720e-01, -2.17338133e+00,
            -3.35618734e-01, -1.67795861e+00, -1.39098215e+00, -1.39096355e+00,
            -3.32385808e-01, -2.20834541e+00, 2.68285781e-01, -1.78917277e+00,
            7.14439452e-01, -1.82225907e+00, 5.38432002e-02, -9.48697269e-01,
            -1.79560018e+00, -2.60755122e-01, -1.36371219e+00, -1.12363076e+00,
            3.52526784e-01, 9.74688947e-01, -6.59091055e-01, -1.92110169e+00,
            9.61817920e-01, 1.10466979e-01, -1.74070978e+00, 2.35996276e-01,
            2.66458392e-01, 5.11776388e-01, -4.22770470e-01, -5.01518667e-01,
            -6.74514353e-01, -2.04032588e+00, 6.24135733e-01, -1.00860429e+00,
            -1.20869434e+00, -1.09191227e+00, -9.90680575e-01, -1.74508417e+00,
            -3.98090512e-01, 3.87987524e-01, -6.02133632e-01, -7.73799062e-01,
            -4.79838520e-01, -4.89767551e-01, 2.42014183e-03, -1.22625983e+00,
            -1.89408350e+00, -1.54449260e+00, -2.16307446e-01, 2.24804148e-01,
            1.63136855e-01, 8.17742109e-01, 6.85068190e-01, -2.78682292e-01,
            -2.09007263e+00, 2.92312443e-01, -5.10016918e-01, -1.44344103e+00,
            -5.83410144e-01, -1.71676278e+00, -1.50649571e+00, -1.49973166e+00,
            -1.35697436e+00, -1.47692740e+00, -6.14630103e-01, -1.29880393e+00,
            -2.07584405e+00, -5.11216223e-01, -1.34403944e+00, -4.78691638e-01,
            -1.66473842e+00, -5.31982303e-01, -1.66593182e+00, -7.33766377e-01,
            -1.62023246e+00, -1.02181005e+00, -3.27389210e-01, -1.41320837e+00,
            -8.24594378e-01, 5.25394440e-01, -1.22156549e+00, -8.49891067e-01,
            -7.39182889e-01, -9.82530266e-02, -2.07317996e+00, 1.53992057e+00,
            2.26437593e+00, -6.76014304e-01, -2.09113672e-01, -8.64942092e-03,
            9.76478100e-01, -1.45346420e-02, -5.73182166e-01, -1.95388699e+00,
            -1.02166450e+00, -5.90587735e-01, -1.70384955e+00, -1.85054946e+00,
            -2.99974024e-01, 1.06089568e+00, -2.35082674e+00, 3.42316955e-01,
            -6.42771125e-01, -9.12041187e-01, -4.83356744e-01, -1.37228274e+00,
            3.14537406e-01, -8.45137686e-02, -1.31570756e+00, -3.75019222e-01,
            -1.60828781e+00, -4.84681696e-01, -1.31065655e+00, -4.02286202e-02,
            -2.01719475e+00, -1.11378193e+00, 3.52912426e-01, -9.41884220e-01,
            -5.25218129e-01, -1.77293980e+00, -6.72002435e-02, 1.20008305e-01,
            3.78115475e-01, -1.83169377e+00, -1.50664413e+00, -1.06811893e+00,
            -3.79809648e-01, -5.96868277e-01, 1.98485756e+00, -5.59798896e-01,
            -2.07565832e+00, -2.38732004e+00, -1.17256868e+00, -1.39947581e+00,
            -2.00424719e+00, -1.27984846e+00, -9.68914032e-01, -1.08717775e+00,
            -9.26078737e-01, -1.80117083e+00, 4.01354134e-02, -1.38454926e+00
        ])

        # Clear everything.
        policy_params['sess'].close()
        del policy, policy_params
        tf.compat.v1.reset_default_graph()

        # =================================================================== #
        # test case 1.b                                                       #
        # =================================================================== #

        policy_params = self.policy_params.copy()
        policy_params['sess'] = tf.compat.v1.Session()
        policy_params['env_name'] = "AntGather"
        policy_params['ac_space'] = Box(low=-1, high=1, shape=(8,))
        policy_params['ob_space'] = Box(low=-2, high=2, shape=(145,))
        policy_params['co_space'] = None
        policy_params["pretrain_ckpt"] = 950000
        policy_params["pretrain_path"] = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "supp_files/antgather_pretrained")

        # Initialize the policy.
        policy = TD3GoalConditionedPolicy(**policy_params)
        policy_params["sess"].run(tf.compat.v1.global_variables_initializer())
        policy.initialize()

        # Check that a specific variable was properly imported.
        val = policy_params['sess'].run(
            [v for v in get_trainable_vars()
             if v.name == 'level_1/model/qf_0/fc0/bias:0'][0])
        np.testing.assert_almost_equal(val, [
            -1.4366732, 0.36718956, -0.2640507, 1.0437311, 1.435836,
            -0.7175612, 2.4794638, -2.008584, -0.13273863, 1.2509729,
            -0.9034862, -2.010076, -2.3473845, -1.2298152, -1.0250556,
            -0.14490157, -0.8938364, -0.88856345, -0.79820406, -0.2834655,
            -1.2690866, -0.7234725, -0.06116989, -0.96936285, 2.1851509,
            0.34769204, -0.84226114, -1.1418648, -0.65562344, 0.3946888,
            -1.104742, 1.2037259, -1.7700378, -1.2426982, 0.23546624,
            -0.40742588, -0.12413302, -0.81522846, -2.1267579, 0.4729991,
            -1.2849647, -0.79315525, -1.6224108, 0.6998365, -1.3191581,
            -1.9422212, 0.72227126, -1.29492, -1.1103692, -1.1423935,
            0.0076926, 2.6213, -1.5342246, -1.0392663, -2.2443452, 0.64512855,
            0.18015313, 0.60515064, 0.0554107, 1.8528932, 0.57574534,
            0.13323912, -0.4050354, -1.3367869, 0.81102616, -0.14901115,
            1.543448, -1.3410215, -1.2213068, 0.5141707, -1.324863, -0.5085816,
            -1.6287427, 0.13013484, -0.77610874, 0.9606328, -1.5942631,
            2.1985466, -1.6981885, -0.36084813, -1.6960877, -1.360367,
            -1.3659251, -3.0274456, -1.4273877, 0.5129615, -0.1093144,
            -0.43663844, -1.1985509, -0.29366097, -1.6280884, 0.22439516,
            -0.74784815, -0.8907019, 0.76174164, 0.35366687, 0.65905726,
            -0.01560472, -0.11770402, -1.1197222, -0.47018343, -1.9040893,
            0.03329567, -1.6015282, -0.30328047, 0.39247894, -0.750885,
            1.2188467, -1.9867113, -0.12792407, -0.627084, -1.7039652,
            -0.43332908, -1.0184548, -0.6147298, -2.1101472, -0.26271334,
            -1.751358, -1.3182638, -1.3871324, -0.3446217, -2.1281924,
            0.31755528, -1.7161235, 0.70565104, -1.7920085, 0.0363204,
            -0.88730013, -1.7582611, -0.11671179, -1.28648, -1.130786,
            0.36456427, 1.0425327, -0.5649138, -1.8939986, 0.92786133,
            0.3719001, -1.6747571, 0.2629407, 0.24945816, 0.513846,
            -0.20199735, -0.5183297, -0.7256823, -2.0206127, 0.6360105,
            -0.9467028, -1.0234271, -1.0853163, -0.9800702, -1.6657795,
            -0.378815, 0.5030308, -0.5950817, -0.80788535, -0.46545032,
            -0.4003353, 0.15714464, -1.1822406, -1.8132814, -1.4949279,
            -0.19650824, 0.17290121, 0.14448208, 0.7915036, 0.68795997,
            -0.15296787, -2.0713952, 0.28636524, -0.51249075, -1.4200838,
            -0.5479608, -1.6923913, -1.4860058, -1.4889748, -1.3654616,
            -1.5259326, -0.62569904, -1.2274802, -2.0149336, -0.32267615,
            -1.3334148, -0.4404422, -1.632482, -0.4593529, -1.5977027,
            -0.7732608, -1.3987267, -0.7833183, -0.3451375, -1.3842496,
            -0.86086637, 0.5492374, -1.2013254, -0.76750344, -0.72928184,
            -0.05954946, -2.0095, 1.5386069, 2.2867568, -0.5790614,
            -0.17414978, 0.00453592, 0.97018963, 0.00640345, -0.52000904,
            -1.9536601, -1.0231961, -0.5838816, -1.6897516, -1.8527417,
            -0.31952888, 1.1161242, -2.2025425, 0.38021305, -0.65044636,
            -0.8635659, -0.4389457, -1.3815931, 0.29587498, -0.03644859,
            -1.2045414, -0.34211895, -1.616288, -0.41780123, -1.3086252,
            0.01040083, -2.0251021, -1.0961326, 0.40894645, -0.93678725,
            -0.5369472, -1.7250992, -0.05198791, 0.13407248, 0.41778204,
            -1.7781827, -1.5506623, -1.04724, -0.31670633, -0.6066604,
            2.0247016, -0.49830088, -2.002778, -2.3635867, -1.1983025,
            -1.3054183, -1.9579597, -1.2624493, -0.93990403, -1.0444145,
            -0.9772149, -1.7095491, 0.0431014, -1.3628837
        ])

        # Clear everything.
        policy_params['sess'].close()
        del policy, policy_params
        tf.compat.v1.reset_default_graph()

        # =================================================================== #
        # test case 2                                                         #
        # =================================================================== #

        policy_params = self.policy_params.copy()
        policy_params['sess'] = tf.compat.v1.Session()
        policy_params['env_name'] = "AntGather"
        policy_params['num_levels'] = 3
        policy_params["pretrain_path"] = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "supp_files/antgather_pretrained")

        # Initialize the policy.
        policy = TD3GoalConditionedPolicy(**policy_params)
        policy_params["sess"].run(tf.compat.v1.global_variables_initializer())

        # Check for the AssertionError.
        self.assertRaisesRegex(
            AssertionError,
            "Number of levels between the checkpoint and current policy do "
            "not match. Policy=3, Checkpoint=2",
            policy.initialize)

        # Clear everything.
        policy_params['sess'].close()
        del policy, policy_params
        tf.compat.v1.reset_default_graph()

        # =================================================================== #
        # test case 3                                                         #
        # =================================================================== #

        pass  # TODO

        # =================================================================== #
        # test case 4                                                         #
        # =================================================================== #

        policy_params = self.policy_params.copy()
        policy_params['sess'] = tf.compat.v1.Session()
        policy_params["pretrain_path"] = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "supp_files/antgather_pretrained")

        # Initialize the policy.
        policy = TD3GoalConditionedPolicy(**policy_params)
        policy_params["sess"].run(tf.compat.v1.global_variables_initializer())

        # Check for the AssertionError.
        self.assertRaises(AssertionError, policy.initialize)

        # Clear everything.
        policy_params['sess'].close()
        del policy, policy_params
        tf.compat.v1.reset_default_graph()


class TestTD3GoalConditionedPolicy(unittest.TestCase):
    """Test GoalConditionedPolicy in hbaselines/goal_conditioned/td3.py."""

    def setUp(self):
        self.policy_params = {
            'sess': tf.compat.v1.Session(),
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'ob_space': Box(low=-2, high=2, shape=(2,)),
            'co_space': Box(low=-3, high=3, shape=(2,)),
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
        self.assertEqual(policy.cooperative_gradients,
                         self.policy_params['cooperative_gradients'])
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
        self.assertEqual(policy.cooperative_gradients,
                         self.policy_params['cooperative_gradients'])
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

    def test_cooperative_gradients(self):
        """Check the functionality of the cooperative-gradients feature."""
        pass  # TODO


class TestSACGoalConditionedPolicy(unittest.TestCase):
    """Test GoalConditionedPolicy in hbaselines/goal_conditioned/sac.py."""

    def setUp(self):
        self.policy_params = {
            'sess': tf.compat.v1.Session(),
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'ob_space': Box(low=-2, high=2, shape=(2,)),
            'co_space': Box(low=-3, high=3, shape=(2,)),
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
        self.assertEqual(policy.cooperative_gradients,
                         self.policy_params['cooperative_gradients'])
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
        self.assertEqual(policy.cooperative_gradients,
                         self.policy_params['cooperative_gradients'])
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

    def test_cooperative_gradients(self):
        """Check the functionality of the cooperative-gradients feature."""
        pass  # TODO


if __name__ == '__main__':
    unittest.main()
