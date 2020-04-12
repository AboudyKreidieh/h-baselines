"""Contains tests for the model abstractions and different models."""
import unittest
import numpy as np
import tensorflow as tf
from gym.spaces import Box

from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.fcnet.base import ActorCriticPolicy
from hbaselines.fcnet.td3 import FeedForwardPolicy as TD3FeedForwardPolicy
from hbaselines.fcnet.sac import FeedForwardPolicy as SACFeedForwardPolicy
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy as \
    TD3GoalConditionedPolicy
from hbaselines.goal_conditioned.sac import GoalConditionedPolicy as \
    SACGoalConditionedPolicy
from hbaselines.multi_fcnet.td3 import MultiFeedForwardPolicy as \
    TD3MultiFeedForwardPolicy
from hbaselines.multi_fcnet.sac import MultiFeedForwardPolicy as \
    SACMultiFeedForwardPolicy
from hbaselines.algorithms.off_policy import SAC_PARAMS, TD3_PARAMS
from hbaselines.algorithms.off_policy import FEEDFORWARD_PARAMS
from hbaselines.algorithms.off_policy import GOAL_CONDITIONED_PARAMS
from hbaselines.algorithms.off_policy import MULTI_FEEDFORWARD_PARAMS


class TestActorCriticPolicy(unittest.TestCase):
    """Test ActorCriticPolicy in hbaselines/fcnet/base.py."""

    def setUp(self):
        sess = tf.compat.v1.Session()

        self.policy_params = {
            'sess': sess,
            'ac_space': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'ob_space': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'co_space': Box(low=-3, high=3, shape=(3,), dtype=np.float32),
            'verbose': 0,
        }
        self.policy_params.update(FEEDFORWARD_PARAMS.copy())

    def tearDown(self):
        self.policy_params['sess'].close()
        del self.policy_params

    def test_init(self):
        """Validate that the variables are initialized properly."""
        policy = ActorCriticPolicy(**self.policy_params)

        # Check that the abstract class has all the required attributes.
        self.assertEqual(policy.sess, self.policy_params['sess'])
        self.assertEqual(policy.ac_space, self.policy_params['ac_space'])
        self.assertEqual(policy.ob_space, self.policy_params['ob_space'])
        self.assertEqual(policy.co_space, self.policy_params['co_space'])
        self.assertEqual(policy.buffer_size, self.policy_params['buffer_size'])
        self.assertEqual(policy.batch_size, self.policy_params['batch_size'])
        self.assertEqual(policy.actor_lr, self.policy_params['actor_lr'])
        self.assertEqual(policy.critic_lr, self.policy_params['critic_lr'])
        self.assertEqual(policy.verbose, self.policy_params['verbose'])
        self.assertEqual(policy.layers, self.policy_params['layers'])
        self.assertEqual(policy.tau, self.policy_params['tau'])
        self.assertEqual(policy.gamma, self.policy_params['gamma'])
        self.assertEqual(policy.layer_norm, self.policy_params['layer_norm'])
        self.assertEqual(policy.act_fun, self.policy_params['act_fun'])
        self.assertEqual(policy.use_huber, self.policy_params['use_huber'])

        # Check that the abstract class has all the required methods.
        self.assertRaises(NotImplementedError, policy.initialize)
        self.assertRaises(NotImplementedError, policy.update,
                          update_actor=None)
        self.assertRaises(NotImplementedError, policy.get_action,
                          obs=None, context=None, apply_noise=None,
                          random_actions=None)
        self.assertRaises(NotImplementedError, policy.store_transition,
                          obs0=None, context0=None, action=None, reward=None,
                          obs1=None, context1=None, done=None,
                          is_final_step=None, evaluate=False)
        self.assertRaises(NotImplementedError, policy.get_td_map)

    def test_get_obs(self):
        """Check the functionality of the _get_obs() method.

        This method is tested for three cases:

        1. when the context is None
        2. for 1-D observations and contexts
        3. for 2-D observations and contexts
        """
        policy = ActorCriticPolicy(**self.policy_params)

        # test case 1
        obs = np.array([0, 1, 2])
        context = None
        expected = obs
        np.testing.assert_almost_equal(policy._get_obs(obs, context), expected)

        # test case 2
        obs = np.array([0, 1, 2])
        context = np.array([3, 4])
        expected = np.array([0, 1, 2, 3, 4])
        np.testing.assert_almost_equal(policy._get_obs(obs, context), expected)

        # test case 3
        obs = np.array([[0, 1, 2]])
        context = np.array([[3, 4]])
        expected = np.array([[0, 1, 2, 3, 4]])
        np.testing.assert_almost_equal(policy._get_obs(obs, context, axis=1),
                                       expected)

    def test_get_ob_dim(self):
        """Check the functionality of the _get_ob_dim() method.

        This method is tested for two cases:

        1. when the context is None
        2. when the context is not None
        """
        policy = ActorCriticPolicy(**self.policy_params)

        # test case 1
        ob_space = Box(0, 1, shape=(2,), dtype=np.float32)
        co_space = None
        self.assertTupleEqual(policy._get_ob_dim(ob_space, co_space), (2,))

        # test case 2
        ob_space = Box(0, 1, shape=(2,), dtype=np.float32)
        co_space = Box(0, 1, shape=(3,), dtype=np.float32)
        self.assertTupleEqual(policy._get_ob_dim(ob_space, co_space), (5,))

    def test_layer(self):
        """Check the functionality of the _layer() method.

        This method is tested for the following features:

        1. the number of outputs from the layer equals num_outputs
        2. the name is properly used
        3. the proper activation function applied if requested
        4. weights match what the kernel_initializer requests (tested on a
           constant initializer)
        5. layer_norm is applied if requested
        """
        # policy = ActorCriticPolicy(**self.policy_params)

        # test case 1
        pass  # TODO

        # test case 2
        pass  # TODO

        # test case 3
        pass  # TODO

        # test case 4
        pass  # TODO

        # test case 5
        pass  # TODO

    def test_setup_target_updates(self):
        """Check the functionality of the _setup_target_updates() method.

        This test validates both the init and soft update procedures generated
        by the tested method.
        """
        policy = ActorCriticPolicy(**self.policy_params)

        _ = tf.Variable(initial_value=[[1, 1, 1, 1]], dtype=tf.float32,
                        name="0")
        val1 = tf.Variable(initial_value=[[0, 0, 0, 0]], dtype=tf.float32,
                           name="1")

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        init, soft = policy._setup_target_updates("0", "1", None, 0.1, 0)

        # test soft update
        policy.sess.run(soft)
        expected = np.array([[0.1, 0.1, 0.1, 0.1]])
        np.testing.assert_almost_equal(policy.sess.run(val1), expected)

        # test init update
        policy.sess.run(init)
        expected = np.array([[1, 1, 1, 1]])
        np.testing.assert_almost_equal(policy.sess.run(val1), expected)

    def test_remove_fingerprint(self):
        """Check the functionality of the _remove_fingerprint() method.

        This method is tested for two cases:

        1. for an additional_dim of zero
        2. for an additional_dim greater than zero
        """
        policy = ActorCriticPolicy(**self.policy_params)

        # test case 1
        val = tf.constant(value=[[1, 2, 3, 4]], dtype=tf.float32)
        new_val = policy._remove_fingerprint(val, 4, 2, 0)
        expected = np.array([[1, 2, 0, 0]])
        np.testing.assert_almost_equal(policy.sess.run(new_val), expected)

        # test case 2
        val = tf.constant(value=[[1, 2, 3, 4]], dtype=tf.float32)
        new_val = policy._remove_fingerprint(val, 3, 2, 1)
        expected = np.array([[1, 0, 0, 4]])
        np.testing.assert_almost_equal(policy.sess.run(new_val), expected)


class TestTD3FeedForwardPolicy(unittest.TestCase):
    """Test FeedForwardPolicy in hbaselines/fcnet/td3.py."""

    def setUp(self):
        self.policy_params = {
            'sess': tf.compat.v1.Session(),
            'ac_space': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'ob_space': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'co_space': Box(low=-3, high=3, shape=(3,), dtype=np.float32),
            'scope': None,
            'verbose': 0,
        }
        self.policy_params.update(TD3_PARAMS.copy())
        self.policy_params.update(FEEDFORWARD_PARAMS.copy())

    def tearDown(self):
        self.policy_params['sess'].close()
        del self.policy_params

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_init(self):
        """Check the functionality of the __init__() method.

        This method is tested for the following features:

        1. The proper structure graph was generated.
        2. All input placeholders are correct.
        """
        policy = TD3FeedForwardPolicy(**self.policy_params)

        # test case 1
        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['model/pi/fc0/bias:0',
             'model/pi/fc0/kernel:0',
             'model/pi/fc1/bias:0',
             'model/pi/fc1/kernel:0',
             'model/pi/output/bias:0',
             'model/pi/output/kernel:0',
             'model/qf_0/fc0/bias:0',
             'model/qf_0/fc0/kernel:0',
             'model/qf_0/fc1/bias:0',
             'model/qf_0/fc1/kernel:0',
             'model/qf_0/qf_output/bias:0',
             'model/qf_0/qf_output/kernel:0',
             'model/qf_1/fc0/bias:0',
             'model/qf_1/fc0/kernel:0',
             'model/qf_1/fc1/bias:0',
             'model/qf_1/fc1/kernel:0',
             'model/qf_1/qf_output/bias:0',
             'model/qf_1/qf_output/kernel:0',
             'target/pi/fc0/bias:0',
             'target/pi/fc0/kernel:0',
             'target/pi/fc1/bias:0',
             'target/pi/fc1/kernel:0',
             'target/pi/output/bias:0',
             'target/pi/output/kernel:0',
             'target/qf_0/fc0/bias:0',
             'target/qf_0/fc0/kernel:0',
             'target/qf_0/fc1/bias:0',
             'target/qf_0/fc1/kernel:0',
             'target/qf_0/qf_output/bias:0',
             'target/qf_0/qf_output/kernel:0',
             'target/qf_1/fc0/bias:0',
             'target/qf_1/fc0/kernel:0',
             'target/qf_1/fc1/bias:0',
             'target/qf_1/fc1/kernel:0',
             'target/qf_1/qf_output/bias:0',
             'target/qf_1/qf_output/kernel:0']
        )

        # test case 2
        self.assertEqual(
            tuple(v.__int__() for v in policy.terminals1.shape),
            (None, 1))
        self.assertEqual(
            tuple(v.__int__() for v in policy.rew_ph.shape),
            (None, 1))
        self.assertEqual(
            tuple(v.__int__() for v in policy.action_ph.shape),
            (None, self.policy_params['ac_space'].shape[0]))
        self.assertEqual(
            tuple(v.__int__() for v in policy.obs_ph.shape),
            (None, self.policy_params['ob_space'].shape[0] +
             self.policy_params['co_space'].shape[0]))
        self.assertEqual(
            tuple(v.__int__() for v in policy.obs1_ph.shape),
            (None, self.policy_params['ob_space'].shape[0] +
             self.policy_params['co_space'].shape[0]))

    def test_initialize(self):
        """Check the functionality of the initialize() method.

        This test validates that the target variables are properly initialized
        when initialize is called.
        """
        policy = TD3FeedForwardPolicy(**self.policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        model_var_list = [
            'model/pi/fc0/bias:0',
            'model/pi/fc0/kernel:0',
            'model/pi/fc1/bias:0',
            'model/pi/fc1/kernel:0',
            'model/pi/output/bias:0',
            'model/pi/output/kernel:0',
            'model/qf_0/fc0/bias:0',
            'model/qf_0/fc0/kernel:0',
            'model/qf_0/fc1/bias:0',
            'model/qf_0/fc1/kernel:0',
            'model/qf_0/qf_output/bias:0',
            'model/qf_0/qf_output/kernel:0',
            'model/qf_1/fc0/bias:0',
            'model/qf_1/fc0/kernel:0',
            'model/qf_1/fc1/bias:0',
            'model/qf_1/fc1/kernel:0',
            'model/qf_1/qf_output/bias:0',
            'model/qf_1/qf_output/kernel:0',
        ]

        target_var_list = [
            'target/pi/fc0/bias:0',
            'target/pi/fc0/kernel:0',
            'target/pi/fc1/bias:0',
            'target/pi/fc1/kernel:0',
            'target/pi/output/bias:0',
            'target/pi/output/kernel:0',
            'target/qf_0/fc0/bias:0',
            'target/qf_0/fc0/kernel:0',
            'target/qf_0/fc1/bias:0',
            'target/qf_0/fc1/kernel:0',
            'target/qf_0/qf_output/bias:0',
            'target/qf_0/qf_output/kernel:0',
            'target/qf_1/fc0/bias:0',
            'target/qf_1/fc0/kernel:0',
            'target/qf_1/fc1/bias:0',
            'target/qf_1/fc1/kernel:0',
            'target/qf_1/qf_output/bias:0',
            'target/qf_1/qf_output/kernel:0'
        ]

        for model, target in zip(model_var_list, target_var_list):
            with tf.compat.v1.variable_scope(
                    tf.compat.v1.get_variable_scope(), reuse=True):
                model_val = policy.sess.run(model)
                target_val = policy.sess.run(target)
            np.testing.assert_almost_equal(model_val, target_val)

    def test_store_transition(self):
        """Test the `store_transition` method."""
        pass  # TODO


class TestSACFeedForwardPolicy(unittest.TestCase):
    """Test FeedForwardPolicy in hbaselines/fcnet/td3.py."""

    def setUp(self):
        self.policy_params = {
            'sess': tf.compat.v1.Session(),
            'ac_space': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'ob_space': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'co_space': Box(low=-3, high=3, shape=(3,), dtype=np.float32),
            'scope': None,
            'verbose': 0,
        }
        self.policy_params.update(SAC_PARAMS.copy())
        self.policy_params.update(FEEDFORWARD_PARAMS.copy())

    def tearDown(self):
        self.policy_params['sess'].close()
        del self.policy_params

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_init(self):
        """Check the functionality of the __init__() method.

        This method is tested for the following features:

        1. The proper structure graph was generated.
        2. All input placeholders are correct.
        3. self.log_alpha is initialized to zero
        4. self.target_entropy is initialized as specified, with the special
           (None) case as well
        """
        policy = SACFeedForwardPolicy(**self.policy_params)

        # test case 1
        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['model/log_alpha:0',
             'model/pi/fc0/bias:0',
             'model/pi/fc0/kernel:0',
             'model/pi/fc1/bias:0',
             'model/pi/fc1/kernel:0',
             'model/pi/log_std/bias:0',
             'model/pi/log_std/kernel:0',
             'model/pi/mean/bias:0',
             'model/pi/mean/kernel:0',
             'model/value_fns/qf1/fc0/bias:0',
             'model/value_fns/qf1/fc0/kernel:0',
             'model/value_fns/qf1/fc1/bias:0',
             'model/value_fns/qf1/fc1/kernel:0',
             'model/value_fns/qf1/qf_output/bias:0',
             'model/value_fns/qf1/qf_output/kernel:0',
             'model/value_fns/qf2/fc0/bias:0',
             'model/value_fns/qf2/fc0/kernel:0',
             'model/value_fns/qf2/fc1/bias:0',
             'model/value_fns/qf2/fc1/kernel:0',
             'model/value_fns/qf2/qf_output/bias:0',
             'model/value_fns/qf2/qf_output/kernel:0',
             'model/value_fns/vf/fc0/bias:0',
             'model/value_fns/vf/fc0/kernel:0',
             'model/value_fns/vf/fc1/bias:0',
             'model/value_fns/vf/fc1/kernel:0',
             'model/value_fns/vf/vf_output/bias:0',
             'model/value_fns/vf/vf_output/kernel:0',
             'target/value_fns/vf/fc0/bias:0',
             'target/value_fns/vf/fc0/kernel:0',
             'target/value_fns/vf/fc1/bias:0',
             'target/value_fns/vf/fc1/kernel:0',
             'target/value_fns/vf/vf_output/bias:0',
             'target/value_fns/vf/vf_output/kernel:0']
        )

        # test case 2
        self.assertEqual(
            tuple(v.__int__() for v in policy.terminals1.shape),
            (None, 1))
        self.assertEqual(
            tuple(v.__int__() for v in policy.rew_ph.shape),
            (None, 1))
        self.assertEqual(
            tuple(v.__int__() for v in policy.action_ph.shape),
            (None, self.policy_params['ac_space'].shape[0]))
        self.assertEqual(
            tuple(v.__int__() for v in policy.obs_ph.shape),
            (None, self.policy_params['ob_space'].shape[0] +
             self.policy_params['co_space'].shape[0]))
        self.assertEqual(
            tuple(v.__int__() for v in policy.obs1_ph.shape),
            (None, self.policy_params['ob_space'].shape[0] +
             self.policy_params['co_space'].shape[0]))

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # test case 3
        self.assertEqual(policy.sess.run(policy.log_alpha), 0.0)

        # test case 4a
        self.assertEqual(policy.target_entropy,
                         -self.policy_params['ac_space'].shape[0])

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

        # test case 4b
        self.policy_params['target_entropy'] = 5
        policy = SACFeedForwardPolicy(**self.policy_params)
        self.assertEqual(policy.target_entropy,
                         self.policy_params['target_entropy'])

    def test_initialize(self):
        """Check the functionality of the initialize() method.

        This test validates that the target variables are properly initialized
        when initialize is called.
        """
        policy = SACFeedForwardPolicy(**self.policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        model_var_list = [
            'model/value_fns/vf/fc0/kernel:0',
            'model/value_fns/vf/fc0/bias:0',
            'model/value_fns/vf/fc1/kernel:0',
            'model/value_fns/vf/fc1/bias:0',
            'model/value_fns/vf/vf_output/kernel:0',
            'model/value_fns/vf/vf_output/bias:0',
        ]

        target_var_list = [
            'target/value_fns/vf/fc0/kernel:0',
            'target/value_fns/vf/fc0/bias:0',
            'target/value_fns/vf/fc1/kernel:0',
            'target/value_fns/vf/fc1/bias:0',
            'target/value_fns/vf/vf_output/kernel:0',
            'target/value_fns/vf/vf_output/bias:0',
        ]

        for model, target in zip(model_var_list, target_var_list):
            with tf.compat.v1.variable_scope(
                    tf.compat.v1.get_variable_scope(), reuse=True):
                model_val = policy.sess.run(model)
                target_val = policy.sess.run(target)
            np.testing.assert_almost_equal(model_val, target_val)

    def test_store_transition(self):
        """Check the functionality of the store_transition() method."""
        pass  # TODO


class TestBaseGoalConditionedPolicy(unittest.TestCase):
    """Test GoalConditionedPolicy in hbaselines/goal_conditioned/base.py."""

    def setUp(self):
        self.policy_params = {
            'sess': tf.compat.v1.Session(),
            'ac_space': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'ob_space': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'co_space': Box(low=-3, high=3, shape=(2,), dtype=np.float32),
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
                    [np.array([0, 0]), np.array([1, 1]), np.array([2, 2]),
                     np.array([3, 3]), np.array([4, 4])][i])
                for i in range(len(obs_t)))
        )

        for i in range(len(action_t)):
            self.assertTrue(
                all(all(action_t[i][j] ==
                        [[np.array([5, 5]), np.array([5, 5]), np.array([5, 5]),
                          np.array([5, 5]), np.array([5, 5])],
                         [np.array([0]), np.array([1]), np.array([2]),
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
                    [np.array([0, 0]), np.array([1, 1]), np.array([2, 2]),
                     np.array([3, 3]), np.array([4, 4])][i])
                for i in range(len(obs_t)))
        )

        for i in range(len(action_t)):
            self.assertTrue(
                all(all(action_t[i][j] ==
                        [[np.array([5, 5]), np.array([5, 5]), np.array([5, 5]),
                          np.array([5, 5]), np.array([5, 5])],
                         [np.array([0]), np.array([1]), np.array([2]),
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
                    [np.array([0, 0]), np.array([1, 1]), np.array([2, 2]),
                     np.array([3, 3]), np.array([4, 4])][i])
                for i in range(len(obs_t)))
        )

        for i in range(len(action_t)):
            self.assertTrue(
                all(all(action_t[i][j] ==
                        [[np.array([4, 4]), np.array([4, 4]), np.array([4, 4]),
                          np.array([4, 4]), np.array([4, 4])],
                         [np.array([0]), np.array([1]), np.array([2]),
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

        # hindsight sample
        obs_t = policy.replay_buffer._obs_t[1]
        action_t = policy.replay_buffer._action_t[1]
        reward_t = policy.replay_buffer._reward_t[1]
        done_t = policy.replay_buffer._done_t[1]

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
                        [[np.array([4, 4]), np.array([3, 3]), np.array([2, 2]),
                          np.array([1, 1]), np.array([0, 0])],
                         [np.array([0]), np.array([1]), np.array([2]),
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
        """Validate the functionality of the intrinsic rewards."""
        policy = TD3GoalConditionedPolicy(**self.policy_params)

        self.assertAlmostEqual(
            policy.intrinsic_reward_fn(
                states=np.array([1, 2]),
                goals=np.array([3, 2]),
                next_states=np.array([0, 0])
            ),
            -3.6055512754778567
        )

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
            'ac_space': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'ob_space': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'co_space': Box(low=-3, high=3, shape=(2,), dtype=np.float32),
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
            'ac_space': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'ob_space': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'co_space': Box(low=-3, high=3, shape=(2,), dtype=np.float32),
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


class TestBaseMultiFeedForwardPolicy(unittest.TestCase):
    """Test MultiFeedForwardPolicy in hbaselines/multi_fcnet/base.py."""

    def setUp(self):
        self.sess = tf.compat.v1.Session()

        # Shared policy parameters
        self.policy_params_shared = {
            'sess': self.sess,
            'ac_space': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'co_space': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'ob_space': Box(low=-3, high=3, shape=(3,), dtype=np.float32),
            'all_ob_space': Box(low=-3, high=3, shape=(10,), dtype=np.float32),
            'layers': [256, 256],
            'verbose': 0,
        }
        self.policy_params_shared.update(TD3_PARAMS.copy())
        self.policy_params_shared.update(MULTI_FEEDFORWARD_PARAMS.copy())
        self.policy_params_shared['shared'] = True

        # Independent policy parameters
        self.policy_params_independent = {
            'sess': self.sess,
            'ac_space': {
                'a': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                'b': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            },
            'co_space': {
                'a': Box(low=-3, high=3, shape=(3,), dtype=np.float32),
                'b': Box(low=-4, high=4, shape=(4,), dtype=np.float32),
            },
            'ob_space': {
                'a': Box(low=-5, high=5, shape=(5,), dtype=np.float32),
                'b': Box(low=-6, high=6, shape=(6,), dtype=np.float32),
            },
            'all_ob_space': Box(low=-6, high=6, shape=(18,), dtype=np.float32),
            'layers': [256, 256],
            'verbose': 0,
        }
        self.policy_params_independent.update(TD3_PARAMS.copy())
        self.policy_params_independent.update(MULTI_FEEDFORWARD_PARAMS.copy())
        self.policy_params_independent['shared'] = False

    def tearDown(self):
        self.sess.close()
        del self.policy_params_shared
        del self.policy_params_independent

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_store_transition_1(self):
        """Check the functionality of the store_transition() method.

        This test checks for the following cases:

        1. maddpg = False, shared = False
        2. maddpg = False, shared = True
        3. maddpg = True,  shared = False
        4. maddpg = True,  shared = True
        """
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = False
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        del policy  # TODO

    def test_store_transition_2(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = False
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        del policy  # TODO

    def test_store_transition_3(self):
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = True
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        del policy  # TODO

    def test_store_transition_4(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = True
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        del policy  # TODO

    def test_get_td_map_1(self):
        """Check the functionality of the get_td_map() method.

        This test checks for the following cases:

        1. maddpg = False, shared = False
        2. maddpg = False, shared = True
        3. maddpg = True,  shared = False
        4. maddpg = True,  shared = True
        """
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = False
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        del policy  # TODO

    def test_get_td_map_2(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = False
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        del policy  # TODO

    def test_get_td_map_3(self):
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = True
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        del policy  # TODO

    def test_get_td_map_4(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = True
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        del policy  # TODO


class TestTD3MultiFeedForwardPolicy(unittest.TestCase):
    """Test MultiFeedForwardPolicy in hbaselines/multi_fcnet/td3.py."""

    def setUp(self):
        self.sess = tf.compat.v1.Session()

        # Shared policy parameters
        self.policy_params_shared = {
            'sess': self.sess,
            'ac_space': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'co_space': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'ob_space': Box(low=-3, high=3, shape=(3,), dtype=np.float32),
            'all_ob_space': Box(low=-3, high=3, shape=(10,), dtype=np.float32),
            'layers': [256, 256],
            'verbose': 0,
        }
        self.policy_params_shared.update(TD3_PARAMS.copy())
        self.policy_params_shared.update(MULTI_FEEDFORWARD_PARAMS.copy())
        self.policy_params_shared['shared'] = True

        # Independent policy parameters
        self.policy_params_independent = {
            'sess': self.sess,
            'ac_space': {
                'a': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                'b': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            },
            'co_space': {
                'a': Box(low=-3, high=3, shape=(3,), dtype=np.float32),
                'b': Box(low=-4, high=4, shape=(4,), dtype=np.float32),
            },
            'ob_space': {
                'a': Box(low=-5, high=5, shape=(5,), dtype=np.float32),
                'b': Box(low=-6, high=6, shape=(6,), dtype=np.float32),
            },
            'all_ob_space': Box(low=-6, high=6, shape=(18,), dtype=np.float32),
            'layers': [256, 256],
            'verbose': 0,
        }
        self.policy_params_independent.update(TD3_PARAMS.copy())
        self.policy_params_independent.update(MULTI_FEEDFORWARD_PARAMS.copy())
        self.policy_params_independent['shared'] = False

    def tearDown(self):
        self.sess.close()
        del self.policy_params_shared
        del self.policy_params_independent

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_init_1(self):
        """Check the functionality of the __init__() method.

        This method is tested for the following features:

        1. The proper structure graph was generated.
        2. All input placeholders are correct.

        This is done for the following cases:

        1. maddpg = False, shared = False
        2. maddpg = False, shared = True
        3. maddpg = True,  shared = False
        4. maddpg = True,  shared = True
        """
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = False
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['a/model/pi/fc0/bias:0',
             'a/model/pi/fc0/kernel:0',
             'a/model/pi/fc1/bias:0',
             'a/model/pi/fc1/kernel:0',
             'a/model/pi/output/bias:0',
             'a/model/pi/output/kernel:0',
             'a/model/qf_0/fc0/bias:0',
             'a/model/qf_0/fc0/kernel:0',
             'a/model/qf_0/fc1/bias:0',
             'a/model/qf_0/fc1/kernel:0',
             'a/model/qf_0/qf_output/bias:0',
             'a/model/qf_0/qf_output/kernel:0',
             'a/model/qf_1/fc0/bias:0',
             'a/model/qf_1/fc0/kernel:0',
             'a/model/qf_1/fc1/bias:0',
             'a/model/qf_1/fc1/kernel:0',
             'a/model/qf_1/qf_output/bias:0',
             'a/model/qf_1/qf_output/kernel:0',
             'a/target/pi/fc0/bias:0',
             'a/target/pi/fc0/kernel:0',
             'a/target/pi/fc1/bias:0',
             'a/target/pi/fc1/kernel:0',
             'a/target/pi/output/bias:0',
             'a/target/pi/output/kernel:0',
             'a/target/qf_0/fc0/bias:0',
             'a/target/qf_0/fc0/kernel:0',
             'a/target/qf_0/fc1/bias:0',
             'a/target/qf_0/fc1/kernel:0',
             'a/target/qf_0/qf_output/bias:0',
             'a/target/qf_0/qf_output/kernel:0',
             'a/target/qf_1/fc0/bias:0',
             'a/target/qf_1/fc0/kernel:0',
             'a/target/qf_1/fc1/bias:0',
             'a/target/qf_1/fc1/kernel:0',
             'a/target/qf_1/qf_output/bias:0',
             'a/target/qf_1/qf_output/kernel:0',
             'b/model/pi/fc0/bias:0',
             'b/model/pi/fc0/kernel:0',
             'b/model/pi/fc1/bias:0',
             'b/model/pi/fc1/kernel:0',
             'b/model/pi/output/bias:0',
             'b/model/pi/output/kernel:0',
             'b/model/qf_0/fc0/bias:0',
             'b/model/qf_0/fc0/kernel:0',
             'b/model/qf_0/fc1/bias:0',
             'b/model/qf_0/fc1/kernel:0',
             'b/model/qf_0/qf_output/bias:0',
             'b/model/qf_0/qf_output/kernel:0',
             'b/model/qf_1/fc0/bias:0',
             'b/model/qf_1/fc0/kernel:0',
             'b/model/qf_1/fc1/bias:0',
             'b/model/qf_1/fc1/kernel:0',
             'b/model/qf_1/qf_output/bias:0',
             'b/model/qf_1/qf_output/kernel:0',
             'b/target/pi/fc0/bias:0',
             'b/target/pi/fc0/kernel:0',
             'b/target/pi/fc1/bias:0',
             'b/target/pi/fc1/kernel:0',
             'b/target/pi/output/bias:0',
             'b/target/pi/output/kernel:0',
             'b/target/qf_0/fc0/bias:0',
             'b/target/qf_0/fc0/kernel:0',
             'b/target/qf_0/fc1/bias:0',
             'b/target/qf_0/fc1/kernel:0',
             'b/target/qf_0/qf_output/bias:0',
             'b/target/qf_0/qf_output/kernel:0',
             'b/target/qf_1/fc0/bias:0',
             'b/target/qf_1/fc0/kernel:0',
             'b/target/qf_1/fc1/bias:0',
             'b/target/qf_1/fc1/kernel:0',
             'b/target/qf_1/qf_output/bias:0',
             'b/target/qf_1/qf_output/kernel:0']
        )

        # Check observation/action/context spaces of the agents
        self.assertEqual(policy.agents['a'].ac_space,
                         self.policy_params_independent['ac_space']['a'])
        self.assertEqual(policy.agents['a'].ob_space,
                         self.policy_params_independent['ob_space']['a'])
        self.assertEqual(policy.agents['a'].co_space,
                         self.policy_params_independent['co_space']['a'])

        self.assertEqual(policy.agents['b'].ac_space,
                         self.policy_params_independent['ac_space']['b'])
        self.assertEqual(policy.agents['b'].ob_space,
                         self.policy_params_independent['ob_space']['b'])
        self.assertEqual(policy.agents['b'].co_space,
                         self.policy_params_independent['co_space']['b'])

        # Check the instantiation of the class attributes.
        self.assertTrue(not policy.shared)
        self.assertTrue(not policy.maddpg)

    def test_init_2(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = False
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['model/pi/fc0/bias:0',
             'model/pi/fc0/kernel:0',
             'model/pi/fc1/bias:0',
             'model/pi/fc1/kernel:0',
             'model/pi/output/bias:0',
             'model/pi/output/kernel:0',
             'model/qf_0/fc0/bias:0',
             'model/qf_0/fc0/kernel:0',
             'model/qf_0/fc1/bias:0',
             'model/qf_0/fc1/kernel:0',
             'model/qf_0/qf_output/bias:0',
             'model/qf_0/qf_output/kernel:0',
             'model/qf_1/fc0/bias:0',
             'model/qf_1/fc0/kernel:0',
             'model/qf_1/fc1/bias:0',
             'model/qf_1/fc1/kernel:0',
             'model/qf_1/qf_output/bias:0',
             'model/qf_1/qf_output/kernel:0',
             'target/pi/fc0/bias:0',
             'target/pi/fc0/kernel:0',
             'target/pi/fc1/bias:0',
             'target/pi/fc1/kernel:0',
             'target/pi/output/bias:0',
             'target/pi/output/kernel:0',
             'target/qf_0/fc0/bias:0',
             'target/qf_0/fc0/kernel:0',
             'target/qf_0/fc1/bias:0',
             'target/qf_0/fc1/kernel:0',
             'target/qf_0/qf_output/bias:0',
             'target/qf_0/qf_output/kernel:0',
             'target/qf_1/fc0/bias:0',
             'target/qf_1/fc0/kernel:0',
             'target/qf_1/fc1/bias:0',
             'target/qf_1/fc1/kernel:0',
             'target/qf_1/qf_output/bias:0',
             'target/qf_1/qf_output/kernel:0']
        )

        # Check observation/action/context spaces of the agents
        self.assertEqual(policy.agents['policy'].ac_space,
                         self.policy_params_shared['ac_space'])
        self.assertEqual(policy.agents['policy'].ob_space,
                         self.policy_params_shared['ob_space'])
        self.assertEqual(policy.agents['policy'].co_space,
                         self.policy_params_shared['co_space'])

        # Check the instantiation of the class attributes.
        self.assertTrue(policy.shared)
        self.assertTrue(not policy.maddpg)

    def test_init_3(self):
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = True
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['a/model/centralized_qf_0/fc0/bias:0',
             'a/model/centralized_qf_0/fc0/kernel:0',
             'a/model/centralized_qf_0/fc1/bias:0',
             'a/model/centralized_qf_0/fc1/kernel:0',
             'a/model/centralized_qf_0/qf_output/bias:0',
             'a/model/centralized_qf_0/qf_output/kernel:0',
             'a/model/centralized_qf_1/fc0/bias:0',
             'a/model/centralized_qf_1/fc0/kernel:0',
             'a/model/centralized_qf_1/fc1/bias:0',
             'a/model/centralized_qf_1/fc1/kernel:0',
             'a/model/centralized_qf_1/qf_output/bias:0',
             'a/model/centralized_qf_1/qf_output/kernel:0',
             'a/model/pi/fc0/bias:0',
             'a/model/pi/fc0/kernel:0',
             'a/model/pi/fc1/bias:0',
             'a/model/pi/fc1/kernel:0',
             'a/model/pi/output/bias:0',
             'a/model/pi/output/kernel:0',
             'a/target/centralized_qf_0/fc0/bias:0',
             'a/target/centralized_qf_0/fc0/kernel:0',
             'a/target/centralized_qf_0/fc1/bias:0',
             'a/target/centralized_qf_0/fc1/kernel:0',
             'a/target/centralized_qf_0/qf_output/bias:0',
             'a/target/centralized_qf_0/qf_output/kernel:0',
             'a/target/centralized_qf_1/fc0/bias:0',
             'a/target/centralized_qf_1/fc0/kernel:0',
             'a/target/centralized_qf_1/fc1/bias:0',
             'a/target/centralized_qf_1/fc1/kernel:0',
             'a/target/centralized_qf_1/qf_output/bias:0',
             'a/target/centralized_qf_1/qf_output/kernel:0',
             'a/target/pi/fc0/bias:0',
             'a/target/pi/fc0/kernel:0',
             'a/target/pi/fc1/bias:0',
             'a/target/pi/fc1/kernel:0',
             'a/target/pi/output/bias:0',
             'a/target/pi/output/kernel:0',
             'b/model/centralized_qf_0/fc0/bias:0',
             'b/model/centralized_qf_0/fc0/kernel:0',
             'b/model/centralized_qf_0/fc1/bias:0',
             'b/model/centralized_qf_0/fc1/kernel:0',
             'b/model/centralized_qf_0/qf_output/bias:0',
             'b/model/centralized_qf_0/qf_output/kernel:0',
             'b/model/centralized_qf_1/fc0/bias:0',
             'b/model/centralized_qf_1/fc0/kernel:0',
             'b/model/centralized_qf_1/fc1/bias:0',
             'b/model/centralized_qf_1/fc1/kernel:0',
             'b/model/centralized_qf_1/qf_output/bias:0',
             'b/model/centralized_qf_1/qf_output/kernel:0',
             'b/model/pi/fc0/bias:0',
             'b/model/pi/fc0/kernel:0',
             'b/model/pi/fc1/bias:0',
             'b/model/pi/fc1/kernel:0',
             'b/model/pi/output/bias:0',
             'b/model/pi/output/kernel:0',
             'b/target/centralized_qf_0/fc0/bias:0',
             'b/target/centralized_qf_0/fc0/kernel:0',
             'b/target/centralized_qf_0/fc1/bias:0',
             'b/target/centralized_qf_0/fc1/kernel:0',
             'b/target/centralized_qf_0/qf_output/bias:0',
             'b/target/centralized_qf_0/qf_output/kernel:0',
             'b/target/centralized_qf_1/fc0/bias:0',
             'b/target/centralized_qf_1/fc0/kernel:0',
             'b/target/centralized_qf_1/fc1/bias:0',
             'b/target/centralized_qf_1/fc1/kernel:0',
             'b/target/centralized_qf_1/qf_output/bias:0',
             'b/target/centralized_qf_1/qf_output/kernel:0',
             'b/target/pi/fc0/bias:0',
             'b/target/pi/fc0/kernel:0',
             'b/target/pi/fc1/bias:0',
             'b/target/pi/fc1/kernel:0',
             'b/target/pi/output/bias:0',
             'b/target/pi/output/kernel:0']
        )

        # Check observation/action/context spaces of the agents
        for key in policy.ac_space.keys():
            self.assertEqual(int(policy.all_obs_ph[key].shape[-1]),
                             policy.all_ob_space.shape[0])
            self.assertEqual(int(policy.all_obs1_ph[key].shape[-1]),
                             policy.all_ob_space.shape[0])
            self.assertEqual(int(policy.all_action_ph[key].shape[-1]),
                             sum(policy.ac_space[key].shape[0]
                                 for key in policy.ac_space.keys()))
            self.assertEqual(int(policy.action_ph[key].shape[-1]),
                             policy.ac_space[key].shape[0])
            self.assertEqual(int(policy.obs_ph[key].shape[-1]),
                             int(policy.ob_space[key].shape[0]
                                 + policy.co_space[key].shape[0]))
            self.assertEqual(int(policy.obs1_ph[key].shape[-1]),
                             int(policy.ob_space[key].shape[0]
                                 + policy.co_space[key].shape[0]))

        # Check the instantiation of the class attributes.
        self.assertTrue(not policy.shared)
        self.assertTrue(policy.maddpg)

    def test_init_4(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = True
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['model/centralized_qf_0/fc0/bias:0',
             'model/centralized_qf_0/fc0/kernel:0',
             'model/centralized_qf_0/fc1/bias:0',
             'model/centralized_qf_0/fc1/kernel:0',
             'model/centralized_qf_0/qf_output/bias:0',
             'model/centralized_qf_0/qf_output/kernel:0',
             'model/centralized_qf_1/fc0/bias:0',
             'model/centralized_qf_1/fc0/kernel:0',
             'model/centralized_qf_1/fc1/bias:0',
             'model/centralized_qf_1/fc1/kernel:0',
             'model/centralized_qf_1/qf_output/bias:0',
             'model/centralized_qf_1/qf_output/kernel:0',
             'model/pi/fc0/bias:0',
             'model/pi/fc0/kernel:0',
             'model/pi/fc1/bias:0',
             'model/pi/fc1/kernel:0',
             'model/pi/output/bias:0',
             'model/pi/output/kernel:0',
             'target/centralized_qf_0/fc0/bias:0',
             'target/centralized_qf_0/fc0/kernel:0',
             'target/centralized_qf_0/fc1/bias:0',
             'target/centralized_qf_0/fc1/kernel:0',
             'target/centralized_qf_0/qf_output/bias:0',
             'target/centralized_qf_0/qf_output/kernel:0',
             'target/centralized_qf_1/fc0/bias:0',
             'target/centralized_qf_1/fc0/kernel:0',
             'target/centralized_qf_1/fc1/bias:0',
             'target/centralized_qf_1/fc1/kernel:0',
             'target/centralized_qf_1/qf_output/bias:0',
             'target/centralized_qf_1/qf_output/kernel:0',
             'target/pi/fc0/bias:0',
             'target/pi/fc0/kernel:0',
             'target/pi/fc1/bias:0',
             'target/pi/fc1/kernel:0',
             'target/pi/output/bias:0',
             'target/pi/output/kernel:0']
        )

        # Check observation/action/context spaces of the agents
        self.assertEqual(int(policy.all_obs_ph.shape[-1]),
                         policy.all_ob_space.shape[0])
        self.assertEqual(int(policy.all_obs1_ph.shape[-1]),
                         policy.all_ob_space.shape[0])
        self.assertEqual(int(policy.all_action_ph.shape[-1]),
                         policy.n_agents * policy.ac_space.shape[0])
        self.assertEqual(int(policy.action_ph[0].shape[-1]),
                         policy.ac_space.shape[0])
        self.assertEqual(int(policy.obs_ph[0].shape[-1]),
                         int(policy.ob_space.shape[0]
                             + policy.co_space.shape[0]))
        self.assertEqual(int(policy.obs1_ph[0].shape[-1]),
                         int(policy.ob_space.shape[0]
                             + policy.co_space.shape[0]))

        # Check the instantiation of the class attributes.
        self.assertTrue(policy.shared)
        self.assertTrue(policy.maddpg)

    def test_initialize_1(self):
        """Check the functionality of the initialize() method.

        This test validates that the target variables are properly initialized
        when initialize is called.

        This is done for the following cases:

        1. maddpg = False, shared = False
        2. maddpg = False, shared = True
        3. maddpg = True,  shared = False
        4. maddpg = True,  shared = True
        """
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = False
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        model_var_list = [
            'a/model/pi/fc0/bias:0',
            'a/model/pi/fc0/kernel:0',
            'a/model/pi/fc1/bias:0',
            'a/model/pi/fc1/kernel:0',
            'a/model/pi/output/bias:0',
            'a/model/pi/output/kernel:0',
            'a/model/qf_0/fc0/bias:0',
            'a/model/qf_0/fc0/kernel:0',
            'a/model/qf_0/fc1/bias:0',
            'a/model/qf_0/fc1/kernel:0',
            'a/model/qf_0/qf_output/bias:0',
            'a/model/qf_0/qf_output/kernel:0',
            'a/model/qf_1/fc0/bias:0',
            'a/model/qf_1/fc0/kernel:0',
            'a/model/qf_1/fc1/bias:0',
            'a/model/qf_1/fc1/kernel:0',
            'a/model/qf_1/qf_output/bias:0',
            'a/model/qf_1/qf_output/kernel:0',
            'b/model/pi/fc0/bias:0',
            'b/model/pi/fc0/kernel:0',
            'b/model/pi/fc1/bias:0',
            'b/model/pi/fc1/kernel:0',
            'b/model/pi/output/bias:0',
            'b/model/pi/output/kernel:0',
            'b/model/qf_0/fc0/bias:0',
            'b/model/qf_0/fc0/kernel:0',
            'b/model/qf_0/fc1/bias:0',
            'b/model/qf_0/fc1/kernel:0',
            'b/model/qf_0/qf_output/bias:0',
            'b/model/qf_0/qf_output/kernel:0',
            'b/model/qf_1/fc0/bias:0',
            'b/model/qf_1/fc0/kernel:0',
            'b/model/qf_1/fc1/bias:0',
            'b/model/qf_1/fc1/kernel:0',
            'b/model/qf_1/qf_output/bias:0',
            'b/model/qf_1/qf_output/kernel:0',
        ]

        target_var_list = [
            'a/target/pi/fc0/bias:0',
            'a/target/pi/fc0/kernel:0',
            'a/target/pi/fc1/bias:0',
            'a/target/pi/fc1/kernel:0',
            'a/target/pi/output/bias:0',
            'a/target/pi/output/kernel:0',
            'a/target/qf_0/fc0/bias:0',
            'a/target/qf_0/fc0/kernel:0',
            'a/target/qf_0/fc1/bias:0',
            'a/target/qf_0/fc1/kernel:0',
            'a/target/qf_0/qf_output/bias:0',
            'a/target/qf_0/qf_output/kernel:0',
            'a/target/qf_1/fc0/bias:0',
            'a/target/qf_1/fc0/kernel:0',
            'a/target/qf_1/fc1/bias:0',
            'a/target/qf_1/fc1/kernel:0',
            'a/target/qf_1/qf_output/bias:0',
            'a/target/qf_1/qf_output/kernel:0',
            'b/target/pi/fc0/bias:0',
            'b/target/pi/fc0/kernel:0',
            'b/target/pi/fc1/bias:0',
            'b/target/pi/fc1/kernel:0',
            'b/target/pi/output/bias:0',
            'b/target/pi/output/kernel:0',
            'b/target/qf_0/fc0/bias:0',
            'b/target/qf_0/fc0/kernel:0',
            'b/target/qf_0/fc1/bias:0',
            'b/target/qf_0/fc1/kernel:0',
            'b/target/qf_0/qf_output/bias:0',
            'b/target/qf_0/qf_output/kernel:0',
            'b/target/qf_1/fc0/bias:0',
            'b/target/qf_1/fc0/kernel:0',
            'b/target/qf_1/fc1/bias:0',
            'b/target/qf_1/fc1/kernel:0',
            'b/target/qf_1/qf_output/bias:0',
            'b/target/qf_1/qf_output/kernel:0',
        ]

        for model, target in zip(model_var_list, target_var_list):
            with tf.compat.v1.variable_scope(
                    tf.compat.v1.get_variable_scope(), reuse=True):
                model_val = policy.sess.run(model)
                target_val = policy.sess.run(target)
            np.testing.assert_almost_equal(model_val, target_val)

    def test_initialize_2(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = False
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        model_var_list = [
            'model/pi/fc0/bias:0',
            'model/pi/fc0/kernel:0',
            'model/pi/fc1/bias:0',
            'model/pi/fc1/kernel:0',
            'model/pi/output/bias:0',
            'model/pi/output/kernel:0',
            'model/qf_0/fc0/bias:0',
            'model/qf_0/fc0/kernel:0',
            'model/qf_0/fc1/bias:0',
            'model/qf_0/fc1/kernel:0',
            'model/qf_0/qf_output/bias:0',
            'model/qf_0/qf_output/kernel:0',
            'model/qf_1/fc0/bias:0',
            'model/qf_1/fc0/kernel:0',
            'model/qf_1/fc1/bias:0',
            'model/qf_1/fc1/kernel:0',
            'model/qf_1/qf_output/bias:0',
            'model/qf_1/qf_output/kernel:0',
        ]

        target_var_list = [
            'target/pi/fc0/bias:0',
            'target/pi/fc0/kernel:0',
            'target/pi/fc1/bias:0',
            'target/pi/fc1/kernel:0',
            'target/pi/output/bias:0',
            'target/pi/output/kernel:0',
            'target/qf_0/fc0/bias:0',
            'target/qf_0/fc0/kernel:0',
            'target/qf_0/fc1/bias:0',
            'target/qf_0/fc1/kernel:0',
            'target/qf_0/qf_output/bias:0',
            'target/qf_0/qf_output/kernel:0',
            'target/qf_1/fc0/bias:0',
            'target/qf_1/fc0/kernel:0',
            'target/qf_1/fc1/bias:0',
            'target/qf_1/fc1/kernel:0',
            'target/qf_1/qf_output/bias:0',
            'target/qf_1/qf_output/kernel:0'
        ]

        for model, target in zip(model_var_list, target_var_list):
            with tf.compat.v1.variable_scope(
                    tf.compat.v1.get_variable_scope(), reuse=True):
                model_val = policy.sess.run(model)
                target_val = policy.sess.run(target)
            np.testing.assert_almost_equal(model_val, target_val)

    def test_initialize_3(self):
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = True
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        model_var_list = [
            'a/model/centralized_qf_0/fc0/bias:0',
            'a/model/centralized_qf_0/fc0/kernel:0',
            'a/model/centralized_qf_0/fc1/bias:0',
            'a/model/centralized_qf_0/fc1/kernel:0',
            'a/model/centralized_qf_0/qf_output/bias:0',
            'a/model/centralized_qf_0/qf_output/kernel:0',
            'a/model/centralized_qf_1/fc0/bias:0',
            'a/model/centralized_qf_1/fc0/kernel:0',
            'a/model/centralized_qf_1/fc1/bias:0',
            'a/model/centralized_qf_1/fc1/kernel:0',
            'a/model/centralized_qf_1/qf_output/bias:0',
            'a/model/centralized_qf_1/qf_output/kernel:0',
            'a/model/pi/fc0/bias:0',
            'a/model/pi/fc0/kernel:0',
            'a/model/pi/fc1/bias:0',
            'a/model/pi/fc1/kernel:0',
            'a/model/pi/output/bias:0',
            'a/model/pi/output/kernel:0',
            'b/model/centralized_qf_0/fc0/bias:0',
            'b/model/centralized_qf_0/fc0/kernel:0',
            'b/model/centralized_qf_0/fc1/bias:0',
            'b/model/centralized_qf_0/fc1/kernel:0',
            'b/model/centralized_qf_0/qf_output/bias:0',
            'b/model/centralized_qf_0/qf_output/kernel:0',
            'b/model/centralized_qf_1/fc0/bias:0',
            'b/model/centralized_qf_1/fc0/kernel:0',
            'b/model/centralized_qf_1/fc1/bias:0',
            'b/model/centralized_qf_1/fc1/kernel:0',
            'b/model/centralized_qf_1/qf_output/bias:0',
            'b/model/centralized_qf_1/qf_output/kernel:0',
            'b/model/pi/fc0/bias:0',
            'b/model/pi/fc0/kernel:0',
            'b/model/pi/fc1/bias:0',
            'b/model/pi/fc1/kernel:0',
            'b/model/pi/output/bias:0',
            'b/model/pi/output/kernel:0',
        ]

        target_var_list = [
            'a/target/centralized_qf_0/fc0/bias:0',
            'a/target/centralized_qf_0/fc0/kernel:0',
            'a/target/centralized_qf_0/fc1/bias:0',
            'a/target/centralized_qf_0/fc1/kernel:0',
            'a/target/centralized_qf_0/qf_output/bias:0',
            'a/target/centralized_qf_0/qf_output/kernel:0',
            'a/target/centralized_qf_1/fc0/bias:0',
            'a/target/centralized_qf_1/fc0/kernel:0',
            'a/target/centralized_qf_1/fc1/bias:0',
            'a/target/centralized_qf_1/fc1/kernel:0',
            'a/target/centralized_qf_1/qf_output/bias:0',
            'a/target/centralized_qf_1/qf_output/kernel:0',
            'a/target/pi/fc0/bias:0',
            'a/target/pi/fc0/kernel:0',
            'a/target/pi/fc1/bias:0',
            'a/target/pi/fc1/kernel:0',
            'a/target/pi/output/bias:0',
            'a/target/pi/output/kernel:0',
            'b/target/centralized_qf_0/fc0/bias:0',
            'b/target/centralized_qf_0/fc0/kernel:0',
            'b/target/centralized_qf_0/fc1/bias:0',
            'b/target/centralized_qf_0/fc1/kernel:0',
            'b/target/centralized_qf_0/qf_output/bias:0',
            'b/target/centralized_qf_0/qf_output/kernel:0',
            'b/target/centralized_qf_1/fc0/bias:0',
            'b/target/centralized_qf_1/fc0/kernel:0',
            'b/target/centralized_qf_1/fc1/bias:0',
            'b/target/centralized_qf_1/fc1/kernel:0',
            'b/target/centralized_qf_1/qf_output/bias:0',
            'b/target/centralized_qf_1/qf_output/kernel:0',
            'b/target/pi/fc0/bias:0',
            'b/target/pi/fc0/kernel:0',
            'b/target/pi/fc1/bias:0',
            'b/target/pi/fc1/kernel:0',
            'b/target/pi/output/bias:0',
            'b/target/pi/output/kernel:0',
        ]

        for model, target in zip(model_var_list, target_var_list):
            with tf.compat.v1.variable_scope(
                    tf.compat.v1.get_variable_scope(), reuse=True):
                model_val = policy.sess.run(model)
                target_val = policy.sess.run(target)
            np.testing.assert_almost_equal(model_val, target_val)

    def test_initialize_4(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = True
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        model_var_list = [
            'model/centralized_qf_0/fc0/bias:0',
            'model/centralized_qf_0/fc0/kernel:0',
            'model/centralized_qf_0/fc1/bias:0',
            'model/centralized_qf_0/fc1/kernel:0',
            'model/centralized_qf_0/qf_output/bias:0',
            'model/centralized_qf_0/qf_output/kernel:0',
            'model/centralized_qf_1/fc0/bias:0',
            'model/centralized_qf_1/fc0/kernel:0',
            'model/centralized_qf_1/fc1/bias:0',
            'model/centralized_qf_1/fc1/kernel:0',
            'model/centralized_qf_1/qf_output/bias:0',
            'model/centralized_qf_1/qf_output/kernel:0',
            'model/pi/fc0/bias:0',
            'model/pi/fc0/kernel:0',
            'model/pi/fc1/bias:0',
            'model/pi/fc1/kernel:0',
            'model/pi/output/bias:0',
            'model/pi/output/kernel:0',
        ]

        target_var_list = [
            'target/centralized_qf_0/fc0/bias:0',
            'target/centralized_qf_0/fc0/kernel:0',
            'target/centralized_qf_0/fc1/bias:0',
            'target/centralized_qf_0/fc1/kernel:0',
            'target/centralized_qf_0/qf_output/bias:0',
            'target/centralized_qf_0/qf_output/kernel:0',
            'target/centralized_qf_1/fc0/bias:0',
            'target/centralized_qf_1/fc0/kernel:0',
            'target/centralized_qf_1/fc1/bias:0',
            'target/centralized_qf_1/fc1/kernel:0',
            'target/centralized_qf_1/qf_output/bias:0',
            'target/centralized_qf_1/qf_output/kernel:0',
            'target/pi/fc0/bias:0',
            'target/pi/fc0/kernel:0',
            'target/pi/fc1/bias:0',
            'target/pi/fc1/kernel:0',
            'target/pi/output/bias:0',
            'target/pi/output/kernel:0',
        ]

        for model, target in zip(model_var_list, target_var_list):
            with tf.compat.v1.variable_scope(
                    tf.compat.v1.get_variable_scope(), reuse=True):
                model_val = policy.sess.run(model)
                target_val = policy.sess.run(target)
            np.testing.assert_almost_equal(model_val, target_val)


class TestSACMultiFeedForwardPolicy(unittest.TestCase):
    """Test MultiFeedForwardPolicy in hbaselines/multi_fcnet/sac.py."""

    def setUp(self):
        self.sess = tf.compat.v1.Session()

        # Shared policy parameters
        self.policy_params_shared = {
            'sess': self.sess,
            'ac_space': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'co_space': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'ob_space': Box(low=-3, high=3, shape=(3,), dtype=np.float32),
            'all_ob_space': Box(low=-3, high=3, shape=(10,), dtype=np.float32),
            'layers': [256, 256],
            'verbose': 0,
        }
        self.policy_params_shared.update(SAC_PARAMS.copy())
        self.policy_params_shared.update(MULTI_FEEDFORWARD_PARAMS.copy())
        self.policy_params_shared['shared'] = True

        # Independent policy parameters
        self.policy_params_independent = {
            'sess': self.sess,
            'ac_space': {
                'a': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                'b': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            },
            'co_space': {
                'a': Box(low=-3, high=3, shape=(3,), dtype=np.float32),
                'b': Box(low=-4, high=4, shape=(4,), dtype=np.float32),
            },
            'ob_space': {
                'a': Box(low=-5, high=5, shape=(5,), dtype=np.float32),
                'b': Box(low=-6, high=6, shape=(6,), dtype=np.float32),
            },
            'all_ob_space': Box(low=-6, high=6, shape=(18,), dtype=np.float32),
            'layers': [256, 256],
            'verbose': 0,
        }
        self.policy_params_independent.update(SAC_PARAMS.copy())
        self.policy_params_independent.update(MULTI_FEEDFORWARD_PARAMS.copy())
        self.policy_params_independent['shared'] = False

    def tearDown(self):
        self.sess.close()
        del self.policy_params_shared
        del self.policy_params_independent

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_init_1(self):
        """Check the functionality of the __init__() method.

        This method is tested for the following features:

        1. The proper structure graph was generated.
        2. All input placeholders are correct.

        This is done for the following cases:

        1. maddpg = False, shared = False
        2. maddpg = False, shared = True
        3. maddpg = True,  shared = False
        4. maddpg = True,  shared = True
        """
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = False
        policy = SACMultiFeedForwardPolicy(**policy_params)

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['a/model/log_alpha:0',
             'a/model/pi/fc0/bias:0',
             'a/model/pi/fc0/kernel:0',
             'a/model/pi/fc1/bias:0',
             'a/model/pi/fc1/kernel:0',
             'a/model/pi/log_std/bias:0',
             'a/model/pi/log_std/kernel:0',
             'a/model/pi/mean/bias:0',
             'a/model/pi/mean/kernel:0',
             'a/model/value_fns/qf1/fc0/bias:0',
             'a/model/value_fns/qf1/fc0/kernel:0',
             'a/model/value_fns/qf1/fc1/bias:0',
             'a/model/value_fns/qf1/fc1/kernel:0',
             'a/model/value_fns/qf1/qf_output/bias:0',
             'a/model/value_fns/qf1/qf_output/kernel:0',
             'a/model/value_fns/qf2/fc0/bias:0',
             'a/model/value_fns/qf2/fc0/kernel:0',
             'a/model/value_fns/qf2/fc1/bias:0',
             'a/model/value_fns/qf2/fc1/kernel:0',
             'a/model/value_fns/qf2/qf_output/bias:0',
             'a/model/value_fns/qf2/qf_output/kernel:0',
             'a/model/value_fns/vf/fc0/bias:0',
             'a/model/value_fns/vf/fc0/kernel:0',
             'a/model/value_fns/vf/fc1/bias:0',
             'a/model/value_fns/vf/fc1/kernel:0',
             'a/model/value_fns/vf/vf_output/bias:0',
             'a/model/value_fns/vf/vf_output/kernel:0',
             'a/target/value_fns/vf/fc0/bias:0',
             'a/target/value_fns/vf/fc0/kernel:0',
             'a/target/value_fns/vf/fc1/bias:0',
             'a/target/value_fns/vf/fc1/kernel:0',
             'a/target/value_fns/vf/vf_output/bias:0',
             'a/target/value_fns/vf/vf_output/kernel:0',
             'b/model/log_alpha:0',
             'b/model/pi/fc0/bias:0',
             'b/model/pi/fc0/kernel:0',
             'b/model/pi/fc1/bias:0',
             'b/model/pi/fc1/kernel:0',
             'b/model/pi/log_std/bias:0',
             'b/model/pi/log_std/kernel:0',
             'b/model/pi/mean/bias:0',
             'b/model/pi/mean/kernel:0',
             'b/model/value_fns/qf1/fc0/bias:0',
             'b/model/value_fns/qf1/fc0/kernel:0',
             'b/model/value_fns/qf1/fc1/bias:0',
             'b/model/value_fns/qf1/fc1/kernel:0',
             'b/model/value_fns/qf1/qf_output/bias:0',
             'b/model/value_fns/qf1/qf_output/kernel:0',
             'b/model/value_fns/qf2/fc0/bias:0',
             'b/model/value_fns/qf2/fc0/kernel:0',
             'b/model/value_fns/qf2/fc1/bias:0',
             'b/model/value_fns/qf2/fc1/kernel:0',
             'b/model/value_fns/qf2/qf_output/bias:0',
             'b/model/value_fns/qf2/qf_output/kernel:0',
             'b/model/value_fns/vf/fc0/bias:0',
             'b/model/value_fns/vf/fc0/kernel:0',
             'b/model/value_fns/vf/fc1/bias:0',
             'b/model/value_fns/vf/fc1/kernel:0',
             'b/model/value_fns/vf/vf_output/bias:0',
             'b/model/value_fns/vf/vf_output/kernel:0',
             'b/target/value_fns/vf/fc0/bias:0',
             'b/target/value_fns/vf/fc0/kernel:0',
             'b/target/value_fns/vf/fc1/bias:0',
             'b/target/value_fns/vf/fc1/kernel:0',
             'b/target/value_fns/vf/vf_output/bias:0',
             'b/target/value_fns/vf/vf_output/kernel:0']
        )

        # Check observation/action/context spaces of the agents
        self.assertEqual(policy.agents['a'].ac_space,
                         self.policy_params_independent['ac_space']['a'])
        self.assertEqual(policy.agents['a'].ob_space,
                         self.policy_params_independent['ob_space']['a'])
        self.assertEqual(policy.agents['a'].co_space,
                         self.policy_params_independent['co_space']['a'])

        self.assertEqual(policy.agents['b'].ac_space,
                         self.policy_params_independent['ac_space']['b'])
        self.assertEqual(policy.agents['b'].ob_space,
                         self.policy_params_independent['ob_space']['b'])
        self.assertEqual(policy.agents['b'].co_space,
                         self.policy_params_independent['co_space']['b'])

        # Check the instantiation of the class attributes.
        self.assertTrue(not policy.shared)
        self.assertTrue(not policy.maddpg)

    def test_init_2(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = False
        policy = SACMultiFeedForwardPolicy(**policy_params)

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['model/log_alpha:0',
             'model/pi/fc0/bias:0',
             'model/pi/fc0/kernel:0',
             'model/pi/fc1/bias:0',
             'model/pi/fc1/kernel:0',
             'model/pi/log_std/bias:0',
             'model/pi/log_std/kernel:0',
             'model/pi/mean/bias:0',
             'model/pi/mean/kernel:0',
             'model/value_fns/qf1/fc0/bias:0',
             'model/value_fns/qf1/fc0/kernel:0',
             'model/value_fns/qf1/fc1/bias:0',
             'model/value_fns/qf1/fc1/kernel:0',
             'model/value_fns/qf1/qf_output/bias:0',
             'model/value_fns/qf1/qf_output/kernel:0',
             'model/value_fns/qf2/fc0/bias:0',
             'model/value_fns/qf2/fc0/kernel:0',
             'model/value_fns/qf2/fc1/bias:0',
             'model/value_fns/qf2/fc1/kernel:0',
             'model/value_fns/qf2/qf_output/bias:0',
             'model/value_fns/qf2/qf_output/kernel:0',
             'model/value_fns/vf/fc0/bias:0',
             'model/value_fns/vf/fc0/kernel:0',
             'model/value_fns/vf/fc1/bias:0',
             'model/value_fns/vf/fc1/kernel:0',
             'model/value_fns/vf/vf_output/bias:0',
             'model/value_fns/vf/vf_output/kernel:0',
             'target/value_fns/vf/fc0/bias:0',
             'target/value_fns/vf/fc0/kernel:0',
             'target/value_fns/vf/fc1/bias:0',
             'target/value_fns/vf/fc1/kernel:0',
             'target/value_fns/vf/vf_output/bias:0',
             'target/value_fns/vf/vf_output/kernel:0']
        )

        # Check observation/action/context spaces of the agents
        self.assertEqual(policy.agents['policy'].ac_space,
                         self.policy_params_shared['ac_space'])
        self.assertEqual(policy.agents['policy'].ob_space,
                         self.policy_params_shared['ob_space'])
        self.assertEqual(policy.agents['policy'].co_space,
                         self.policy_params_shared['co_space'])

        # Check the instantiation of the class attributes.
        self.assertTrue(policy.shared)
        self.assertTrue(not policy.maddpg)

    def test_init_3(self):
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = True
        policy = SACMultiFeedForwardPolicy(**policy_params)

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['a/model/centralized_value_fns/qf1/fc0/bias:0',
             'a/model/centralized_value_fns/qf1/fc0/kernel:0',
             'a/model/centralized_value_fns/qf1/fc1/bias:0',
             'a/model/centralized_value_fns/qf1/fc1/kernel:0',
             'a/model/centralized_value_fns/qf1/qf_output/bias:0',
             'a/model/centralized_value_fns/qf1/qf_output/kernel:0',
             'a/model/centralized_value_fns/qf2/fc0/bias:0',
             'a/model/centralized_value_fns/qf2/fc0/kernel:0',
             'a/model/centralized_value_fns/qf2/fc1/bias:0',
             'a/model/centralized_value_fns/qf2/fc1/kernel:0',
             'a/model/centralized_value_fns/qf2/qf_output/bias:0',
             'a/model/centralized_value_fns/qf2/qf_output/kernel:0',
             'a/model/centralized_value_fns/vf/fc0/bias:0',
             'a/model/centralized_value_fns/vf/fc0/kernel:0',
             'a/model/centralized_value_fns/vf/fc1/bias:0',
             'a/model/centralized_value_fns/vf/fc1/kernel:0',
             'a/model/centralized_value_fns/vf/vf_output/bias:0',
             'a/model/centralized_value_fns/vf/vf_output/kernel:0',
             'a/model/log_alpha:0',
             'a/model/pi/fc0/bias:0',
             'a/model/pi/fc0/kernel:0',
             'a/model/pi/fc1/bias:0',
             'a/model/pi/fc1/kernel:0',
             'a/model/pi/log_std/bias:0',
             'a/model/pi/log_std/kernel:0',
             'a/model/pi/mean/bias:0',
             'a/model/pi/mean/kernel:0',
             'a/target/centralized_value_fns/vf/fc0/bias:0',
             'a/target/centralized_value_fns/vf/fc0/kernel:0',
             'a/target/centralized_value_fns/vf/fc1/bias:0',
             'a/target/centralized_value_fns/vf/fc1/kernel:0',
             'a/target/centralized_value_fns/vf/vf_output/bias:0',
             'a/target/centralized_value_fns/vf/vf_output/kernel:0',
             'b/model/centralized_value_fns/qf1/fc0/bias:0',
             'b/model/centralized_value_fns/qf1/fc0/kernel:0',
             'b/model/centralized_value_fns/qf1/fc1/bias:0',
             'b/model/centralized_value_fns/qf1/fc1/kernel:0',
             'b/model/centralized_value_fns/qf1/qf_output/bias:0',
             'b/model/centralized_value_fns/qf1/qf_output/kernel:0',
             'b/model/centralized_value_fns/qf2/fc0/bias:0',
             'b/model/centralized_value_fns/qf2/fc0/kernel:0',
             'b/model/centralized_value_fns/qf2/fc1/bias:0',
             'b/model/centralized_value_fns/qf2/fc1/kernel:0',
             'b/model/centralized_value_fns/qf2/qf_output/bias:0',
             'b/model/centralized_value_fns/qf2/qf_output/kernel:0',
             'b/model/centralized_value_fns/vf/fc0/bias:0',
             'b/model/centralized_value_fns/vf/fc0/kernel:0',
             'b/model/centralized_value_fns/vf/fc1/bias:0',
             'b/model/centralized_value_fns/vf/fc1/kernel:0',
             'b/model/centralized_value_fns/vf/vf_output/bias:0',
             'b/model/centralized_value_fns/vf/vf_output/kernel:0',
             'b/model/log_alpha:0',
             'b/model/pi/fc0/bias:0',
             'b/model/pi/fc0/kernel:0',
             'b/model/pi/fc1/bias:0',
             'b/model/pi/fc1/kernel:0',
             'b/model/pi/log_std/bias:0',
             'b/model/pi/log_std/kernel:0',
             'b/model/pi/mean/bias:0',
             'b/model/pi/mean/kernel:0',
             'b/target/centralized_value_fns/vf/fc0/bias:0',
             'b/target/centralized_value_fns/vf/fc0/kernel:0',
             'b/target/centralized_value_fns/vf/fc1/bias:0',
             'b/target/centralized_value_fns/vf/fc1/kernel:0',
             'b/target/centralized_value_fns/vf/vf_output/bias:0',
             'b/target/centralized_value_fns/vf/vf_output/kernel:0']
        )

        # Check observation/action/context spaces of the agents
        for key in policy.ac_space.keys():
            self.assertEqual(int(policy.all_obs_ph[key].shape[-1]),
                             int(policy.all_ob_space.shape[0]))
            self.assertEqual(int(policy.all_obs1_ph[key].shape[-1]),
                             int(policy.all_ob_space.shape[0]))
            self.assertEqual(int(policy.all_action_ph[key].shape[-1]),
                             sum(policy.ac_space[key].shape[0]
                                 for key in policy.ac_space.keys()))
            self.assertEqual(int(policy.action_ph[key].shape[-1]),
                             int(policy.ac_space[key].shape[0]))
            self.assertEqual(int(policy.obs_ph[key].shape[-1]),
                             int(policy.ob_space[key].shape[0]
                                 + policy.co_space[key].shape[0]))
            self.assertEqual(int(policy.obs1_ph[key].shape[-1]),
                             int(policy.ob_space[key].shape[0]
                                 + policy.co_space[key].shape[0]))

        # Check the instantiation of the class attributes.
        self.assertTrue(not policy.shared)
        self.assertTrue(policy.maddpg)

    def test_init_4(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = True
        policy = SACMultiFeedForwardPolicy(**policy_params)

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['model/centralized_value_fns/qf1/fc0/bias:0',
             'model/centralized_value_fns/qf1/fc0/kernel:0',
             'model/centralized_value_fns/qf1/fc1/bias:0',
             'model/centralized_value_fns/qf1/fc1/kernel:0',
             'model/centralized_value_fns/qf1/qf_output/bias:0',
             'model/centralized_value_fns/qf1/qf_output/kernel:0',
             'model/centralized_value_fns/qf2/fc0/bias:0',
             'model/centralized_value_fns/qf2/fc0/kernel:0',
             'model/centralized_value_fns/qf2/fc1/bias:0',
             'model/centralized_value_fns/qf2/fc1/kernel:0',
             'model/centralized_value_fns/qf2/qf_output/bias:0',
             'model/centralized_value_fns/qf2/qf_output/kernel:0',
             'model/centralized_value_fns/vf/fc0/bias:0',
             'model/centralized_value_fns/vf/fc0/kernel:0',
             'model/centralized_value_fns/vf/fc1/bias:0',
             'model/centralized_value_fns/vf/fc1/kernel:0',
             'model/centralized_value_fns/vf/vf_output/bias:0',
             'model/centralized_value_fns/vf/vf_output/kernel:0',
             'model/log_alpha:0',
             'model/pi/fc0/bias:0',
             'model/pi/fc0/kernel:0',
             'model/pi/fc1/bias:0',
             'model/pi/fc1/kernel:0',
             'model/pi/log_std/bias:0',
             'model/pi/log_std/kernel:0',
             'model/pi/mean/bias:0',
             'model/pi/mean/kernel:0',
             'target/centralized_value_fns/vf/fc0/bias:0',
             'target/centralized_value_fns/vf/fc0/kernel:0',
             'target/centralized_value_fns/vf/fc1/bias:0',
             'target/centralized_value_fns/vf/fc1/kernel:0',
             'target/centralized_value_fns/vf/vf_output/bias:0',
             'target/centralized_value_fns/vf/vf_output/kernel:0']
        )

        # Check observation/action/context spaces of the agents
        self.assertEqual(int(policy.all_obs_ph.shape[-1]),
                         policy.all_ob_space.shape[0])
        self.assertEqual(int(policy.all_obs1_ph.shape[-1]),
                         policy.all_ob_space.shape[0])
        self.assertEqual(int(policy.all_action_ph.shape[-1]),
                         policy.n_agents * policy.ac_space.shape[0])
        self.assertEqual(int(policy.action_ph[0].shape[-1]),
                         policy.ac_space.shape[0])
        self.assertEqual(int(policy.obs_ph[0].shape[-1]),
                         int(policy.ob_space.shape[0]
                             + policy.co_space.shape[0]))
        self.assertEqual(int(policy.obs1_ph[0].shape[-1]),
                         int(policy.ob_space.shape[0]
                             + policy.co_space.shape[0]))

        # Check the instantiation of the class attributes.
        self.assertTrue(policy.shared)
        self.assertTrue(policy.maddpg)

    def test_initialize_1(self):
        """Check the functionality of the initialize() method.

        This test validates that the target variables are properly initialized
        when initialize is called.

        This is done for the following cases:

        1. maddpg = False, shared = False
        2. maddpg = False, shared = True
        3. maddpg = True,  shared = False
        4. maddpg = True,  shared = True
        """
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = False
        policy = SACMultiFeedForwardPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        model_var_list = [
            'a/model/value_fns/vf/fc0/kernel:0',
            'a/model/value_fns/vf/fc0/bias:0',
            'a/model/value_fns/vf/fc1/kernel:0',
            'a/model/value_fns/vf/fc1/bias:0',
            'a/model/value_fns/vf/vf_output/kernel:0',
            'a/model/value_fns/vf/vf_output/bias:0',
            'b/model/value_fns/vf/fc0/kernel:0',
            'b/model/value_fns/vf/fc0/bias:0',
            'b/model/value_fns/vf/fc1/kernel:0',
            'b/model/value_fns/vf/fc1/bias:0',
            'b/model/value_fns/vf/vf_output/kernel:0',
            'b/model/value_fns/vf/vf_output/bias:0',
        ]

        target_var_list = [
            'a/target/value_fns/vf/fc0/kernel:0',
            'a/target/value_fns/vf/fc0/bias:0',
            'a/target/value_fns/vf/fc1/kernel:0',
            'a/target/value_fns/vf/fc1/bias:0',
            'a/target/value_fns/vf/vf_output/kernel:0',
            'a/target/value_fns/vf/vf_output/bias:0',
            'b/target/value_fns/vf/fc0/kernel:0',
            'b/target/value_fns/vf/fc0/bias:0',
            'b/target/value_fns/vf/fc1/kernel:0',
            'b/target/value_fns/vf/fc1/bias:0',
            'b/target/value_fns/vf/vf_output/kernel:0',
            'b/target/value_fns/vf/vf_output/bias:0',
        ]

        for model, target in zip(model_var_list, target_var_list):
            with tf.compat.v1.variable_scope(
                    tf.compat.v1.get_variable_scope(), reuse=True):
                model_val = policy.sess.run(model)
                target_val = policy.sess.run(target)
            np.testing.assert_almost_equal(model_val, target_val)

    def test_initialize_2(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = False
        policy = SACMultiFeedForwardPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        model_var_list = [
            'model/value_fns/vf/fc0/kernel:0',
            'model/value_fns/vf/fc0/bias:0',
            'model/value_fns/vf/fc1/kernel:0',
            'model/value_fns/vf/fc1/bias:0',
            'model/value_fns/vf/vf_output/kernel:0',
            'model/value_fns/vf/vf_output/bias:0',
        ]

        target_var_list = [
            'target/value_fns/vf/fc0/kernel:0',
            'target/value_fns/vf/fc0/bias:0',
            'target/value_fns/vf/fc1/kernel:0',
            'target/value_fns/vf/fc1/bias:0',
            'target/value_fns/vf/vf_output/kernel:0',
            'target/value_fns/vf/vf_output/bias:0',
        ]

        for model, target in zip(model_var_list, target_var_list):
            with tf.compat.v1.variable_scope(
                    tf.compat.v1.get_variable_scope(), reuse=True):
                model_val = policy.sess.run(model)
                target_val = policy.sess.run(target)
            np.testing.assert_almost_equal(model_val, target_val)

    def test_initialize_3(self):
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = True
        policy = SACMultiFeedForwardPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        model_var_list = [
            'a/model/centralized_value_fns/vf/fc0/kernel:0',
            'a/model/centralized_value_fns/vf/fc0/bias:0',
            'a/model/centralized_value_fns/vf/fc1/kernel:0',
            'a/model/centralized_value_fns/vf/fc1/bias:0',
            'a/model/centralized_value_fns/vf/vf_output/kernel:0',
            'a/model/centralized_value_fns/vf/vf_output/bias:0',
            'b/model/centralized_value_fns/vf/fc0/kernel:0',
            'b/model/centralized_value_fns/vf/fc0/bias:0',
            'b/model/centralized_value_fns/vf/fc1/kernel:0',
            'b/model/centralized_value_fns/vf/fc1/bias:0',
            'b/model/centralized_value_fns/vf/vf_output/kernel:0',
            'b/model/centralized_value_fns/vf/vf_output/bias:0',
        ]

        target_var_list = [
            'a/target/centralized_value_fns/vf/fc0/kernel:0',
            'a/target/centralized_value_fns/vf/fc0/bias:0',
            'a/target/centralized_value_fns/vf/fc1/kernel:0',
            'a/target/centralized_value_fns/vf/fc1/bias:0',
            'a/target/centralized_value_fns/vf/vf_output/kernel:0',
            'a/target/centralized_value_fns/vf/vf_output/bias:0',
            'b/target/centralized_value_fns/vf/fc0/kernel:0',
            'b/target/centralized_value_fns/vf/fc0/bias:0',
            'b/target/centralized_value_fns/vf/fc1/kernel:0',
            'b/target/centralized_value_fns/vf/fc1/bias:0',
            'b/target/centralized_value_fns/vf/vf_output/kernel:0',
            'b/target/centralized_value_fns/vf/vf_output/bias:0',
        ]

        for model, target in zip(model_var_list, target_var_list):
            with tf.compat.v1.variable_scope(
                    tf.compat.v1.get_variable_scope(), reuse=True):
                model_val = policy.sess.run(model)
                target_val = policy.sess.run(target)
            np.testing.assert_almost_equal(model_val, target_val)

    def test_initialize_4(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = True
        policy = SACMultiFeedForwardPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        model_var_list = [
            'model/centralized_value_fns/vf/fc0/bias:0',
            'model/centralized_value_fns/vf/fc0/kernel:0',
            'model/centralized_value_fns/vf/fc1/bias:0',
            'model/centralized_value_fns/vf/fc1/kernel:0',
            'model/centralized_value_fns/vf/vf_output/bias:0',
            'model/centralized_value_fns/vf/vf_output/kernel:0',
        ]

        target_var_list = [
            'target/centralized_value_fns/vf/fc0/bias:0',
            'target/centralized_value_fns/vf/fc0/kernel:0',
            'target/centralized_value_fns/vf/fc1/bias:0',
            'target/centralized_value_fns/vf/fc1/kernel:0',
            'target/centralized_value_fns/vf/vf_output/bias:0',
            'target/centralized_value_fns/vf/vf_output/kernel:0',
        ]

        for model, target in zip(model_var_list, target_var_list):
            with tf.compat.v1.variable_scope(
                    tf.compat.v1.get_variable_scope(), reuse=True):
                model_val = policy.sess.run(model)
                target_val = policy.sess.run(target)
            np.testing.assert_almost_equal(model_val, target_val)


if __name__ == '__main__':
    unittest.main()
