"""Contains tests for the model abstractions and different models."""
import unittest
import numpy as np
import tensorflow as tf
import random
from gym.spaces import Box
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.fcnet.base import ActorCriticPolicy
from hbaselines.fcnet.td3 import FeedForwardPolicy as TD3FeedForwardPolicy
# from hbaselines.fcnet.sac import FeedForwardPolicy as SACFeedForwardPolicy
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy
from hbaselines.algorithms.off_policy import SAC_PARAMS, TD3_PARAMS
from hbaselines.algorithms.off_policy import FEEDFORWARD_PARAMS
from hbaselines.algorithms.off_policy import GOAL_CONDITIONED_PARAMS


class TestActorCriticPolicy(unittest.TestCase):
    """Test ActorCriticPolicy in hbaselines/fcnet/base.py."""

    def setUp(self):
        self.policy_params = {
            'sess': tf.compat.v1.Session(),
            'ac_space': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'ob_space': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'co_space': Box(low=-3, high=3, shape=(3,), dtype=np.float32),
            'verbose': 2,
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
        self.assertRaises(NotImplementedError, policy.value,
                          obs=None, context=None, action=None)
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
        ob_space = Box(0, 1, shape=(2,))
        co_space = None
        self.assertTupleEqual(policy._get_ob_dim(ob_space, co_space), (2,))

        # test case 2
        ob_space = Box(0, 1, shape=(2,))
        co_space = Box(0, 1, shape=(3,))
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
            'verbose': 2,
        }
        self.policy_params.update(TD3_PARAMS.copy())
        self.policy_params.update(FEEDFORWARD_PARAMS.copy())

    def tearDown(self):
        self.policy_params['sess'].close()
        del self.policy_params

    def test_init(self):
        """Validate that the graph and variables are initialized properly."""
        policy = TD3FeedForwardPolicy(**self.policy_params)

        # Check that the abstract class has all the required attributes.
        self.assertEqual(policy.buffer_size, self.policy_params['buffer_size'])
        self.assertEqual(policy.batch_size, self.policy_params['batch_size'])
        self.assertEqual(policy.actor_lr, self.policy_params['actor_lr'])
        self.assertEqual(policy.critic_lr, self.policy_params['critic_lr'])
        self.assertEqual(policy.verbose,  self.policy_params['verbose'])
        self.assertEqual(policy.tau,  self.policy_params['tau'])
        self.assertEqual(policy.gamma,  self.policy_params['gamma'])
        self.assertEqual(policy.layer_norm, self.policy_params['layer_norm'])
        self.assertListEqual(policy.layers, [256, 256])
        self.assertEqual(policy.act_fun, self.policy_params['act_fun'])

        # Check that all trainable variables have been created in the
        # TensorFlow graph.
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

        # Check that all the input placeholders were properly created.
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

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_initialize(self):
        """Test that the target variables are properly initialized."""
        pass  # TODO

    def test_optimization(self):
        """Test the losses and gradient update steps."""
        pass  # TODO

    def test_update_target(self):
        """Test the soft and init target updates."""
        pass  # TODO

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
            'verbose': 2,
        }
        self.policy_params.update(SAC_PARAMS.copy())
        self.policy_params.update(FEEDFORWARD_PARAMS.copy())

    def tearDown(self):
        self.policy_params['sess'].close()
        del self.policy_params

    def test_init(self):
        """Validate that the graph and variables are initialized properly."""
        # policy = SACFeedForwardPolicy(**self.policy_params)

        pass  # TODO

    def test_gaussian_likelihood(self):
        """TODO."""
        pass  # TODO

    def test_apply_squashing(self):
        """TODO."""
        pass  # TODO

    def test_initialize(self):
        """Test that the target variables are properly initialized."""
        pass  # TODO


class TestGoalConditionedPolicy(unittest.TestCase):
    """Test GoalConditionedPolicy in hbaselines/goal_conditioned/td3.py."""

    def setUp(self):
        self.policy_params = {
            'sess': tf.compat.v1.Session(),
            'ac_space': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'ob_space': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'co_space': Box(low=-3, high=3, shape=(3,), dtype=np.float32),
            'layers': None,
            'verbose': 2,
        }
        self.policy_params.update(TD3_PARAMS.copy())
        self.policy_params.update(GOAL_CONDITIONED_PARAMS.copy())

    def tearDown(self):
        self.policy_params['sess'].close()
        del self.policy_params

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_init(self):
        """Validate that the graph and variables are initialized properly."""
        policy = GoalConditionedPolicy(**self.policy_params)

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

        # Check that all trainable variables have been created in the
        # TensorFlow graph.
        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['Manager/model/pi/fc0/bias:0',
             'Manager/model/pi/fc0/kernel:0',
             'Manager/model/pi/fc1/bias:0',
             'Manager/model/pi/fc1/kernel:0',
             'Manager/model/pi/output/bias:0',
             'Manager/model/pi/output/kernel:0',
             'Manager/model/qf_0/fc0/bias:0',
             'Manager/model/qf_0/fc0/kernel:0',
             'Manager/model/qf_0/fc1/bias:0',
             'Manager/model/qf_0/fc1/kernel:0',
             'Manager/model/qf_0/qf_output/bias:0',
             'Manager/model/qf_0/qf_output/kernel:0',
             'Manager/model/qf_1/fc0/bias:0',
             'Manager/model/qf_1/fc0/kernel:0',
             'Manager/model/qf_1/fc1/bias:0',
             'Manager/model/qf_1/fc1/kernel:0',
             'Manager/model/qf_1/qf_output/bias:0',
             'Manager/model/qf_1/qf_output/kernel:0',
             'Manager/target/pi/fc0/bias:0',
             'Manager/target/pi/fc0/kernel:0',
             'Manager/target/pi/fc1/bias:0',
             'Manager/target/pi/fc1/kernel:0',
             'Manager/target/pi/output/bias:0',
             'Manager/target/pi/output/kernel:0',
             'Manager/target/qf_0/fc0/bias:0',
             'Manager/target/qf_0/fc0/kernel:0',
             'Manager/target/qf_0/fc1/bias:0',
             'Manager/target/qf_0/fc1/kernel:0',
             'Manager/target/qf_0/qf_output/bias:0',
             'Manager/target/qf_0/qf_output/kernel:0',
             'Manager/target/qf_1/fc0/bias:0',
             'Manager/target/qf_1/fc0/kernel:0',
             'Manager/target/qf_1/fc1/bias:0',
             'Manager/target/qf_1/fc1/kernel:0',
             'Manager/target/qf_1/qf_output/bias:0',
             'Manager/target/qf_1/qf_output/kernel:0',
             'Worker/model/pi/fc0/bias:0',
             'Worker/model/pi/fc0/kernel:0',
             'Worker/model/pi/fc1/bias:0',
             'Worker/model/pi/fc1/kernel:0',
             'Worker/model/pi/output/bias:0',
             'Worker/model/pi/output/kernel:0',
             'Worker/model/qf_0/fc0/bias:0',
             'Worker/model/qf_0/fc0/kernel:0',
             'Worker/model/qf_0/fc1/bias:0',
             'Worker/model/qf_0/fc1/kernel:0',
             'Worker/model/qf_0/qf_output/bias:0',
             'Worker/model/qf_0/qf_output/kernel:0',
             'Worker/model/qf_1/fc0/bias:0',
             'Worker/model/qf_1/fc0/kernel:0',
             'Worker/model/qf_1/fc1/bias:0',
             'Worker/model/qf_1/fc1/kernel:0',
             'Worker/model/qf_1/qf_output/bias:0',
             'Worker/model/qf_1/qf_output/kernel:0',
             'Worker/target/pi/fc0/bias:0',
             'Worker/target/pi/fc0/kernel:0',
             'Worker/target/pi/fc1/bias:0',
             'Worker/target/pi/fc1/kernel:0',
             'Worker/target/pi/output/bias:0',
             'Worker/target/pi/output/kernel:0',
             'Worker/target/qf_0/fc0/bias:0',
             'Worker/target/qf_0/fc0/kernel:0',
             'Worker/target/qf_0/fc1/bias:0',
             'Worker/target/qf_0/fc1/kernel:0',
             'Worker/target/qf_0/qf_output/bias:0',
             'Worker/target/qf_0/qf_output/kernel:0',
             'Worker/target/qf_1/fc0/bias:0',
             'Worker/target/qf_1/fc0/kernel:0',
             'Worker/target/qf_1/fc1/bias:0',
             'Worker/target/qf_1/fc1/kernel:0',
             'Worker/target/qf_1/qf_output/bias:0',
             'Worker/target/qf_1/qf_output/kernel:0']
        )

        # Test the worker_reward function.
        self.assertAlmostEqual(
            policy.worker_reward_fn(
                states=np.array([1, 2, 3]),
                goals=np.array([3, 2, 1]),
                next_states=np.array([0, 0, 0])
            ),
            -3.7416573867873044
        )

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_store_transition(self):
        """Test the `store_transition` method."""
        pass  # TODO

    def test_meta_period(self):
        """Verify that the rate of the Manager is dictated by meta_period."""
        # Test for a meta period of 5.
        policy_params = self.policy_params.copy()
        policy_params['meta_period'] = 5
        policy = GoalConditionedPolicy(**policy_params)

        # FIXME: add test
        del policy

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

        # Test for a meta period of 10.
        policy_params = self.policy_params.copy()
        policy_params['meta_period'] = 10
        policy = GoalConditionedPolicy(**policy_params)

        # FIXME: add test
        del policy

    def test_relative_goals(self):
        """Validate the functionality of relative goals.

        This should affect the worker reward function as well as transformation
        from relative goals to absolute goals.
        """
        policy_params = self.policy_params.copy()
        policy_params["relative_goals"] = True
        policy = GoalConditionedPolicy(**policy_params)

        # Test the updated reward function.
        states = np.array([1, 2, 3])
        goals = np.array([4, 5, 6])
        next_states = np.array([7, 8, 9])
        self.assertAlmostEqual(
            policy.worker_reward_fn(states, goals, next_states),
            -2.2360679775221506
        )

    def test_off_policy_corrections(self):
        """Validate the functionality of the off-policy corrections."""
        # Set a random variable seed.
        np.random.seed(1)
        random.seed(1)
        tf.compat.v1.set_random_seed(1)

        policy_params = self.policy_params.copy()
        policy_params["relative_goals"] = True
        policy_params["off_policy_corrections"] = True
        policy = GoalConditionedPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Test the _sample method.
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
        samples = policy._sample(states, next_states, num_samples, orig_goals)

        # Check that the shape is correct.
        self.assertTupleEqual(
            samples.shape, (states.shape[0], states.shape[1], num_samples))

        # Check the last few elements are the deterministic components that
        # they are expected to be.
        np.testing.assert_array_almost_equal(
            samples[:, :, -2:].reshape(states.shape[0] * states.shape[1], 2).T,
            np.vstack(
                [np.array([-2] * states.shape[0] * states.shape[1]),
                 orig_goals.flatten()]
            )
        )

        # Test the _log_probs method.
        manager_obs = np.array([[1, 2], [3, -1], [0, 0]])
        worker_obs = np.array([[1, 1], [2, 2], [3, 3]])
        actions = np.array([[1], [-1], [0]])
        goals = np.array([[0, 0], [-1, -1], [-2, -2]])
        error = policy._log_probs(manager_obs, worker_obs, actions, goals)
        np.testing.assert_array_almost_equal(
            error, [-3.907223e-03, -3.918726e-03, -7.482369e-08])

        # Test the _sample_best_meta_action method.  FIXME

    def test_connected_gradients(self):
        """Validate the functionality of the connected-gradients feature."""
        pass  # TODO

    def test_centralized_value_functions(self):
        """Validate the functionality of the centralized value function."""
        pass  # TODO


if __name__ == '__main__':
    unittest.main()
