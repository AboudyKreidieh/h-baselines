"""Contains tests for the model abstractions and different models."""
import unittest
import numpy as np
import tensorflow as tf
import random
from gym.spaces import Box
from hbaselines.hiro.tf_util import get_trainable_vars
from hbaselines.hiro.policy import ActorCriticPolicy, FeedForwardPolicy
from hbaselines.hiro.policy import GoalDirectedPolicy


class TestActorCriticPolicy(unittest.TestCase):
    """Test the FeedForwardPolicy object in hbaselines/hiro/policy.py."""

    def setUp(self):
        self.policy_params = {
            'sess': tf.Session(),
            'ac_space': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'ob_space': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'co_space': Box(low=-3, high=3, shape=(3,), dtype=np.float32),
        }

    def tearDown(self):
        self.policy_params['sess'].close()
        del self.policy_params

    def test_init(self):
        """Validate that the graph and variables are initialized properly."""
        policy = ActorCriticPolicy(**self.policy_params)

        # Check that the abstract class has all the required attributes.
        self.assertEqual(policy.sess, self.policy_params['sess'])
        self.assertEqual(policy.ac_space, self.policy_params['ac_space'])
        self.assertEqual(policy.ob_space, self.policy_params['ob_space'])
        self.assertEqual(policy.co_space, self.policy_params['co_space'])

        # Check that the abstract class has all the required methods.
        self.assertTrue(hasattr(policy, "initialize"))
        self.assertTrue(hasattr(policy, "update"))
        self.assertTrue(hasattr(policy, "get_action"))
        self.assertTrue(hasattr(policy, "value"))
        self.assertTrue(hasattr(policy, "store_transition"))
        self.assertTrue(hasattr(policy, "get_stats"))


class TestFeedForwardPolicy(unittest.TestCase):
    """Test the FeedForwardPolicy object in hbaselines/hiro/policy.py."""

    def setUp(self):
        self.policy_params = {
            'sess': tf.Session(),
            'ac_space': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'ob_space': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'co_space': Box(low=-3, high=3, shape=(3,), dtype=np.float32),
            'buffer_size': 1e6,
            'batch_size': 100,
            'actor_lr': 1e-3,
            'critic_lr': 1e-4,
            'verbose': 2,
            'tau': 0.005,
            'gamma': 0.001,
            'layer_norm': False,
            'reuse': False,
            'layers': None,
            'act_fun': tf.nn.relu,
            'scope': None
        }

    def tearDown(self):
        self.policy_params['sess'].close()
        del self.policy_params

    def test_init(self):
        """Validate that the graph and variables are initialized properly."""
        policy = FeedForwardPolicy(**self.policy_params)

        # Check that the abstract class has all the required attributes.
        self.assertEqual(policy.buffer_size, self.policy_params['buffer_size'])
        self.assertEqual(policy.batch_size, self.policy_params['batch_size'])
        self.assertEqual(policy.actor_lr, self.policy_params['actor_lr'])
        self.assertEqual(policy.critic_lr, self.policy_params['critic_lr'])
        self.assertEqual(policy.verbose,  self.policy_params['verbose'])
        self.assertEqual(policy.tau,  self.policy_params['tau'])
        self.assertEqual(policy.gamma,  self.policy_params['gamma'])
        self.assertEqual(policy.layer_norm, self.policy_params['layer_norm'])
        self.assertEqual(policy.reuse, self.policy_params['reuse'])
        self.assertListEqual(policy.layers, [256, 256])
        self.assertEqual(policy.activ, self.policy_params['act_fun'])

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
            tuple(v.__int__() for v in policy.critic_target.shape),
            (None, 1))
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

    def test_optimization(self):
        """Test the losses and gradient update steps."""
        pass

    def test_update_target(self):
        """Test the soft and init target updates."""
        pass

    def test_store_transition(self):
        """Test the `store_transition` method."""
        pass


class TestGoalDirectedPolicy(unittest.TestCase):
    """Test the GoalDirectedPolicy object in hbaselines/hiro/policy.py."""

    def setUp(self):
        self.policy_params = {
            'sess': tf.Session(),
            'ac_space': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'ob_space': Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'co_space': Box(low=-3, high=3, shape=(3,), dtype=np.float32),
            'buffer_size': 1e6,
            'batch_size': 100,
            'actor_lr': 1e-3,
            'critic_lr': 1e-4,
            'verbose': 2,
            'tau': 0.005,
            'gamma': 0.001,
            'layer_norm': False,
            'reuse': False,
            'layers': None,
            'act_fun': tf.nn.relu,
            'meta_period': 10,
            'relative_goals': False,
            'off_policy_corrections': False,
            'use_fingerprints': False,
            'centralized_value_functions': False,
            'connected_gradients': False
        }

    def tearDown(self):
        self.policy_params['sess'].close()
        del self.policy_params

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_init(self):
        """Validate that the graph and variables are initialized properly."""
        policy = GoalDirectedPolicy(**self.policy_params)

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
            policy.worker_reward(
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
        pass

    def test_meta_period(self):
        """Verify that the rate of the Manager is dictated by meta_period."""
        # Test for a meta period of 5.
        policy_params = self.policy_params.copy()
        policy_params['meta_period'] = 5
        policy = GoalDirectedPolicy(**policy_params)

        # FIXME: add test
        del policy

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

        # Test for a meta period of 10.
        policy_params = self.policy_params.copy()
        policy_params['meta_period'] = 10
        policy = GoalDirectedPolicy(**policy_params)

        # FIXME: add test
        del policy

    def test_relative_goals(self):
        """Validate the functionality of relative goals.

        This should affect the worker reward function as well as transformation
        from relative goals to absolute goals.
        """
        policy_params = self.policy_params.copy()
        policy_params["relative_goals"] = True
        policy = GoalDirectedPolicy(**policy_params)

        # Test the updated reward function.
        states = np.array([1, 2, 3])
        goals = np.array([4, 5, 6])
        next_states = np.array([7, 8, 9])
        self.assertAlmostEqual(
            policy.worker_reward(states, goals, next_states),
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
        policy = GoalDirectedPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.global_variables_initializer())

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
            error, [-3.923483e-03, -3.844697e-03, -6.648080e-07])

        # Test the _sample_best_meta_action method.  FIXME

    def test_centralized_value_functions(self):
        """Validate the functionality of the centralized value function.

        TODO: describe content
        """
        pass

    def test_connected_gradients(self):
        """Validate the functionality of the connected-gradients feature.

        TODO: describe content
        """
        pass


if __name__ == '__main__':
    unittest.main()
