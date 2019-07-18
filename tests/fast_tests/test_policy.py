"""Contains tests for the model abstractions and different models."""
import unittest
import numpy as np
import tensorflow as tf
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
            'clip_norm': 0,
            'critic_l2_reg': 0,
            'verbose': False,
            'tau': 0.005,
            'gamma': 0.001,
            'normalize_observations': False,
            'observation_range': (-5, 5),
            'normalize_returns': False,
            'return_range': (-5, 5),
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
        self.assertEqual(policy.clip_norm, self.policy_params['clip_norm'])
        self.assertEqual(
            policy.critic_l2_reg,  self.policy_params['critic_l2_reg'])
        self.assertEqual(policy.verbose,  self.policy_params['verbose'])
        self.assertEqual(policy.tau,  self.policy_params['tau'])
        self.assertEqual(policy.gamma,  self.policy_params['gamma'])
        self.assertEqual(
            policy.normalize_observations,
            self.policy_params['normalize_observations'])
        self.assertEqual(
            policy.observation_range, self.policy_params['observation_range'])
        self.assertEqual(
            policy.normalize_returns, self.policy_params['normalize_returns'])
        self.assertEqual(
            policy.return_range, self.policy_params['return_range'])
        self.assertEqual(policy.layer_norm, self.policy_params['layer_norm'])
        self.assertEqual(policy.reuse, self.policy_params['reuse'])
        self.assertListEqual(policy.layers, [300, 300])
        self.assertEqual(policy.activ, self.policy_params['act_fun'])

        # Check that all variables have been created in the tensorflow graph.
        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['model/pi/fc0/bias:0', 'model/pi/fc0/kernel:0',
             'model/pi/fc1/bias:0',
             'model/pi/fc1/kernel:0',
             'model/pi/pi/bias:0',
             'model/pi/pi/kernel:0',
             'model/qf/fc0/bias:0',
             'model/qf/fc0/kernel:0',
             'model/qf/fc1/bias:0',
             'model/qf/fc1/kernel:0',
             'model/qf/qf_output/bias:0',
             'model/qf/qf_output/kernel:0',
             'target/pi/fc0/bias:0',
             'target/pi/fc0/kernel:0',
             'target/pi/fc1/bias:0',
             'target/pi/fc1/kernel:0',
             'target/pi/pi/bias:0',
             'target/pi/pi/kernel:0',
             'target/qf/fc0/bias:0',
             'target/qf/fc0/kernel:0',
             'target/qf/fc1/bias:0',
             'target/qf/fc1/kernel:0',
             'target/qf/qf_output/bias:0',
             'target/qf/qf_output/kernel:0']
        )


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
            'clip_norm': 0,
            'critic_l2_reg': 0,
            'verbose': False,
            'tau': 0.005,
            'gamma': 0.001,
            'normalize_observations': False,
            'observation_range': (-5, 5),
            'normalize_returns': False,
            'return_range': (-5, 5),
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

    def test_init(self):
        """Validate that the graph and variables are initialized properly."""
        pass
        # policy = GoalDirectedPolicy(**self.policy_params)

    def test_meta_period(self):
        """Verify that the rate of the Manager is dictated by meta_period."""
        pass

    def test_relative_goals(self):
        """Validate the functionality of relative goals.

        This should affect the worker reward function as well as TODO.
        """
        pass

    def test_off_policy_corrections(self):
        """Validate the functionality of the off-policy corrections.

        TODO: describe content
        """
        pass

    def test_fingerprints(self):
        """Validate the functionality of the fingerprints.

        This feature should TODO: describe content
        """
        pass

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
