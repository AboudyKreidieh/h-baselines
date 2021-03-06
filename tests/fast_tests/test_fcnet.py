"""Tests for the policies in the hbaselines/fcnet subdirectory."""
import unittest
import numpy as np
import tensorflow as tf
from gym.spaces import Box
from copy import deepcopy

from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.fcnet.td3 import FeedForwardPolicy as TD3FeedForwardPolicy
from hbaselines.fcnet.sac import FeedForwardPolicy as SACFeedForwardPolicy
from hbaselines.fcnet.ppo import FeedForwardPolicy as PPOFeedForwardPolicy
from hbaselines.fcnet.trpo import FeedForwardPolicy as TRPOFeedForwardPolicy
from hbaselines.algorithms.rl_algorithm import TD3_PARAMS
from hbaselines.algorithms.rl_algorithm import SAC_PARAMS
from hbaselines.algorithms.rl_algorithm import PPO_PARAMS
from hbaselines.algorithms.rl_algorithm import TRPO_PARAMS
from hbaselines.algorithms.rl_algorithm import FEEDFORWARD_PARAMS


class TestTD3FeedForwardPolicy(unittest.TestCase):
    """Test FeedForwardPolicy in hbaselines/fcnet/td3.py."""

    def setUp(self):
        self.policy_params = {
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'ob_space': Box(low=-2, high=2, shape=(2,)),
            'co_space': Box(low=-3, high=3, shape=(3,)),
            'scope': None,
            'verbose': 0,
        }
        self.policy_params.update(TD3_PARAMS.copy())
        self.policy_params.update(FEEDFORWARD_PARAMS.copy())
        self.policy_params["model_params"]["model_type"] = "fcnet"

    def tearDown(self):
        del self.policy_params

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_init(self):
        """Check the functionality of the __init__() method.

        This method is tested for the following features:

        1. The proper structure graph was generated.
        2. All input placeholders are correct.
        """
        policy_params = deepcopy(self.policy_params)
        policy_params['sess'] = tf.compat.v1.Session()
        policy = TD3FeedForwardPolicy(**policy_params)

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

        # Kill the session,
        policy_params['sess'].close()

    def test_init_conv(self):
        """Check the functionality of the __init__() method with conv policies.

        This method tests that the proper structure graph was generated.
        """
        policy_params = deepcopy(self.policy_params)
        policy_params['sess'] = tf.compat.v1.Session()
        policy_params["model_params"]["model_type"] = "conv"
        _ = TD3FeedForwardPolicy(**policy_params)

        # test case 1
        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['model/pi/conv0/bias:0',
             'model/pi/conv0/kernel:0',
             'model/pi/conv1/bias:0',
             'model/pi/conv1/kernel:0',
             'model/pi/conv2/bias:0',
             'model/pi/conv2/kernel:0',
             'model/pi/fc0/bias:0',
             'model/pi/fc0/kernel:0',
             'model/pi/fc1/bias:0',
             'model/pi/fc1/kernel:0',
             'model/pi/output/bias:0',
             'model/pi/output/kernel:0',
             'model/qf_0/conv0/bias:0',
             'model/qf_0/conv0/kernel:0',
             'model/qf_0/conv1/bias:0',
             'model/qf_0/conv1/kernel:0',
             'model/qf_0/conv2/bias:0',
             'model/qf_0/conv2/kernel:0',
             'model/qf_0/fc0/bias:0',
             'model/qf_0/fc0/kernel:0',
             'model/qf_0/fc1/bias:0',
             'model/qf_0/fc1/kernel:0',
             'model/qf_0/qf_output/bias:0',
             'model/qf_0/qf_output/kernel:0',
             'model/qf_1/conv0/bias:0',
             'model/qf_1/conv0/kernel:0',
             'model/qf_1/conv1/bias:0',
             'model/qf_1/conv1/kernel:0',
             'model/qf_1/conv2/bias:0',
             'model/qf_1/conv2/kernel:0',
             'model/qf_1/fc0/bias:0',
             'model/qf_1/fc0/kernel:0',
             'model/qf_1/fc1/bias:0',
             'model/qf_1/fc1/kernel:0',
             'model/qf_1/qf_output/bias:0',
             'model/qf_1/qf_output/kernel:0',
             'target/pi/conv0/bias:0',
             'target/pi/conv0/kernel:0',
             'target/pi/conv1/bias:0',
             'target/pi/conv1/kernel:0',
             'target/pi/conv2/bias:0',
             'target/pi/conv2/kernel:0',
             'target/pi/fc0/bias:0',
             'target/pi/fc0/kernel:0',
             'target/pi/fc1/bias:0',
             'target/pi/fc1/kernel:0',
             'target/pi/output/bias:0',
             'target/pi/output/kernel:0',
             'target/qf_0/conv0/bias:0',
             'target/qf_0/conv0/kernel:0',
             'target/qf_0/conv1/bias:0',
             'target/qf_0/conv1/kernel:0',
             'target/qf_0/conv2/bias:0',
             'target/qf_0/conv2/kernel:0',
             'target/qf_0/fc0/bias:0',
             'target/qf_0/fc0/kernel:0',
             'target/qf_0/fc1/bias:0',
             'target/qf_0/fc1/kernel:0',
             'target/qf_0/qf_output/bias:0',
             'target/qf_0/qf_output/kernel:0',
             'target/qf_1/conv0/bias:0',
             'target/qf_1/conv0/kernel:0',
             'target/qf_1/conv1/bias:0',
             'target/qf_1/conv1/kernel:0',
             'target/qf_1/conv2/bias:0',
             'target/qf_1/conv2/kernel:0',
             'target/qf_1/fc0/bias:0',
             'target/qf_1/fc0/kernel:0',
             'target/qf_1/fc1/bias:0',
             'target/qf_1/fc1/kernel:0',
             'target/qf_1/qf_output/bias:0',
             'target/qf_1/qf_output/kernel:0']
        )

        # Kill the session,
        policy_params['sess'].close()

    def test_initialize(self):
        """Check the functionality of the initialize() method.

        This test validates that the target variables are properly initialized
        when initialize is called.
        """
        policy_params = deepcopy(self.policy_params)
        policy_params['sess'] = tf.compat.v1.Session()
        policy = TD3FeedForwardPolicy(**policy_params)

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

        # Kill the session,
        policy_params['sess'].close()


class TestSACFeedForwardPolicy(unittest.TestCase):
    """Test FeedForwardPolicy in hbaselines/fcnet/sac.py."""

    def setUp(self):
        self.policy_params = {
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'ob_space': Box(low=-2, high=2, shape=(2,)),
            'co_space': Box(low=-3, high=3, shape=(3,)),
            'scope': None,
            'verbose': 0,
        }
        self.policy_params.update(SAC_PARAMS.copy())
        self.policy_params.update(FEEDFORWARD_PARAMS.copy())
        self.policy_params["model_params"]["model_type"] = "fcnet"

    def tearDown(self):
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
        policy_params = deepcopy(self.policy_params)
        policy_params['sess'] = tf.compat.v1.Session()
        policy = SACFeedForwardPolicy(**policy_params)

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
        policy_params = deepcopy(self.policy_params)
        policy_params['sess'] = tf.compat.v1.Session()
        policy_params['target_entropy'] = 5
        policy = SACFeedForwardPolicy(**policy_params)
        self.assertEqual(policy.target_entropy,
                         policy_params['target_entropy'])

        # Kill the session,
        policy_params['sess'].close()

    def test_init_conv(self):
        """Check the functionality of the __init__() method with conv policies.

        This method tests that the proper structure graph was generated.
        """
        policy_params = deepcopy(self.policy_params)
        policy_params['sess'] = tf.compat.v1.Session()
        policy_params["model_params"]["model_type"] = "conv"
        _ = SACFeedForwardPolicy(**policy_params)

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['model/log_alpha:0',
             'model/pi/conv0/bias:0',
             'model/pi/conv0/kernel:0',
             'model/pi/conv1/bias:0',
             'model/pi/conv1/kernel:0',
             'model/pi/conv2/bias:0',
             'model/pi/conv2/kernel:0',
             'model/pi/fc0/bias:0',
             'model/pi/fc0/kernel:0',
             'model/pi/fc1/bias:0',
             'model/pi/fc1/kernel:0',
             'model/pi/log_std/bias:0',
             'model/pi/log_std/kernel:0',
             'model/pi/mean/bias:0',
             'model/pi/mean/kernel:0',
             'model/value_fns/qf1/conv0/bias:0',
             'model/value_fns/qf1/conv0/kernel:0',
             'model/value_fns/qf1/conv1/bias:0',
             'model/value_fns/qf1/conv1/kernel:0',
             'model/value_fns/qf1/conv2/bias:0',
             'model/value_fns/qf1/conv2/kernel:0',
             'model/value_fns/qf1/fc0/bias:0',
             'model/value_fns/qf1/fc0/kernel:0',
             'model/value_fns/qf1/fc1/bias:0',
             'model/value_fns/qf1/fc1/kernel:0',
             'model/value_fns/qf1/qf_output/bias:0',
             'model/value_fns/qf1/qf_output/kernel:0',
             'model/value_fns/qf2/conv0/bias:0',
             'model/value_fns/qf2/conv0/kernel:0',
             'model/value_fns/qf2/conv1/bias:0',
             'model/value_fns/qf2/conv1/kernel:0',
             'model/value_fns/qf2/conv2/bias:0',
             'model/value_fns/qf2/conv2/kernel:0',
             'model/value_fns/qf2/fc0/bias:0',
             'model/value_fns/qf2/fc0/kernel:0',
             'model/value_fns/qf2/fc1/bias:0',
             'model/value_fns/qf2/fc1/kernel:0',
             'model/value_fns/qf2/qf_output/bias:0',
             'model/value_fns/qf2/qf_output/kernel:0',
             'model/value_fns/vf/conv0/bias:0',
             'model/value_fns/vf/conv0/kernel:0',
             'model/value_fns/vf/conv1/bias:0',
             'model/value_fns/vf/conv1/kernel:0',
             'model/value_fns/vf/conv2/bias:0',
             'model/value_fns/vf/conv2/kernel:0',
             'model/value_fns/vf/fc0/bias:0',
             'model/value_fns/vf/fc0/kernel:0',
             'model/value_fns/vf/fc1/bias:0',
             'model/value_fns/vf/fc1/kernel:0',
             'model/value_fns/vf/vf_output/bias:0',
             'model/value_fns/vf/vf_output/kernel:0',
             'target/value_fns/vf/conv0/bias:0',
             'target/value_fns/vf/conv0/kernel:0',
             'target/value_fns/vf/conv1/bias:0',
             'target/value_fns/vf/conv1/kernel:0',
             'target/value_fns/vf/conv2/bias:0',
             'target/value_fns/vf/conv2/kernel:0',
             'target/value_fns/vf/fc0/bias:0',
             'target/value_fns/vf/fc0/kernel:0',
             'target/value_fns/vf/fc1/bias:0',
             'target/value_fns/vf/fc1/kernel:0',
             'target/value_fns/vf/vf_output/bias:0',
             'target/value_fns/vf/vf_output/kernel:0']
        )

        # Kill the session,
        policy_params['sess'].close()

    def test_initialize(self):
        """Check the functionality of the initialize() method.

        This test validates that the target variables are properly initialized
        when initialize is called.
        """
        policy_params = deepcopy(self.policy_params)
        policy_params['sess'] = tf.compat.v1.Session()
        policy = SACFeedForwardPolicy(**policy_params)

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

        # Kill the session,
        policy_params['sess'].close()


class TestPPOFeedForwardPolicy(unittest.TestCase):
    """Test FeedForwardPolicy in hbaselines/fcnet/ppo.py."""

    def setUp(self):
        self.policy_params = {
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'ob_space': Box(low=-2, high=2, shape=(2,)),
            'co_space': Box(low=-3, high=3, shape=(3,)),
            'scope': None,
            'verbose': 0,
        }
        self.policy_params.update(PPO_PARAMS.copy())
        self.policy_params.update(FEEDFORWARD_PARAMS.copy())
        self.policy_params["model_params"]["model_type"] = "fcnet"

    def tearDown(self):
        del self.policy_params

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_init(self):
        """Check the functionality of the __init__() method.

        This method is tested for the following features:

        1. The proper structure graph was generated.
        2. All input placeholders are correct.
        """
        policy_params = deepcopy(self.policy_params)
        policy_params['sess'] = tf.compat.v1.Session()
        policy = PPOFeedForwardPolicy(**policy_params)

        # test case 1
        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['model/logstd:0',
             'model/pi/fc0/bias:0',
             'model/pi/fc0/kernel:0',
             'model/pi/fc1/bias:0',
             'model/pi/fc1/kernel:0',
             'model/pi/output/bias:0',
             'model/pi/output/kernel:0',
             'model/vf/fc0/bias:0',
             'model/vf/fc0/kernel:0',
             'model/vf/fc1/bias:0',
             'model/vf/fc1/kernel:0',
             'model/vf/output/bias:0',
             'model/vf/output/kernel:0']
        )

        # test case 2
        self.assertEqual(
            tuple(v.__int__() for v in policy.rew_ph.shape),
            (None,))
        self.assertEqual(
            tuple(v.__int__() for v in policy.action_ph.shape),
            (None, 1))
        self.assertEqual(
            tuple(v.__int__() for v in policy.obs_ph.shape),
            (None, 5))
        self.assertEqual(
            tuple(v.__int__() for v in policy.advs_ph.shape),
            (None,))
        self.assertEqual(
            tuple(v.__int__() for v in policy.old_neglog_pac_ph.shape),
            (None,))
        self.assertEqual(
            tuple(v.__int__() for v in policy.old_vpred_ph.shape),
            (None,))

        # Kill the session,
        policy_params['sess'].close()

    def test_init_conv(self):
        """Check the functionality of the __init__() method with conv policies.

        This method tests that the proper structure graph was generated.
        """
        policy_params = deepcopy(self.policy_params)
        policy_params['sess'] = tf.compat.v1.Session()
        policy_params["model_params"]["model_type"] = "conv"
        _ = PPOFeedForwardPolicy(**policy_params)

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['model/logstd:0',
             'model/pi/conv0/bias:0',
             'model/pi/conv0/kernel:0',
             'model/pi/conv1/bias:0',
             'model/pi/conv1/kernel:0',
             'model/pi/conv2/bias:0',
             'model/pi/conv2/kernel:0',
             'model/pi/fc0/bias:0',
             'model/pi/fc0/kernel:0',
             'model/pi/fc1/bias:0',
             'model/pi/fc1/kernel:0',
             'model/pi/output/bias:0',
             'model/pi/output/kernel:0',
             'model/vf/conv0/bias:0',
             'model/vf/conv0/kernel:0',
             'model/vf/conv1/bias:0',
             'model/vf/conv1/kernel:0',
             'model/vf/conv2/bias:0',
             'model/vf/conv2/kernel:0',
             'model/vf/fc0/bias:0',
             'model/vf/fc0/kernel:0',
             'model/vf/fc1/bias:0',
             'model/vf/fc1/kernel:0',
             'model/vf/output/bias:0',
             'model/vf/output/kernel:0']
        )

        # Kill the session,
        policy_params['sess'].close()


class TestTRPOFeedForwardPolicy(unittest.TestCase):
    """Test FeedForwardPolicy in hbaselines/fcnet/trpo.py."""

    def setUp(self):
        self.policy_params = {
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'ob_space': Box(low=-2, high=2, shape=(2,)),
            'co_space': Box(low=-3, high=3, shape=(3,)),
            'scope': None,
            'verbose': 0,
        }
        self.policy_params.update(TRPO_PARAMS.copy())
        self.policy_params.update(FEEDFORWARD_PARAMS.copy())
        self.policy_params["model_params"]["model_type"] = "fcnet"

    def tearDown(self):
        del self.policy_params

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_init(self):
        """Check the functionality of the __init__() method.

        This method is tested for the following features:

        1. The proper structure graph was generated.
        2. All input placeholders are correct.
        """
        policy_params = deepcopy(self.policy_params)
        policy_params['sess'] = tf.compat.v1.Session()
        policy = TRPOFeedForwardPolicy(**policy_params)

        # test case 1
        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['model/logstd:0',
             'model/pi/fc0/bias:0',
             'model/pi/fc0/kernel:0',
             'model/pi/fc1/bias:0',
             'model/pi/fc1/kernel:0',
             'model/pi/output/bias:0',
             'model/pi/output/kernel:0',
             'model/vf/fc0/bias:0',
             'model/vf/fc0/kernel:0',
             'model/vf/fc1/bias:0',
             'model/vf/fc1/kernel:0',
             'model/vf/output/bias:0',
             'model/vf/output/kernel:0',
             'oldpi/model/logstd:0',
             'oldpi/model/pi/fc0/bias:0',
             'oldpi/model/pi/fc0/kernel:0',
             'oldpi/model/pi/fc1/bias:0',
             'oldpi/model/pi/fc1/kernel:0',
             'oldpi/model/pi/output/bias:0',
             'oldpi/model/pi/output/kernel:0',
             'oldpi/model/vf/fc0/bias:0',
             'oldpi/model/vf/fc0/kernel:0',
             'oldpi/model/vf/fc1/bias:0',
             'oldpi/model/vf/fc1/kernel:0',
             'oldpi/model/vf/output/bias:0',
             'oldpi/model/vf/output/kernel:0']
        )

        # test case 2
        self.assertEqual(
            tuple(v.__int__() for v in policy.ret_ph.shape),
            (None,))
        self.assertEqual(
            tuple(v.__int__() for v in policy.action_ph.shape),
            (None, 1))
        self.assertEqual(
            tuple(v.__int__() for v in policy.obs_ph.shape),
            (None, 5))
        self.assertEqual(
            tuple(v.__int__() for v in policy.advs_ph.shape),
            (None,))
        self.assertEqual(
            tuple(v.__int__() for v in policy.old_vpred_ph.shape),
            (None,))

        # Kill the session,
        policy_params['sess'].close()

    def test_init_conv(self):
        """Check the functionality of the __init__() method with conv policies.

        This method tests that the proper structure graph was generated.
        """
        policy_params = deepcopy(self.policy_params)
        policy_params['sess'] = tf.compat.v1.Session()
        policy_params["model_params"]["model_type"] = "conv"
        _ = TRPOFeedForwardPolicy(**policy_params)

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['model/logstd:0',
             'model/pi/conv0/bias:0',
             'model/pi/conv0/kernel:0',
             'model/pi/conv1/bias:0',
             'model/pi/conv1/kernel:0',
             'model/pi/conv2/bias:0',
             'model/pi/conv2/kernel:0',
             'model/pi/fc0/bias:0',
             'model/pi/fc0/kernel:0',
             'model/pi/fc1/bias:0',
             'model/pi/fc1/kernel:0',
             'model/pi/output/bias:0',
             'model/pi/output/kernel:0',
             'model/vf/conv0/bias:0',
             'model/vf/conv0/kernel:0',
             'model/vf/conv1/bias:0',
             'model/vf/conv1/kernel:0',
             'model/vf/conv2/bias:0',
             'model/vf/conv2/kernel:0',
             'model/vf/fc0/bias:0',
             'model/vf/fc0/kernel:0',
             'model/vf/fc1/bias:0',
             'model/vf/fc1/kernel:0',
             'model/vf/output/bias:0',
             'model/vf/output/kernel:0',
             'oldpi/model/logstd:0',
             'oldpi/model/pi/conv0/bias:0',
             'oldpi/model/pi/conv0/kernel:0',
             'oldpi/model/pi/conv1/bias:0',
             'oldpi/model/pi/conv1/kernel:0',
             'oldpi/model/pi/conv2/bias:0',
             'oldpi/model/pi/conv2/kernel:0',
             'oldpi/model/pi/fc0/bias:0',
             'oldpi/model/pi/fc0/kernel:0',
             'oldpi/model/pi/fc1/bias:0',
             'oldpi/model/pi/fc1/kernel:0',
             'oldpi/model/pi/output/bias:0',
             'oldpi/model/pi/output/kernel:0',
             'oldpi/model/vf/conv0/bias:0',
             'oldpi/model/vf/conv0/kernel:0',
             'oldpi/model/vf/conv1/bias:0',
             'oldpi/model/vf/conv1/kernel:0',
             'oldpi/model/vf/conv2/bias:0',
             'oldpi/model/vf/conv2/kernel:0',
             'oldpi/model/vf/fc0/bias:0',
             'oldpi/model/vf/fc0/kernel:0',
             'oldpi/model/vf/fc1/bias:0',
             'oldpi/model/vf/fc1/kernel:0',
             'oldpi/model/vf/output/bias:0',
             'oldpi/model/vf/output/kernel:0']
        )

        # Kill the session,
        policy_params['sess'].close()


if __name__ == '__main__':
    unittest.main()
