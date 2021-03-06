"""Tests for the policies in the hbaselines/base_policies subdirectory."""
import unittest
import numpy as np
import tensorflow as tf
from gym.spaces import Box

from hbaselines.base_policies import Policy
from hbaselines.algorithms.rl_algorithm import FEEDFORWARD_PARAMS
from hbaselines.utils.tf_util import setup_target_updates


class TestPolicy(unittest.TestCase):
    """Test Policy in hbaselines/base_policies/policy.py."""

    def setUp(self):
        sess = tf.compat.v1.Session()

        self.policy_params = {
            'sess': sess,
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'ob_space': Box(low=-2, high=2, shape=(2,)),
            'co_space': Box(low=-3, high=3, shape=(3,)),
            'verbose': 0,
        }
        self.policy_params.update(FEEDFORWARD_PARAMS.copy())

    def tearDown(self):
        self.policy_params['sess'].close()
        del self.policy_params

    def test_init(self):
        """Validate that the variables are initialized properly."""
        policy = Policy(**self.policy_params)

        # Check that the abstract class has all the required attributes.
        self.assertEqual(policy.sess, self.policy_params['sess'])
        self.assertEqual(policy.ac_space, self.policy_params['ac_space'])
        self.assertEqual(policy.ob_space, self.policy_params['ob_space'])
        self.assertEqual(policy.co_space, self.policy_params['co_space'])
        self.assertEqual(policy.verbose, self.policy_params['verbose'])

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

    def test_init_assertions(self):
        """Test the assertions in the __init__ methods.

        This tests the following cases:

        1. the required model_params are not specified
        2. the required conv-related model_params are not specified
        3. the model_type is not an applicable one.
        """
        # test case 1
        policy_params = self.policy_params.copy()
        model_type = policy_params["model_params"]["model_type"]
        layers = policy_params["model_params"]["layers"]
        del policy_params["model_params"]["model_type"]
        del policy_params["model_params"]["layers"]
        self.assertRaises(AssertionError, Policy, **policy_params)

        # Undo changes.
        policy_params["model_params"]["model_type"] = model_type
        policy_params["model_params"]["layers"] = layers

        # test case 2
        policy_params = policy_params.copy()
        policy_params["model_params"]["model_type"] = "conv"
        strides = policy_params["model_params"]["strides"]
        filters = policy_params["model_params"]["filters"]
        del policy_params["model_params"]["strides"]
        del policy_params["model_params"]["filters"]
        self.assertRaises(AssertionError, Policy, **policy_params)

        # Undo changes.
        policy_params["model_params"]["strides"] = strides
        policy_params["model_params"]["filters"] = filters

        # test case 3
        policy_params = self.policy_params.copy()
        policy_params["model_params"]["model_type"] = "blank"
        self.assertRaises(AssertionError, Policy, **policy_params)

        # Undo changes.
        policy_params["model_params"]["model_type"] = "fcnet"

    def test_get_obs(self):
        """Check the functionality of the _get_obs() method.

        This method is tested for three cases:

        1. when the context is None
        2. for 1-D observations and contexts
        3. for 2-D observations and contexts
        """
        policy = Policy(**self.policy_params)

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
        policy = Policy(**self.policy_params)

        # test case 1
        ob_space = Box(0, 1, shape=(2,))
        co_space = None
        self.assertTupleEqual(policy._get_ob_dim(ob_space, co_space), (2,))

        # test case 2
        ob_space = Box(0, 1, shape=(2,))
        co_space = Box(0, 1, shape=(3,))
        self.assertTupleEqual(policy._get_ob_dim(ob_space, co_space), (5,))

    def test_setup_target_updates(self):
        """Check the functionality of the setup_target_updates() method.

        This test validates both the init and soft update procedures generated
        by the tested method.
        """
        policy = Policy(**self.policy_params)

        _ = tf.Variable(initial_value=[[1, 1, 1, 1]], dtype=tf.float32,
                        name="0")
        val1 = tf.Variable(initial_value=[[0, 0, 0, 0]], dtype=tf.float32,
                           name="1")

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        init, soft = setup_target_updates("0", "1", None, 0.1, 0)

        # test soft update
        policy.sess.run(soft)
        expected = np.array([[0.1, 0.1, 0.1, 0.1]])
        np.testing.assert_almost_equal(policy.sess.run(val1), expected)

        # test init update
        policy.sess.run(init)
        expected = np.array([[1, 1, 1, 1]])
        np.testing.assert_almost_equal(policy.sess.run(val1), expected)


if __name__ == '__main__':
    unittest.main()
