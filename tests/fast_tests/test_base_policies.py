"""Tests for the policies in the hbaselines/base_policies subdirectory."""
import unittest
import numpy as np
import tensorflow as tf
from gym.spaces import Box

from hbaselines.base_policies import ActorCriticPolicy
from hbaselines.base_policies import ImitationLearningPolicy
from hbaselines.algorithms.off_policy import FEEDFORWARD_PARAMS


class TestActorCriticPolicy(unittest.TestCase):
    """Test ActorCriticPolicy in hbaselines/base_policies/actor_critic.py."""

    def setUp(self):
        sess = tf.compat.v1.Session()

        self.policy_params = {
            'sess': sess,
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'ob_space': Box(low=-2, high=2, shape=(2,)),
            'co_space': Box(low=-3, high=3, shape=(3,)),
            'verbose': 0,
            'buffer_size': 200000,
            'batch_size': 128,
            'actor_lr': 3e-4,
            'critic_lr': 3e-4,
            'tau': 0.005,
            'gamma': 0.99,
            'use_huber': False,
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
        self.assertEqual(policy.tau, self.policy_params['tau'])
        self.assertEqual(policy.gamma, self.policy_params['gamma'])
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
        self.assertRaises(AssertionError, ActorCriticPolicy, **policy_params)

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
        self.assertRaises(AssertionError, ActorCriticPolicy, **policy_params)

        # Undo changes.
        policy_params["model_params"]["strides"] = strides
        policy_params["model_params"]["filters"] = filters

        # test case 3
        policy_params = self.policy_params.copy()
        policy_params["model_params"]["model_type"] = "blank"
        self.assertRaises(AssertionError, ActorCriticPolicy, **policy_params)

        # Undo changes.
        policy_params["model_params"]["model_type"] = "fcnet"

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


class TestImitationLearningPolicy(unittest.TestCase):
    """Test ImitationLearningPolicy in hbaselines/base_policies."""

    def setUp(self):
        sess = tf.compat.v1.Session()

        self.policy_params = {
            'sess': sess,
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'ob_space': Box(low=-2, high=2, shape=(2,)),
            'co_space': Box(low=-3, high=3, shape=(3,)),
            'verbose': 0,
        }
        self.policy_params.update({
            "buffer_size": 200000,
            "batch_size": 128,
            "learning_rate": 3e-4,
            "layer_norm": False,
            "layers": [256, 256],
            "act_fun": tf.nn.relu,
            "use_huber": False,
            "stochastic": False
        })

    def tearDown(self):
        self.policy_params['sess'].close()
        del self.policy_params

    def test_init(self):
        """Validate that the variables are initialized properly."""
        policy = ImitationLearningPolicy(**self.policy_params)

        # Check that the abstract class has all the required attributes.
        self.assertEqual(policy.sess, self.policy_params['sess'])
        self.assertEqual(policy.ob_space, self.policy_params['ob_space'])
        self.assertEqual(policy.ac_space, self.policy_params['ac_space'])
        self.assertEqual(policy.co_space, self.policy_params['co_space'])
        self.assertEqual(policy.buffer_size, self.policy_params['buffer_size'])
        self.assertEqual(policy.batch_size, self.policy_params['batch_size'])
        self.assertEqual(policy.learning_rate,
                         self.policy_params['learning_rate'])
        self.assertEqual(policy.verbose, self.policy_params['verbose'])
        self.assertEqual(policy.layer_norm, self.policy_params['layer_norm'])
        self.assertEqual(policy.layers, self.policy_params['layers'])
        self.assertEqual(policy.act_fun, self.policy_params['act_fun'])
        self.assertEqual(policy.use_huber, self.policy_params['use_huber'])
        self.assertEqual(policy.stochastic, self.policy_params['stochastic'])

        # Check that the abstract class has all the required methods.
        self.assertRaises(NotImplementedError, policy.update)
        self.assertRaises(NotImplementedError, policy.get_action,
                          obs=None, context=None)
        self.assertRaises(NotImplementedError, policy.store_transition,
                          obs0=None, context0=None, action=None, obs1=None,
                          context1=None)
        self.assertRaises(NotImplementedError, policy.get_td_map)

    def test_get_obs(self):
        """Check the functionality of the _get_obs() method.

        This method is tested for three cases:

        1. when the context is None
        2. for 1-D observations and contexts
        3. for 2-D observations and contexts
        """
        policy = ImitationLearningPolicy(**self.policy_params)

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
        policy = ImitationLearningPolicy(**self.policy_params)

        # test case 1
        ob_space = Box(0, 1, shape=(2,))
        co_space = None
        self.assertTupleEqual(policy._get_ob_dim(ob_space, co_space), (2,))

        # test case 2
        ob_space = Box(0, 1, shape=(2,))
        co_space = Box(0, 1, shape=(3,))
        self.assertTupleEqual(policy._get_ob_dim(ob_space, co_space), (5,))


if __name__ == '__main__':
    unittest.main()
