"""Tests for the policies in the hbaselines/multi_fcnet subdirectory."""
import unittest
import numpy as np
import tensorflow as tf
from gym.spaces import Box

from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.multi_fcnet.td3 import MultiFeedForwardPolicy as \
    TD3MultiFeedForwardPolicy
from hbaselines.multi_fcnet.sac import MultiFeedForwardPolicy as \
    SACMultiFeedForwardPolicy
from hbaselines.algorithms.off_policy import SAC_PARAMS, TD3_PARAMS
from hbaselines.algorithms.off_policy import MULTI_FEEDFORWARD_PARAMS


class TestBaseMultiFeedForwardPolicy(unittest.TestCase):
    """Test MultiFeedForwardPolicy in hbaselines/multi_fcnet/base.py."""

    def setUp(self):
        self.sess = tf.compat.v1.Session()

        # Shared policy parameters
        self.policy_params_shared = {
            'sess': self.sess,
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'co_space': Box(low=-2, high=2, shape=(2,)),
            'ob_space': Box(low=-3, high=3, shape=(3,)),
            'all_ob_space': Box(low=-3, high=3, shape=(10,)),
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
                'a': Box(low=-1, high=1, shape=(1,)),
                'b': Box(low=-2, high=2, shape=(2,)),
            },
            'co_space': {
                'a': Box(low=-3, high=3, shape=(3,)),
                'b': Box(low=-4, high=4, shape=(4,)),
            },
            'ob_space': {
                'a': Box(low=-5, high=5, shape=(5,)),
                'b': Box(low=-6, high=6, shape=(6,)),
            },
            'all_ob_space': Box(low=-6, high=6, shape=(18,)),
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
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'co_space': Box(low=-2, high=2, shape=(2,)),
            'ob_space': Box(low=-3, high=3, shape=(3,)),
            'all_ob_space': Box(low=-3, high=3, shape=(10,)),
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
                'a': Box(low=-1, high=1, shape=(1,)),
                'b': Box(low=-2, high=2, shape=(2,)),
            },
            'co_space': {
                'a': Box(low=-3, high=3, shape=(3,)),
                'b': Box(low=-4, high=4, shape=(4,)),
            },
            'ob_space': {
                'a': Box(low=-5, high=5, shape=(5,)),
                'b': Box(low=-6, high=6, shape=(6,)),
            },
            'all_ob_space': Box(low=-6, high=6, shape=(18,)),
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
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'co_space': Box(low=-2, high=2, shape=(2,)),
            'ob_space': Box(low=-3, high=3, shape=(3,)),
            'all_ob_space': Box(low=-3, high=3, shape=(10,)),
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
                'a': Box(low=-1, high=1, shape=(1,)),
                'b': Box(low=-2, high=2, shape=(2,)),
            },
            'co_space': {
                'a': Box(low=-3, high=3, shape=(3,)),
                'b': Box(low=-4, high=4, shape=(4,)),
            },
            'ob_space': {
                'a': Box(low=-5, high=5, shape=(5,)),
                'b': Box(low=-6, high=6, shape=(6,)),
            },
            'all_ob_space': Box(low=-6, high=6, shape=(18,)),
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
