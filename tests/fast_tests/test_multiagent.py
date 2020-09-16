"""Tests for the policies in the hbaselines/multiagent subdirectory."""
import unittest
import numpy as np
import tensorflow as tf
from gym.spaces import Box

from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.multiagent.td3 import MultiFeedForwardPolicy as \
    TD3MultiFeedForwardPolicy
from hbaselines.multiagent.sac import MultiFeedForwardPolicy as \
    SACMultiFeedForwardPolicy
from hbaselines.multiagent.h_td3 import MultiGoalConditionedPolicy as \
    TD3MultiGoalConditionedPolicy
from hbaselines.multiagent.h_sac import MultiGoalConditionedPolicy as \
    SACMultiGoalConditionedPolicy
from hbaselines.algorithms.rl_algorithm import SAC_PARAMS
from hbaselines.algorithms.rl_algorithm import TD3_PARAMS
from hbaselines.algorithms.rl_algorithm import MULTIAGENT_PARAMS
from hbaselines.algorithms.rl_algorithm import GOAL_CONDITIONED_PARAMS


class TestMultiActorCriticPolicy(unittest.TestCase):
    """Test MultiActorCriticPolicy in hbaselines/multiagent/base.py."""

    def setUp(self):
        self.sess = tf.compat.v1.Session()

        # Shared policy parameters
        self.policy_params_shared = {
            'sess': self.sess,
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'co_space': Box(low=-2, high=2, shape=(2,)),
            'ob_space': Box(low=-3, high=3, shape=(3,)),
            'all_ob_space': Box(low=-3, high=3, shape=(10,)),
            'verbose': 0,
        }
        self.policy_params_shared.update(TD3_PARAMS.copy())
        self.policy_params_shared.update(MULTIAGENT_PARAMS.copy())
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
            'verbose': 0,
        }
        self.policy_params_independent.update(TD3_PARAMS.copy())
        self.policy_params_independent.update(MULTIAGENT_PARAMS.copy())
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
        """
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = False
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        for i in range(4):
            action_0 = np.array([i for _ in range(1)])
            action_1 = np.array([i for _ in range(2)])
            context0_0 = np.array([i for _ in range(3)])
            context0_1 = np.array([i for _ in range(4)])
            obs0_0 = np.array([i for _ in range(5)])
            obs0_1 = np.array([i for _ in range(6)])
            reward = i
            obs1_0 = np.array([i+1 for _ in range(5)])
            obs1_1 = np.array([i+1 for _ in range(6)])
            context1_0 = np.array([i for _ in range(3)])
            context1_1 = np.array([i for _ in range(4)])
            done = False
            is_final_step = False
            evaluate = False

            policy.store_transition(
                obs0={"a": obs0_0, "b": obs0_1},
                context0={"a": context0_0, "b": context0_1},
                action={"a": action_0, "b": action_1},
                reward={"a": reward, "b": reward},
                obs1={"a": obs1_0, "b": obs1_1},
                context1={"a": context1_0, "b": context1_1},
                done=done,
                is_final_step=is_final_step,
                evaluate=evaluate,
                env_num=0,
            )

        # =================================================================== #
        # test for agent a                                                    #
        # =================================================================== #

        obs_t = policy.agents["a"].replay_buffer.obs_t
        action_t = policy.agents["a"].replay_buffer.action_t
        reward = policy.agents["a"].replay_buffer.reward
        done = policy.agents["a"].replay_buffer.done

        # check the various attributes
        np.testing.assert_almost_equal(
            obs_t[:4, :],
            np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1.],
                      [2., 2., 2., 2., 2., 2., 2., 2.],
                      [3., 3., 3., 3., 3., 3., 3., 3.]])
        )

        np.testing.assert_almost_equal(
            action_t[:4, :],
            np.array([[0.], [1.], [2.], [3.]])
        )

        np.testing.assert_almost_equal(
            reward[:4],
            np.array([0., 1., 2., 3.])
        )

        np.testing.assert_almost_equal(
            done[:4],
            [0., 0., 0., 0.]
        )

        # =================================================================== #
        # test for agent b                                                    #
        # =================================================================== #

        obs_t = policy.agents["b"].replay_buffer.obs_t
        action_t = policy.agents["b"].replay_buffer.action_t
        reward = policy.agents["b"].replay_buffer.reward
        done = policy.agents["b"].replay_buffer.done

        # check the various attributes
        np.testing.assert_almost_equal(
            obs_t[:4, :],
            np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                      [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]])
        )

        np.testing.assert_almost_equal(
            action_t[:4, :],
            np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
        )

        np.testing.assert_almost_equal(
            reward[:4],
            np.array([0., 1., 2., 3.])
        )

        np.testing.assert_almost_equal(
            done[:4],
            [0., 0., 0., 0.]
        )

    def test_store_transition_2(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = False
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        for i in range(4):
            obs0 = np.array([i for _ in range(2)])
            context0 = np.array([i for _ in range(3)])
            action = np.array([i for _ in range(1)])
            reward = i
            obs1 = np.array([i+1 for _ in range(2)])
            context1 = np.array([i for _ in range(3)])
            is_final_step = False
            evaluate = False

            policy.store_transition(
                obs0={"a": obs0, "b": obs0 + 1},
                context0={"a": context0, "b": context0 + 1},
                action={"a": action, "b": action + 1},
                reward={"a": reward, "b": reward + 1},
                obs1={"a": obs1, "b": obs1 + 1},
                context1={"a": context1, "b": context1 + 1},
                done=0.,
                is_final_step=is_final_step,
                evaluate=evaluate,
                env_num=0,
            )

        # extract the attributes
        obs_t = policy.agents["policy"].replay_buffer.obs_t
        action_t = policy.agents["policy"].replay_buffer.action_t
        reward = policy.agents["policy"].replay_buffer.reward
        done = policy.agents["policy"].replay_buffer.done

        # check the various attributes
        np.testing.assert_almost_equal(
            obs_t[:8, :],
            np.array([[0., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1.],
                      [1., 1., 1., 1., 1.],
                      [2., 2., 2., 2., 2.],
                      [2., 2., 2., 2., 2.],
                      [3., 3., 3., 3., 3.],
                      [3., 3., 3., 3., 3.],
                      [4., 4., 4., 4., 4.]])
        )

        np.testing.assert_almost_equal(
            action_t[:8, :],
            np.array([[0.], [1.], [1.], [2.], [2.], [3.], [3.], [4.]])
        )

        np.testing.assert_almost_equal(
            reward[:8],
            np.array([0., 1., 1., 2., 2., 3., 3., 4.])
        )

        np.testing.assert_almost_equal(
            done[:8],
            [0., 0., 0., 0., 0., 0., 0., 0.]
        )


class TestTD3MultiFeedForwardPolicy(unittest.TestCase):
    """Test MultiFeedForwardPolicy in hbaselines/multiagent/td3.py."""

    def setUp(self):
        self.sess = tf.compat.v1.Session()

        # Shared policy parameters
        self.policy_params_shared = {
            'sess': self.sess,
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'co_space': Box(low=-2, high=2, shape=(2,)),
            'ob_space': Box(low=-3, high=3, shape=(3,)),
            'all_ob_space': Box(low=-3, high=3, shape=(10,)),
            'verbose': 0,
        }
        self.policy_params_shared.update(TD3_PARAMS.copy())
        self.policy_params_shared.update(MULTIAGENT_PARAMS.copy())
        self.policy_params_shared['shared'] = True
        self.policy_params_shared["model_params"]["model_type"] = "fcnet"

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
            'verbose': 0,
        }
        self.policy_params_independent.update(TD3_PARAMS.copy())
        self.policy_params_independent.update(MULTIAGENT_PARAMS.copy())
        self.policy_params_independent['shared'] = False
        self.policy_params_independent["model_params"]["model_type"] = "fcnet"

    def tearDown(self):
        self.sess.close()
        del self.policy_params_shared
        del self.policy_params_independent

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_deprecated(self):
        """Make sure that the original path still works (temporarily)."""
        raised = False
        try:
            from hbaselines.multi_fcnet.td3 import MultiFeedForwardPolicy
            policy_params = self.policy_params_independent.copy()
            _ = MultiFeedForwardPolicy(**policy_params)
        except ModuleNotFoundError:  # pragma: no cover
            raised = True  # pragma: no cover

        self.assertFalse(raised, 'Exception raised')

    def test_init_1(self):
        """Check the functionality of the __init__() method.

        This method is tested for the following features:

        1. The proper structure graph was generated.
        2. All input placeholders are correct.

        This is done for the following cases:

        1. maddpg = False, shared = False, model_type = "fcnet"
        2. maddpg = False, shared = True,  model_type = "fcnet"
        3. maddpg = True,  shared = False, model_type = "fcnet"
        4. maddpg = True,  shared = True,  model_type = "fcnet"
        5. maddpg = True,  shared = False, model_type = "conv"
        6. maddpg = True,  shared = True,  model_type = "conv"
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

    def test_init_5(self):
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = True
        policy_params["model_params"]["model_type"] = "conv"
        _ = TD3MultiFeedForwardPolicy(**policy_params)

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['a/model/centralized_qf_0/conv0/bias:0',
             'a/model/centralized_qf_0/conv0/kernel:0',
             'a/model/centralized_qf_0/conv1/bias:0',
             'a/model/centralized_qf_0/conv1/kernel:0',
             'a/model/centralized_qf_0/conv2/bias:0',
             'a/model/centralized_qf_0/conv2/kernel:0',
             'a/model/centralized_qf_0/fc0/bias:0',
             'a/model/centralized_qf_0/fc0/kernel:0',
             'a/model/centralized_qf_0/fc1/bias:0',
             'a/model/centralized_qf_0/fc1/kernel:0',
             'a/model/centralized_qf_0/qf_output/bias:0',
             'a/model/centralized_qf_0/qf_output/kernel:0',
             'a/model/centralized_qf_1/conv0/bias:0',
             'a/model/centralized_qf_1/conv0/kernel:0',
             'a/model/centralized_qf_1/conv1/bias:0',
             'a/model/centralized_qf_1/conv1/kernel:0',
             'a/model/centralized_qf_1/conv2/bias:0',
             'a/model/centralized_qf_1/conv2/kernel:0',
             'a/model/centralized_qf_1/fc0/bias:0',
             'a/model/centralized_qf_1/fc0/kernel:0',
             'a/model/centralized_qf_1/fc1/bias:0',
             'a/model/centralized_qf_1/fc1/kernel:0',
             'a/model/centralized_qf_1/qf_output/bias:0',
             'a/model/centralized_qf_1/qf_output/kernel:0',
             'a/model/pi/conv0/bias:0',
             'a/model/pi/conv0/kernel:0',
             'a/model/pi/conv1/bias:0',
             'a/model/pi/conv1/kernel:0',
             'a/model/pi/conv2/bias:0',
             'a/model/pi/conv2/kernel:0',
             'a/model/pi/fc0/bias:0',
             'a/model/pi/fc0/kernel:0',
             'a/model/pi/fc1/bias:0',
             'a/model/pi/fc1/kernel:0',
             'a/model/pi/output/bias:0',
             'a/model/pi/output/kernel:0',
             'a/target/centralized_qf_0/conv0/bias:0',
             'a/target/centralized_qf_0/conv0/kernel:0',
             'a/target/centralized_qf_0/conv1/bias:0',
             'a/target/centralized_qf_0/conv1/kernel:0',
             'a/target/centralized_qf_0/conv2/bias:0',
             'a/target/centralized_qf_0/conv2/kernel:0',
             'a/target/centralized_qf_0/fc0/bias:0',
             'a/target/centralized_qf_0/fc0/kernel:0',
             'a/target/centralized_qf_0/fc1/bias:0',
             'a/target/centralized_qf_0/fc1/kernel:0',
             'a/target/centralized_qf_0/qf_output/bias:0',
             'a/target/centralized_qf_0/qf_output/kernel:0',
             'a/target/centralized_qf_1/conv0/bias:0',
             'a/target/centralized_qf_1/conv0/kernel:0',
             'a/target/centralized_qf_1/conv1/bias:0',
             'a/target/centralized_qf_1/conv1/kernel:0',
             'a/target/centralized_qf_1/conv2/bias:0',
             'a/target/centralized_qf_1/conv2/kernel:0',
             'a/target/centralized_qf_1/fc0/bias:0',
             'a/target/centralized_qf_1/fc0/kernel:0',
             'a/target/centralized_qf_1/fc1/bias:0',
             'a/target/centralized_qf_1/fc1/kernel:0',
             'a/target/centralized_qf_1/qf_output/bias:0',
             'a/target/centralized_qf_1/qf_output/kernel:0',
             'a/target/pi/conv0/bias:0',
             'a/target/pi/conv0/kernel:0',
             'a/target/pi/conv1/bias:0',
             'a/target/pi/conv1/kernel:0',
             'a/target/pi/conv2/bias:0',
             'a/target/pi/conv2/kernel:0',
             'a/target/pi/fc0/bias:0',
             'a/target/pi/fc0/kernel:0',
             'a/target/pi/fc1/bias:0',
             'a/target/pi/fc1/kernel:0',
             'a/target/pi/output/bias:0',
             'a/target/pi/output/kernel:0',
             'b/model/centralized_qf_0/conv0/bias:0',
             'b/model/centralized_qf_0/conv0/kernel:0',
             'b/model/centralized_qf_0/conv1/bias:0',
             'b/model/centralized_qf_0/conv1/kernel:0',
             'b/model/centralized_qf_0/conv2/bias:0',
             'b/model/centralized_qf_0/conv2/kernel:0',
             'b/model/centralized_qf_0/fc0/bias:0',
             'b/model/centralized_qf_0/fc0/kernel:0',
             'b/model/centralized_qf_0/fc1/bias:0',
             'b/model/centralized_qf_0/fc1/kernel:0',
             'b/model/centralized_qf_0/qf_output/bias:0',
             'b/model/centralized_qf_0/qf_output/kernel:0',
             'b/model/centralized_qf_1/conv0/bias:0',
             'b/model/centralized_qf_1/conv0/kernel:0',
             'b/model/centralized_qf_1/conv1/bias:0',
             'b/model/centralized_qf_1/conv1/kernel:0',
             'b/model/centralized_qf_1/conv2/bias:0',
             'b/model/centralized_qf_1/conv2/kernel:0',
             'b/model/centralized_qf_1/fc0/bias:0',
             'b/model/centralized_qf_1/fc0/kernel:0',
             'b/model/centralized_qf_1/fc1/bias:0',
             'b/model/centralized_qf_1/fc1/kernel:0',
             'b/model/centralized_qf_1/qf_output/bias:0',
             'b/model/centralized_qf_1/qf_output/kernel:0',
             'b/model/pi/conv0/bias:0',
             'b/model/pi/conv0/kernel:0',
             'b/model/pi/conv1/bias:0',
             'b/model/pi/conv1/kernel:0',
             'b/model/pi/conv2/bias:0',
             'b/model/pi/conv2/kernel:0',
             'b/model/pi/fc0/bias:0',
             'b/model/pi/fc0/kernel:0',
             'b/model/pi/fc1/bias:0',
             'b/model/pi/fc1/kernel:0',
             'b/model/pi/output/bias:0',
             'b/model/pi/output/kernel:0',
             'b/target/centralized_qf_0/conv0/bias:0',
             'b/target/centralized_qf_0/conv0/kernel:0',
             'b/target/centralized_qf_0/conv1/bias:0',
             'b/target/centralized_qf_0/conv1/kernel:0',
             'b/target/centralized_qf_0/conv2/bias:0',
             'b/target/centralized_qf_0/conv2/kernel:0',
             'b/target/centralized_qf_0/fc0/bias:0',
             'b/target/centralized_qf_0/fc0/kernel:0',
             'b/target/centralized_qf_0/fc1/bias:0',
             'b/target/centralized_qf_0/fc1/kernel:0',
             'b/target/centralized_qf_0/qf_output/bias:0',
             'b/target/centralized_qf_0/qf_output/kernel:0',
             'b/target/centralized_qf_1/conv0/bias:0',
             'b/target/centralized_qf_1/conv0/kernel:0',
             'b/target/centralized_qf_1/conv1/bias:0',
             'b/target/centralized_qf_1/conv1/kernel:0',
             'b/target/centralized_qf_1/conv2/bias:0',
             'b/target/centralized_qf_1/conv2/kernel:0',
             'b/target/centralized_qf_1/fc0/bias:0',
             'b/target/centralized_qf_1/fc0/kernel:0',
             'b/target/centralized_qf_1/fc1/bias:0',
             'b/target/centralized_qf_1/fc1/kernel:0',
             'b/target/centralized_qf_1/qf_output/bias:0',
             'b/target/centralized_qf_1/qf_output/kernel:0',
             'b/target/pi/conv0/bias:0',
             'b/target/pi/conv0/kernel:0',
             'b/target/pi/conv1/bias:0',
             'b/target/pi/conv1/kernel:0',
             'b/target/pi/conv2/bias:0',
             'b/target/pi/conv2/kernel:0',
             'b/target/pi/fc0/bias:0',
             'b/target/pi/fc0/kernel:0',
             'b/target/pi/fc1/bias:0',
             'b/target/pi/fc1/kernel:0',
             'b/target/pi/output/bias:0',
             'b/target/pi/output/kernel:0']
        )

    def test_init_6(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = True
        policy_params["model_params"]["model_type"] = "conv"
        _ = TD3MultiFeedForwardPolicy(**policy_params)

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['model/centralized_qf_0/conv0/bias:0',
             'model/centralized_qf_0/conv0/kernel:0',
             'model/centralized_qf_0/conv1/bias:0',
             'model/centralized_qf_0/conv1/kernel:0',
             'model/centralized_qf_0/conv2/bias:0',
             'model/centralized_qf_0/conv2/kernel:0',
             'model/centralized_qf_0/fc0/bias:0',
             'model/centralized_qf_0/fc0/kernel:0',
             'model/centralized_qf_0/fc1/bias:0',
             'model/centralized_qf_0/fc1/kernel:0',
             'model/centralized_qf_0/qf_output/bias:0',
             'model/centralized_qf_0/qf_output/kernel:0',
             'model/centralized_qf_1/conv0/bias:0',
             'model/centralized_qf_1/conv0/kernel:0',
             'model/centralized_qf_1/conv1/bias:0',
             'model/centralized_qf_1/conv1/kernel:0',
             'model/centralized_qf_1/conv2/bias:0',
             'model/centralized_qf_1/conv2/kernel:0',
             'model/centralized_qf_1/fc0/bias:0',
             'model/centralized_qf_1/fc0/kernel:0',
             'model/centralized_qf_1/fc1/bias:0',
             'model/centralized_qf_1/fc1/kernel:0',
             'model/centralized_qf_1/qf_output/bias:0',
             'model/centralized_qf_1/qf_output/kernel:0',
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
             'target/centralized_qf_0/conv0/bias:0',
             'target/centralized_qf_0/conv0/kernel:0',
             'target/centralized_qf_0/conv1/bias:0',
             'target/centralized_qf_0/conv1/kernel:0',
             'target/centralized_qf_0/conv2/bias:0',
             'target/centralized_qf_0/conv2/kernel:0',
             'target/centralized_qf_0/fc0/bias:0',
             'target/centralized_qf_0/fc0/kernel:0',
             'target/centralized_qf_0/fc1/bias:0',
             'target/centralized_qf_0/fc1/kernel:0',
             'target/centralized_qf_0/qf_output/bias:0',
             'target/centralized_qf_0/qf_output/kernel:0',
             'target/centralized_qf_1/conv0/bias:0',
             'target/centralized_qf_1/conv0/kernel:0',
             'target/centralized_qf_1/conv1/bias:0',
             'target/centralized_qf_1/conv1/kernel:0',
             'target/centralized_qf_1/conv2/bias:0',
             'target/centralized_qf_1/conv2/kernel:0',
             'target/centralized_qf_1/fc0/bias:0',
             'target/centralized_qf_1/fc0/kernel:0',
             'target/centralized_qf_1/fc1/bias:0',
             'target/centralized_qf_1/fc1/kernel:0',
             'target/centralized_qf_1/qf_output/bias:0',
             'target/centralized_qf_1/qf_output/kernel:0',
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
             'target/pi/output/kernel:0']
        )

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

    def test_store_transition_1(self):
        """Check the functionality of the store_transition() method.

        This test checks for the following cases:

        1. maddpg = True,  shared = False
        2. maddpg = True,  shared = True
        """
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = True
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        for i in range(4):
            action_0 = np.array([i for _ in range(1)])
            action_1 = np.array([i for _ in range(2)])
            context0_0 = np.array([i for _ in range(3)])
            context0_1 = np.array([i for _ in range(4)])
            obs0_0 = np.array([i for _ in range(5)])
            obs0_1 = np.array([i for _ in range(6)])
            reward = i
            obs1_0 = np.array([i+1 for _ in range(5)])
            obs1_1 = np.array([i+1 for _ in range(6)])
            context1_0 = np.array([i for _ in range(3)])
            context1_1 = np.array([i for _ in range(4)])
            done = False
            is_final_step = False
            evaluate = False
            all_obs0 = np.array([i for _ in range(18)])
            all_obs1 = np.array([i+1 for _ in range(18)])

            policy.store_transition(
                obs0={"a": obs0_0, "b": obs0_1},
                context0={"a": context0_0, "b": context0_1},
                action={"a": action_0, "b": action_1},
                reward={"a": reward, "b": reward},
                obs1={"a": obs1_0, "b": obs1_1},
                context1={"a": context1_0, "b": context1_1},
                done=done,
                is_final_step=is_final_step,
                evaluate=evaluate,
                env_num=0,
                all_obs0=all_obs0,
                all_obs1=all_obs1,
            )

        # =================================================================== #
        # test for agent a                                                    #
        # =================================================================== #

        obs_t = policy.replay_buffer["a"].obs_t
        action_t = policy.replay_buffer["a"].action_t
        reward = policy.replay_buffer["a"].reward
        done = policy.replay_buffer["a"].done
        all_obs_t = policy.replay_buffer["a"].all_obs_t

        # check the various attributes
        np.testing.assert_almost_equal(
            obs_t[:4, :],
            np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1.],
                      [2., 2., 2., 2., 2., 2., 2., 2.],
                      [3., 3., 3., 3., 3., 3., 3., 3.]])
        )

        np.testing.assert_almost_equal(
            action_t[:4, :],
            np.array([[0.], [1.], [2.], [3.]])
        )

        np.testing.assert_almost_equal(
            reward[:4],
            np.array([0., 1., 2., 3.])
        )

        np.testing.assert_almost_equal(
            done[:4],
            [0., 0., 0., 0.]
        )

        np.testing.assert_almost_equal(
            all_obs_t[:4, :],
            np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                       1., 1., 1., 1.],
                      [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
                       2., 2., 2., 2.],
                      [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
                       3., 3., 3., 3.]])
        )

        # =================================================================== #
        # test for agent b                                                    #
        # =================================================================== #

        obs_t = policy.replay_buffer["b"].obs_t
        action_t = policy.replay_buffer["b"].action_t
        reward = policy.replay_buffer["b"].reward
        done = policy.replay_buffer["b"].done
        all_obs_t = policy.replay_buffer["b"].all_obs_t

        # check the various attributes
        np.testing.assert_almost_equal(
            obs_t[:4, :],
            np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                      [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]])
        )

        np.testing.assert_almost_equal(
            action_t[:4, :],
            np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
        )

        np.testing.assert_almost_equal(
            reward[:4],
            np.array([0., 1., 2., 3.])
        )

        np.testing.assert_almost_equal(
            done[:4],
            [0., 0., 0., 0.]
        )

        np.testing.assert_almost_equal(
            all_obs_t[:4, :],
            np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                       1., 1., 1., 1.],
                      [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
                       2., 2., 2., 2.],
                      [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
                       3., 3., 3., 3.]])
        )

    def test_store_transition_2(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = True
        policy_params["n_agents"] = 2
        policy = TD3MultiFeedForwardPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        for i in range(4):
            obs0 = np.array([i for _ in range(2)])
            context0 = np.array([i for _ in range(3)])
            action = np.array([i for _ in range(1)])
            reward = i
            obs1 = np.array([i+1 for _ in range(2)])
            context1 = np.array([i for _ in range(3)])
            is_final_step = False
            evaluate = False
            all_obs0 = np.array([i for _ in range(10)])
            all_obs1 = np.array([i+1 for _ in range(10)])

            policy.store_transition(
                obs0={"a": obs0, "b": obs0 + 1},
                context0={"a": context0, "b": context0 + 1},
                action={"a": action, "b": action + 1},
                reward={"a": reward, "b": reward + 1},
                obs1={"a": obs1, "b": obs1 + 1},
                context1={"a": context1, "b": context1 + 1},
                done=0.,
                is_final_step=is_final_step,
                evaluate=evaluate,
                env_num=0,
                all_obs0=all_obs0,
                all_obs1=all_obs1,
            )

        # extract the attributes
        obs_t = policy.replay_buffer.obs_t
        action_t = policy.replay_buffer.action
        reward = policy.replay_buffer.reward
        done = policy.replay_buffer.done
        all_obs_t = policy.replay_buffer.all_obs_t

        # check the various attributes
        np.testing.assert_almost_equal(
            obs_t[0][:4, :],
            np.array([[0., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1.],
                      [2., 2., 2., 2., 2.],
                      [3., 3., 3., 3., 3.]])
        )

        np.testing.assert_almost_equal(
            action_t[0][:4, :],
            np.array([[0.], [1.], [2.], [3.]])
        )

        np.testing.assert_almost_equal(
            reward[:4],
            np.array([0., 1., 2., 3.])
        )

        np.testing.assert_almost_equal(
            done[:4],
            [0., 0., 0., 0.]
        )

        np.testing.assert_almost_equal(
            all_obs_t[:4, :],
            np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                      [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]])
        )


class TestSACMultiFeedForwardPolicy(unittest.TestCase):
    """Test MultiFeedForwardPolicy in hbaselines/multiagent/sac.py."""

    def setUp(self):
        self.sess = tf.compat.v1.Session()

        # Shared policy parameters
        self.policy_params_shared = {
            'sess': self.sess,
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'co_space': Box(low=-2, high=2, shape=(2,)),
            'ob_space': Box(low=-3, high=3, shape=(3,)),
            'all_ob_space': Box(low=-3, high=3, shape=(10,)),
            'verbose': 0,
        }
        self.policy_params_shared.update(SAC_PARAMS.copy())
        self.policy_params_shared.update(MULTIAGENT_PARAMS.copy())
        self.policy_params_shared['shared'] = True
        self.policy_params_shared['model_params']['model_type'] = 'fcnet'

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
            'verbose': 0,
        }
        self.policy_params_independent.update(SAC_PARAMS.copy())
        self.policy_params_independent.update(MULTIAGENT_PARAMS.copy())
        self.policy_params_independent['shared'] = False
        self.policy_params_independent['model_params']['model_type'] = 'fcnet'

    def tearDown(self):
        self.sess.close()
        del self.policy_params_shared
        del self.policy_params_independent

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_deprecated(self):
        """Make sure that the original path still works (temporarily)."""
        raised = False
        try:
            from hbaselines.multi_fcnet.sac import MultiFeedForwardPolicy
            policy_params = self.policy_params_independent.copy()
            _ = MultiFeedForwardPolicy(**policy_params)
        except ModuleNotFoundError:  # pragma: no cover
            raised = True  # pragma: no cover

        self.assertFalse(raised, 'Exception raised')

    def test_init_1(self):
        """Check the functionality of the __init__() method.

        This method is tested for the following features:

        1. The proper structure graph was generated.
        2. All input placeholders are correct.

        This is done for the following cases:

        1. maddpg = False, shared = False, model_type = "fcnet"
        2. maddpg = False, shared = True,  model_type = "fcnet"
        3. maddpg = True,  shared = False, model_type = "fcnet"
        4. maddpg = True,  shared = True,  model_type = "fcnet"
        5. maddpg = True,  shared = False, model_type = "conv"
        6. maddpg = True,  shared = True,  model_type = "conv"
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

    def test_init_5(self):
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = True
        policy_params["model_params"]["model_type"] = "conv"
        _ = SACMultiFeedForwardPolicy(**policy_params)

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['a/model/centralized_value_fns/qf1/conv0/bias:0',
             'a/model/centralized_value_fns/qf1/conv0/kernel:0',
             'a/model/centralized_value_fns/qf1/conv1/bias:0',
             'a/model/centralized_value_fns/qf1/conv1/kernel:0',
             'a/model/centralized_value_fns/qf1/conv2/bias:0',
             'a/model/centralized_value_fns/qf1/conv2/kernel:0',
             'a/model/centralized_value_fns/qf1/fc0/bias:0',
             'a/model/centralized_value_fns/qf1/fc0/kernel:0',
             'a/model/centralized_value_fns/qf1/fc1/bias:0',
             'a/model/centralized_value_fns/qf1/fc1/kernel:0',
             'a/model/centralized_value_fns/qf1/qf_output/bias:0',
             'a/model/centralized_value_fns/qf1/qf_output/kernel:0',
             'a/model/centralized_value_fns/qf2/conv0/bias:0',
             'a/model/centralized_value_fns/qf2/conv0/kernel:0',
             'a/model/centralized_value_fns/qf2/conv1/bias:0',
             'a/model/centralized_value_fns/qf2/conv1/kernel:0',
             'a/model/centralized_value_fns/qf2/conv2/bias:0',
             'a/model/centralized_value_fns/qf2/conv2/kernel:0',
             'a/model/centralized_value_fns/qf2/fc0/bias:0',
             'a/model/centralized_value_fns/qf2/fc0/kernel:0',
             'a/model/centralized_value_fns/qf2/fc1/bias:0',
             'a/model/centralized_value_fns/qf2/fc1/kernel:0',
             'a/model/centralized_value_fns/qf2/qf_output/bias:0',
             'a/model/centralized_value_fns/qf2/qf_output/kernel:0',
             'a/model/centralized_value_fns/vf/conv0/bias:0',
             'a/model/centralized_value_fns/vf/conv0/kernel:0',
             'a/model/centralized_value_fns/vf/conv1/bias:0',
             'a/model/centralized_value_fns/vf/conv1/kernel:0',
             'a/model/centralized_value_fns/vf/conv2/bias:0',
             'a/model/centralized_value_fns/vf/conv2/kernel:0',
             'a/model/centralized_value_fns/vf/fc0/bias:0',
             'a/model/centralized_value_fns/vf/fc0/kernel:0',
             'a/model/centralized_value_fns/vf/fc1/bias:0',
             'a/model/centralized_value_fns/vf/fc1/kernel:0',
             'a/model/centralized_value_fns/vf/vf_output/bias:0',
             'a/model/centralized_value_fns/vf/vf_output/kernel:0',
             'a/model/log_alpha:0',
             'a/model/pi/conv0/bias:0',
             'a/model/pi/conv0/kernel:0',
             'a/model/pi/conv1/bias:0',
             'a/model/pi/conv1/kernel:0',
             'a/model/pi/conv2/bias:0',
             'a/model/pi/conv2/kernel:0',
             'a/model/pi/fc0/bias:0',
             'a/model/pi/fc0/kernel:0',
             'a/model/pi/fc1/bias:0',
             'a/model/pi/fc1/kernel:0',
             'a/model/pi/log_std/bias:0',
             'a/model/pi/log_std/kernel:0',
             'a/model/pi/mean/bias:0',
             'a/model/pi/mean/kernel:0',
             'a/target/centralized_value_fns/vf/conv0/bias:0',
             'a/target/centralized_value_fns/vf/conv0/kernel:0',
             'a/target/centralized_value_fns/vf/conv1/bias:0',
             'a/target/centralized_value_fns/vf/conv1/kernel:0',
             'a/target/centralized_value_fns/vf/conv2/bias:0',
             'a/target/centralized_value_fns/vf/conv2/kernel:0',
             'a/target/centralized_value_fns/vf/fc0/bias:0',
             'a/target/centralized_value_fns/vf/fc0/kernel:0',
             'a/target/centralized_value_fns/vf/fc1/bias:0',
             'a/target/centralized_value_fns/vf/fc1/kernel:0',
             'a/target/centralized_value_fns/vf/vf_output/bias:0',
             'a/target/centralized_value_fns/vf/vf_output/kernel:0',
             'b/model/centralized_value_fns/qf1/conv0/bias:0',
             'b/model/centralized_value_fns/qf1/conv0/kernel:0',
             'b/model/centralized_value_fns/qf1/conv1/bias:0',
             'b/model/centralized_value_fns/qf1/conv1/kernel:0',
             'b/model/centralized_value_fns/qf1/conv2/bias:0',
             'b/model/centralized_value_fns/qf1/conv2/kernel:0',
             'b/model/centralized_value_fns/qf1/fc0/bias:0',
             'b/model/centralized_value_fns/qf1/fc0/kernel:0',
             'b/model/centralized_value_fns/qf1/fc1/bias:0',
             'b/model/centralized_value_fns/qf1/fc1/kernel:0',
             'b/model/centralized_value_fns/qf1/qf_output/bias:0',
             'b/model/centralized_value_fns/qf1/qf_output/kernel:0',
             'b/model/centralized_value_fns/qf2/conv0/bias:0',
             'b/model/centralized_value_fns/qf2/conv0/kernel:0',
             'b/model/centralized_value_fns/qf2/conv1/bias:0',
             'b/model/centralized_value_fns/qf2/conv1/kernel:0',
             'b/model/centralized_value_fns/qf2/conv2/bias:0',
             'b/model/centralized_value_fns/qf2/conv2/kernel:0',
             'b/model/centralized_value_fns/qf2/fc0/bias:0',
             'b/model/centralized_value_fns/qf2/fc0/kernel:0',
             'b/model/centralized_value_fns/qf2/fc1/bias:0',
             'b/model/centralized_value_fns/qf2/fc1/kernel:0',
             'b/model/centralized_value_fns/qf2/qf_output/bias:0',
             'b/model/centralized_value_fns/qf2/qf_output/kernel:0',
             'b/model/centralized_value_fns/vf/conv0/bias:0',
             'b/model/centralized_value_fns/vf/conv0/kernel:0',
             'b/model/centralized_value_fns/vf/conv1/bias:0',
             'b/model/centralized_value_fns/vf/conv1/kernel:0',
             'b/model/centralized_value_fns/vf/conv2/bias:0',
             'b/model/centralized_value_fns/vf/conv2/kernel:0',
             'b/model/centralized_value_fns/vf/fc0/bias:0',
             'b/model/centralized_value_fns/vf/fc0/kernel:0',
             'b/model/centralized_value_fns/vf/fc1/bias:0',
             'b/model/centralized_value_fns/vf/fc1/kernel:0',
             'b/model/centralized_value_fns/vf/vf_output/bias:0',
             'b/model/centralized_value_fns/vf/vf_output/kernel:0',
             'b/model/log_alpha:0',
             'b/model/pi/conv0/bias:0',
             'b/model/pi/conv0/kernel:0',
             'b/model/pi/conv1/bias:0',
             'b/model/pi/conv1/kernel:0',
             'b/model/pi/conv2/bias:0',
             'b/model/pi/conv2/kernel:0',
             'b/model/pi/fc0/bias:0',
             'b/model/pi/fc0/kernel:0',
             'b/model/pi/fc1/bias:0',
             'b/model/pi/fc1/kernel:0',
             'b/model/pi/log_std/bias:0',
             'b/model/pi/log_std/kernel:0',
             'b/model/pi/mean/bias:0',
             'b/model/pi/mean/kernel:0',
             'b/target/centralized_value_fns/vf/conv0/bias:0',
             'b/target/centralized_value_fns/vf/conv0/kernel:0',
             'b/target/centralized_value_fns/vf/conv1/bias:0',
             'b/target/centralized_value_fns/vf/conv1/kernel:0',
             'b/target/centralized_value_fns/vf/conv2/bias:0',
             'b/target/centralized_value_fns/vf/conv2/kernel:0',
             'b/target/centralized_value_fns/vf/fc0/bias:0',
             'b/target/centralized_value_fns/vf/fc0/kernel:0',
             'b/target/centralized_value_fns/vf/fc1/bias:0',
             'b/target/centralized_value_fns/vf/fc1/kernel:0',
             'b/target/centralized_value_fns/vf/vf_output/bias:0',
             'b/target/centralized_value_fns/vf/vf_output/kernel:0']
        )

    def test_init_6(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = True
        policy_params["model_params"]["model_type"] = "conv"
        _ = SACMultiFeedForwardPolicy(**policy_params)

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['model/centralized_value_fns/qf1/conv0/bias:0',
             'model/centralized_value_fns/qf1/conv0/kernel:0',
             'model/centralized_value_fns/qf1/conv1/bias:0',
             'model/centralized_value_fns/qf1/conv1/kernel:0',
             'model/centralized_value_fns/qf1/conv2/bias:0',
             'model/centralized_value_fns/qf1/conv2/kernel:0',
             'model/centralized_value_fns/qf1/fc0/bias:0',
             'model/centralized_value_fns/qf1/fc0/kernel:0',
             'model/centralized_value_fns/qf1/fc1/bias:0',
             'model/centralized_value_fns/qf1/fc1/kernel:0',
             'model/centralized_value_fns/qf1/qf_output/bias:0',
             'model/centralized_value_fns/qf1/qf_output/kernel:0',
             'model/centralized_value_fns/qf2/conv0/bias:0',
             'model/centralized_value_fns/qf2/conv0/kernel:0',
             'model/centralized_value_fns/qf2/conv1/bias:0',
             'model/centralized_value_fns/qf2/conv1/kernel:0',
             'model/centralized_value_fns/qf2/conv2/bias:0',
             'model/centralized_value_fns/qf2/conv2/kernel:0',
             'model/centralized_value_fns/qf2/fc0/bias:0',
             'model/centralized_value_fns/qf2/fc0/kernel:0',
             'model/centralized_value_fns/qf2/fc1/bias:0',
             'model/centralized_value_fns/qf2/fc1/kernel:0',
             'model/centralized_value_fns/qf2/qf_output/bias:0',
             'model/centralized_value_fns/qf2/qf_output/kernel:0',
             'model/centralized_value_fns/vf/conv0/bias:0',
             'model/centralized_value_fns/vf/conv0/kernel:0',
             'model/centralized_value_fns/vf/conv1/bias:0',
             'model/centralized_value_fns/vf/conv1/kernel:0',
             'model/centralized_value_fns/vf/conv2/bias:0',
             'model/centralized_value_fns/vf/conv2/kernel:0',
             'model/centralized_value_fns/vf/fc0/bias:0',
             'model/centralized_value_fns/vf/fc0/kernel:0',
             'model/centralized_value_fns/vf/fc1/bias:0',
             'model/centralized_value_fns/vf/fc1/kernel:0',
             'model/centralized_value_fns/vf/vf_output/bias:0',
             'model/centralized_value_fns/vf/vf_output/kernel:0',
             'model/log_alpha:0',
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
             'target/centralized_value_fns/vf/conv0/bias:0',
             'target/centralized_value_fns/vf/conv0/kernel:0',
             'target/centralized_value_fns/vf/conv1/bias:0',
             'target/centralized_value_fns/vf/conv1/kernel:0',
             'target/centralized_value_fns/vf/conv2/bias:0',
             'target/centralized_value_fns/vf/conv2/kernel:0',
             'target/centralized_value_fns/vf/fc0/bias:0',
             'target/centralized_value_fns/vf/fc0/kernel:0',
             'target/centralized_value_fns/vf/fc1/bias:0',
             'target/centralized_value_fns/vf/fc1/kernel:0',
             'target/centralized_value_fns/vf/vf_output/bias:0',
             'target/centralized_value_fns/vf/vf_output/kernel:0']
        )

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

    def test_store_transition_1(self):
        """Check the functionality of the store_transition() method.

        This test checks for the following cases:

        1. maddpg = True,  shared = False
        2. maddpg = True,  shared = True
        """
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = True
        policy = SACMultiFeedForwardPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        for i in range(4):
            action_0 = np.array([i for _ in range(1)])
            action_1 = np.array([i for _ in range(2)])
            context0_0 = np.array([i for _ in range(3)])
            context0_1 = np.array([i for _ in range(4)])
            obs0_0 = np.array([i for _ in range(5)])
            obs0_1 = np.array([i for _ in range(6)])
            reward = i
            obs1_0 = np.array([i+1 for _ in range(5)])
            obs1_1 = np.array([i+1 for _ in range(6)])
            context1_0 = np.array([i for _ in range(3)])
            context1_1 = np.array([i for _ in range(4)])
            done = False
            is_final_step = False
            evaluate = False
            all_obs0 = np.array([i for _ in range(18)])
            all_obs1 = np.array([i+1 for _ in range(18)])

            policy.store_transition(
                obs0={"a": obs0_0, "b": obs0_1},
                context0={"a": context0_0, "b": context0_1},
                action={"a": action_0, "b": action_1},
                reward={"a": reward, "b": reward},
                obs1={"a": obs1_0, "b": obs1_1},
                context1={"a": context1_0, "b": context1_1},
                done=done,
                is_final_step=is_final_step,
                evaluate=evaluate,
                env_num=0,
                all_obs0=all_obs0,
                all_obs1=all_obs1,
            )

        # =================================================================== #
        # test for agent a                                                    #
        # =================================================================== #

        obs_t = policy.replay_buffer["a"].obs_t
        action_t = policy.replay_buffer["a"].action_t
        reward = policy.replay_buffer["a"].reward
        done = policy.replay_buffer["a"].done
        all_obs_t = policy.replay_buffer["a"].all_obs_t

        # check the various attributes
        np.testing.assert_almost_equal(
            obs_t[:4, :],
            np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1.],
                      [2., 2., 2., 2., 2., 2., 2., 2.],
                      [3., 3., 3., 3., 3., 3., 3., 3.]])
        )

        np.testing.assert_almost_equal(
            action_t[:4, :],
            np.array([[0.], [1.], [2.], [3.]])
        )

        np.testing.assert_almost_equal(
            reward[:4],
            np.array([0., 1., 2., 3.])
        )

        np.testing.assert_almost_equal(
            done[:4],
            [0., 0., 0., 0.]
        )

        np.testing.assert_almost_equal(
            all_obs_t[:4, :],
            np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                       1., 1., 1., 1.],
                      [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
                       2., 2., 2., 2.],
                      [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
                       3., 3., 3., 3.]])
        )

        # =================================================================== #
        # test for agent b                                                    #
        # =================================================================== #

        obs_t = policy.replay_buffer["b"].obs_t
        action_t = policy.replay_buffer["b"].action_t
        reward = policy.replay_buffer["b"].reward
        done = policy.replay_buffer["b"].done
        all_obs_t = policy.replay_buffer["b"].all_obs_t

        # check the various attributes
        np.testing.assert_almost_equal(
            obs_t[:4, :],
            np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                      [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]])
        )

        np.testing.assert_almost_equal(
            action_t[:4, :],
            np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
        )

        np.testing.assert_almost_equal(
            reward[:4],
            np.array([0., 1., 2., 3.])
        )

        np.testing.assert_almost_equal(
            done[:4],
            [0., 0., 0., 0.]
        )

        np.testing.assert_almost_equal(
            all_obs_t[:4, :],
            np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                       1., 1., 1., 1.],
                      [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
                       2., 2., 2., 2.],
                      [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
                       3., 3., 3., 3.]])
        )

    def test_store_transition_4(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = True
        policy_params["n_agents"] = 2
        policy = SACMultiFeedForwardPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        for i in range(4):
            obs0 = np.array([i for _ in range(2)])
            context0 = np.array([i for _ in range(3)])
            action = np.array([i for _ in range(1)])
            reward = i
            obs1 = np.array([i+1 for _ in range(2)])
            context1 = np.array([i for _ in range(3)])
            is_final_step = False
            evaluate = False
            all_obs0 = np.array([i for _ in range(10)])
            all_obs1 = np.array([i+1 for _ in range(10)])

            policy.store_transition(
                obs0={"a": obs0, "b": obs0 + 1},
                context0={"a": context0, "b": context0 + 1},
                action={"a": action, "b": action + 1},
                reward={"a": reward, "b": reward + 1},
                obs1={"a": obs1, "b": obs1 + 1},
                context1={"a": context1, "b": context1 + 1},
                done=0.,
                is_final_step=is_final_step,
                evaluate=evaluate,
                env_num=0,
                all_obs0=all_obs0,
                all_obs1=all_obs1,
            )

        # extract the attributes
        obs_t = policy.replay_buffer.obs_t
        action_t = policy.replay_buffer.action
        reward = policy.replay_buffer.reward
        done = policy.replay_buffer.done
        all_obs_t = policy.replay_buffer.all_obs_t

        # check the various attributes
        np.testing.assert_almost_equal(
            obs_t[0][:4, :],
            np.array([[0., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1.],
                      [2., 2., 2., 2., 2.],
                      [3., 3., 3., 3., 3.]])
        )

        np.testing.assert_almost_equal(
            action_t[0][:4, :],
            np.array([[0.], [1.], [2.], [3.]])
        )

        np.testing.assert_almost_equal(
            reward[:4],
            np.array([0., 1., 2., 3.])
        )

        np.testing.assert_almost_equal(
            done[:4],
            [0., 0., 0., 0.]
        )

        np.testing.assert_almost_equal(
            all_obs_t[:4, :],
            np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                      [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]])
        )


class TestTD3MultiGoalConditionedPolicy(unittest.TestCase):
    """Test MultiFeedForwardPolicy in hbaselines/multiagent/h_td3.py."""

    def setUp(self):
        self.sess = tf.compat.v1.Session()

        # Shared policy parameters
        self.policy_params_shared = {
            'sess': self.sess,
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'co_space': Box(low=-2, high=2, shape=(2,)),
            'ob_space': Box(low=-3, high=3, shape=(3,)),
            'all_ob_space': Box(low=-3, high=3, shape=(10,)),
            'verbose': 0,
        }
        self.policy_params_shared.update(TD3_PARAMS.copy())
        self.policy_params_shared.update(GOAL_CONDITIONED_PARAMS.copy())
        self.policy_params_shared.update(MULTIAGENT_PARAMS.copy())
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
            'verbose': 0,
        }
        self.policy_params_independent.update(TD3_PARAMS.copy())
        self.policy_params_independent.update(GOAL_CONDITIONED_PARAMS.copy())
        self.policy_params_independent.update(MULTIAGENT_PARAMS.copy())
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
        """
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = False
        policy = TD3MultiGoalConditionedPolicy(**policy_params)

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['a/level_0/model/pi/fc0/bias:0',
             'a/level_0/model/pi/fc0/kernel:0',
             'a/level_0/model/pi/fc1/bias:0',
             'a/level_0/model/pi/fc1/kernel:0',
             'a/level_0/model/pi/output/bias:0',
             'a/level_0/model/pi/output/kernel:0',
             'a/level_0/model/qf_0/fc0/bias:0',
             'a/level_0/model/qf_0/fc0/kernel:0',
             'a/level_0/model/qf_0/fc1/bias:0',
             'a/level_0/model/qf_0/fc1/kernel:0',
             'a/level_0/model/qf_0/qf_output/bias:0',
             'a/level_0/model/qf_0/qf_output/kernel:0',
             'a/level_0/model/qf_1/fc0/bias:0',
             'a/level_0/model/qf_1/fc0/kernel:0',
             'a/level_0/model/qf_1/fc1/bias:0',
             'a/level_0/model/qf_1/fc1/kernel:0',
             'a/level_0/model/qf_1/qf_output/bias:0',
             'a/level_0/model/qf_1/qf_output/kernel:0',
             'a/level_0/target/pi/fc0/bias:0',
             'a/level_0/target/pi/fc0/kernel:0',
             'a/level_0/target/pi/fc1/bias:0',
             'a/level_0/target/pi/fc1/kernel:0',
             'a/level_0/target/pi/output/bias:0',
             'a/level_0/target/pi/output/kernel:0',
             'a/level_0/target/qf_0/fc0/bias:0',
             'a/level_0/target/qf_0/fc0/kernel:0',
             'a/level_0/target/qf_0/fc1/bias:0',
             'a/level_0/target/qf_0/fc1/kernel:0',
             'a/level_0/target/qf_0/qf_output/bias:0',
             'a/level_0/target/qf_0/qf_output/kernel:0',
             'a/level_0/target/qf_1/fc0/bias:0',
             'a/level_0/target/qf_1/fc0/kernel:0',
             'a/level_0/target/qf_1/fc1/bias:0',
             'a/level_0/target/qf_1/fc1/kernel:0',
             'a/level_0/target/qf_1/qf_output/bias:0',
             'a/level_0/target/qf_1/qf_output/kernel:0',
             'a/level_1/model/pi/fc0/bias:0',
             'a/level_1/model/pi/fc0/kernel:0',
             'a/level_1/model/pi/fc1/bias:0',
             'a/level_1/model/pi/fc1/kernel:0',
             'a/level_1/model/pi/output/bias:0',
             'a/level_1/model/pi/output/kernel:0',
             'a/level_1/model/qf_0/fc0/bias:0',
             'a/level_1/model/qf_0/fc0/kernel:0',
             'a/level_1/model/qf_0/fc1/bias:0',
             'a/level_1/model/qf_0/fc1/kernel:0',
             'a/level_1/model/qf_0/qf_output/bias:0',
             'a/level_1/model/qf_0/qf_output/kernel:0',
             'a/level_1/model/qf_1/fc0/bias:0',
             'a/level_1/model/qf_1/fc0/kernel:0',
             'a/level_1/model/qf_1/fc1/bias:0',
             'a/level_1/model/qf_1/fc1/kernel:0',
             'a/level_1/model/qf_1/qf_output/bias:0',
             'a/level_1/model/qf_1/qf_output/kernel:0',
             'a/level_1/target/pi/fc0/bias:0',
             'a/level_1/target/pi/fc0/kernel:0',
             'a/level_1/target/pi/fc1/bias:0',
             'a/level_1/target/pi/fc1/kernel:0',
             'a/level_1/target/pi/output/bias:0',
             'a/level_1/target/pi/output/kernel:0',
             'a/level_1/target/qf_0/fc0/bias:0',
             'a/level_1/target/qf_0/fc0/kernel:0',
             'a/level_1/target/qf_0/fc1/bias:0',
             'a/level_1/target/qf_0/fc1/kernel:0',
             'a/level_1/target/qf_0/qf_output/bias:0',
             'a/level_1/target/qf_0/qf_output/kernel:0',
             'a/level_1/target/qf_1/fc0/bias:0',
             'a/level_1/target/qf_1/fc0/kernel:0',
             'a/level_1/target/qf_1/fc1/bias:0',
             'a/level_1/target/qf_1/fc1/kernel:0',
             'a/level_1/target/qf_1/qf_output/bias:0',
             'a/level_1/target/qf_1/qf_output/kernel:0',
             'b/level_0/model/pi/fc0/bias:0',
             'b/level_0/model/pi/fc0/kernel:0',
             'b/level_0/model/pi/fc1/bias:0',
             'b/level_0/model/pi/fc1/kernel:0',
             'b/level_0/model/pi/output/bias:0',
             'b/level_0/model/pi/output/kernel:0',
             'b/level_0/model/qf_0/fc0/bias:0',
             'b/level_0/model/qf_0/fc0/kernel:0',
             'b/level_0/model/qf_0/fc1/bias:0',
             'b/level_0/model/qf_0/fc1/kernel:0',
             'b/level_0/model/qf_0/qf_output/bias:0',
             'b/level_0/model/qf_0/qf_output/kernel:0',
             'b/level_0/model/qf_1/fc0/bias:0',
             'b/level_0/model/qf_1/fc0/kernel:0',
             'b/level_0/model/qf_1/fc1/bias:0',
             'b/level_0/model/qf_1/fc1/kernel:0',
             'b/level_0/model/qf_1/qf_output/bias:0',
             'b/level_0/model/qf_1/qf_output/kernel:0',
             'b/level_0/target/pi/fc0/bias:0',
             'b/level_0/target/pi/fc0/kernel:0',
             'b/level_0/target/pi/fc1/bias:0',
             'b/level_0/target/pi/fc1/kernel:0',
             'b/level_0/target/pi/output/bias:0',
             'b/level_0/target/pi/output/kernel:0',
             'b/level_0/target/qf_0/fc0/bias:0',
             'b/level_0/target/qf_0/fc0/kernel:0',
             'b/level_0/target/qf_0/fc1/bias:0',
             'b/level_0/target/qf_0/fc1/kernel:0',
             'b/level_0/target/qf_0/qf_output/bias:0',
             'b/level_0/target/qf_0/qf_output/kernel:0',
             'b/level_0/target/qf_1/fc0/bias:0',
             'b/level_0/target/qf_1/fc0/kernel:0',
             'b/level_0/target/qf_1/fc1/bias:0',
             'b/level_0/target/qf_1/fc1/kernel:0',
             'b/level_0/target/qf_1/qf_output/bias:0',
             'b/level_0/target/qf_1/qf_output/kernel:0',
             'b/level_1/model/pi/fc0/bias:0',
             'b/level_1/model/pi/fc0/kernel:0',
             'b/level_1/model/pi/fc1/bias:0',
             'b/level_1/model/pi/fc1/kernel:0',
             'b/level_1/model/pi/output/bias:0',
             'b/level_1/model/pi/output/kernel:0',
             'b/level_1/model/qf_0/fc0/bias:0',
             'b/level_1/model/qf_0/fc0/kernel:0',
             'b/level_1/model/qf_0/fc1/bias:0',
             'b/level_1/model/qf_0/fc1/kernel:0',
             'b/level_1/model/qf_0/qf_output/bias:0',
             'b/level_1/model/qf_0/qf_output/kernel:0',
             'b/level_1/model/qf_1/fc0/bias:0',
             'b/level_1/model/qf_1/fc0/kernel:0',
             'b/level_1/model/qf_1/fc1/bias:0',
             'b/level_1/model/qf_1/fc1/kernel:0',
             'b/level_1/model/qf_1/qf_output/bias:0',
             'b/level_1/model/qf_1/qf_output/kernel:0',
             'b/level_1/target/pi/fc0/bias:0',
             'b/level_1/target/pi/fc0/kernel:0',
             'b/level_1/target/pi/fc1/bias:0',
             'b/level_1/target/pi/fc1/kernel:0',
             'b/level_1/target/pi/output/bias:0',
             'b/level_1/target/pi/output/kernel:0',
             'b/level_1/target/qf_0/fc0/bias:0',
             'b/level_1/target/qf_0/fc0/kernel:0',
             'b/level_1/target/qf_0/fc1/bias:0',
             'b/level_1/target/qf_0/fc1/kernel:0',
             'b/level_1/target/qf_0/qf_output/bias:0',
             'b/level_1/target/qf_0/qf_output/kernel:0',
             'b/level_1/target/qf_1/fc0/bias:0',
             'b/level_1/target/qf_1/fc0/kernel:0',
             'b/level_1/target/qf_1/fc1/bias:0',
             'b/level_1/target/qf_1/fc1/kernel:0',
             'b/level_1/target/qf_1/qf_output/bias:0',
             'b/level_1/target/qf_1/qf_output/kernel:0']
        )

        # Check observation/action/context spaces of the agents
        self.assertEqual(
            policy.agents['a'].policy[0].ac_space.shape[0],
            self.policy_params_independent['ob_space']['a'].shape[0]
        )
        self.assertEqual(
            policy.agents['a'].policy[0].ob_space.shape[0],
            self.policy_params_independent['ob_space']['a'].shape[0]
        )
        self.assertEqual(
            policy.agents['a'].policy[0].co_space.shape[0],
            self.policy_params_independent['co_space']['a'].shape[0]
        )

        self.assertEqual(
            policy.agents['a'].policy[1].ac_space.shape[0],
            self.policy_params_independent['ac_space']['a'].shape[0]
        )
        self.assertEqual(
            policy.agents['a'].policy[1].ob_space.shape[0],
            self.policy_params_independent['ob_space']['a'].shape[0]
        )
        self.assertEqual(
            policy.agents['a'].policy[1].co_space.shape[0],
            self.policy_params_independent['ob_space']['a'].shape[0]
        )

        self.assertEqual(
            policy.agents['b'].policy[0].ac_space.shape[0],
            self.policy_params_independent['ob_space']['b'].shape[0]
        )
        self.assertEqual(
            policy.agents['b'].policy[0].ob_space.shape[0],
            self.policy_params_independent['ob_space']['b'].shape[0]
        )
        self.assertEqual(
            policy.agents['b'].policy[0].co_space.shape[0],
            self.policy_params_independent['co_space']['b'].shape[0]
        )

        self.assertEqual(
            policy.agents['b'].policy[1].ac_space.shape[0],
            self.policy_params_independent['ac_space']['b'].shape[0]
        )
        self.assertEqual(
            policy.agents['b'].policy[1].ob_space.shape[0],
            self.policy_params_independent['ob_space']['b'].shape[0]
        )
        self.assertEqual(
            policy.agents['b'].policy[1].co_space.shape[0],
            self.policy_params_independent['ob_space']['b'].shape[0]
        )

        # Check the instantiation of the class attributes.
        self.assertTrue(not policy.shared)
        self.assertTrue(not policy.maddpg)

    def test_init_2(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = False
        policy = TD3MultiGoalConditionedPolicy(**policy_params)

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

        # Check observation/action/context spaces of the agents
        self.assertEqual(
            policy.agents['policy'].policy[0].ac_space.shape[0],
            self.policy_params_shared['ob_space'].shape[0]
        )
        self.assertEqual(
            policy.agents['policy'].policy[0].ob_space.shape[0],
            self.policy_params_shared['ob_space'].shape[0]
        )
        self.assertEqual(
            policy.agents['policy'].policy[0].co_space.shape[0],
            self.policy_params_shared['co_space'].shape[0]
        )

        self.assertEqual(
            policy.agents['policy'].policy[1].ac_space.shape[0],
            self.policy_params_shared['ac_space'].shape[0]
        )
        self.assertEqual(
            policy.agents['policy'].policy[1].ob_space.shape[0],
            self.policy_params_shared['ob_space'].shape[0]
        )
        self.assertEqual(
            policy.agents['policy'].policy[1].co_space.shape[0],
            self.policy_params_shared['ob_space'].shape[0]
        )

        # Check the instantiation of the class attributes.
        self.assertTrue(policy.shared)
        self.assertTrue(not policy.maddpg)

    def test_initialize_1(self):
        """Check the functionality of the initialize() method.

        This test validates that the target variables are properly initialized
        when initialize is called.

        This is done for the following cases:

        1. maddpg = False, shared = False
        2. maddpg = False, shared = True
        """
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = False
        policy = TD3MultiGoalConditionedPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        model_var_list = [
            'a/level_0/model/pi/fc0/bias:0',
            'a/level_0/model/pi/fc0/kernel:0',
            'a/level_0/model/pi/fc1/bias:0',
            'a/level_0/model/pi/fc1/kernel:0',
            'a/level_0/model/pi/output/bias:0',
            'a/level_0/model/pi/output/kernel:0',
            'a/level_0/model/qf_0/fc0/bias:0',
            'a/level_0/model/qf_0/fc0/kernel:0',
            'a/level_0/model/qf_0/fc1/bias:0',
            'a/level_0/model/qf_0/fc1/kernel:0',
            'a/level_0/model/qf_0/qf_output/bias:0',
            'a/level_0/model/qf_0/qf_output/kernel:0',
            'a/level_0/model/qf_1/fc0/bias:0',
            'a/level_0/model/qf_1/fc0/kernel:0',
            'a/level_0/model/qf_1/fc1/bias:0',
            'a/level_0/model/qf_1/fc1/kernel:0',
            'a/level_0/model/qf_1/qf_output/bias:0',
            'a/level_0/model/qf_1/qf_output/kernel:0',

            'a/level_1/model/pi/fc0/bias:0',
            'a/level_1/model/pi/fc0/kernel:0',
            'a/level_1/model/pi/fc1/bias:0',
            'a/level_1/model/pi/fc1/kernel:0',
            'a/level_1/model/pi/output/bias:0',
            'a/level_1/model/pi/output/kernel:0',
            'a/level_1/model/qf_0/fc0/bias:0',
            'a/level_1/model/qf_0/fc0/kernel:0',
            'a/level_1/model/qf_0/fc1/bias:0',
            'a/level_1/model/qf_0/fc1/kernel:0',
            'a/level_1/model/qf_0/qf_output/bias:0',
            'a/level_1/model/qf_0/qf_output/kernel:0',
            'a/level_1/model/qf_1/fc0/bias:0',
            'a/level_1/model/qf_1/fc0/kernel:0',
            'a/level_1/model/qf_1/fc1/bias:0',
            'a/level_1/model/qf_1/fc1/kernel:0',
            'a/level_1/model/qf_1/qf_output/bias:0',
            'a/level_1/model/qf_1/qf_output/kernel:0',

            'b/level_0/model/pi/fc0/bias:0',
            'b/level_0/model/pi/fc0/kernel:0',
            'b/level_0/model/pi/fc1/bias:0',
            'b/level_0/model/pi/fc1/kernel:0',
            'b/level_0/model/pi/output/bias:0',
            'b/level_0/model/pi/output/kernel:0',
            'b/level_0/model/qf_0/fc0/bias:0',
            'b/level_0/model/qf_0/fc0/kernel:0',
            'b/level_0/model/qf_0/fc1/bias:0',
            'b/level_0/model/qf_0/fc1/kernel:0',
            'b/level_0/model/qf_0/qf_output/bias:0',
            'b/level_0/model/qf_0/qf_output/kernel:0',
            'b/level_0/model/qf_1/fc0/bias:0',
            'b/level_0/model/qf_1/fc0/kernel:0',
            'b/level_0/model/qf_1/fc1/bias:0',
            'b/level_0/model/qf_1/fc1/kernel:0',
            'b/level_0/model/qf_1/qf_output/bias:0',
            'b/level_0/model/qf_1/qf_output/kernel:0',

            'b/level_1/model/pi/fc0/bias:0',
            'b/level_1/model/pi/fc0/kernel:0',
            'b/level_1/model/pi/fc1/bias:0',
            'b/level_1/model/pi/fc1/kernel:0',
            'b/level_1/model/pi/output/bias:0',
            'b/level_1/model/pi/output/kernel:0',
            'b/level_1/model/qf_0/fc0/bias:0',
            'b/level_1/model/qf_0/fc0/kernel:0',
            'b/level_1/model/qf_0/fc1/bias:0',
            'b/level_1/model/qf_0/fc1/kernel:0',
            'b/level_1/model/qf_0/qf_output/bias:0',
            'b/level_1/model/qf_0/qf_output/kernel:0',
            'b/level_1/model/qf_1/fc0/bias:0',
            'b/level_1/model/qf_1/fc0/kernel:0',
            'b/level_1/model/qf_1/fc1/bias:0',
            'b/level_1/model/qf_1/fc1/kernel:0',
            'b/level_1/model/qf_1/qf_output/bias:0',
            'b/level_1/model/qf_1/qf_output/kernel:0',
        ]

        target_var_list = [
            'a/level_0/target/pi/fc0/bias:0',
            'a/level_0/target/pi/fc0/kernel:0',
            'a/level_0/target/pi/fc1/bias:0',
            'a/level_0/target/pi/fc1/kernel:0',
            'a/level_0/target/pi/output/bias:0',
            'a/level_0/target/pi/output/kernel:0',
            'a/level_0/target/qf_0/fc0/bias:0',
            'a/level_0/target/qf_0/fc0/kernel:0',
            'a/level_0/target/qf_0/fc1/bias:0',
            'a/level_0/target/qf_0/fc1/kernel:0',
            'a/level_0/target/qf_0/qf_output/bias:0',
            'a/level_0/target/qf_0/qf_output/kernel:0',
            'a/level_0/target/qf_1/fc0/bias:0',
            'a/level_0/target/qf_1/fc0/kernel:0',
            'a/level_0/target/qf_1/fc1/bias:0',
            'a/level_0/target/qf_1/fc1/kernel:0',
            'a/level_0/target/qf_1/qf_output/bias:0',
            'a/level_0/target/qf_1/qf_output/kernel:0',

            'a/level_1/target/pi/fc0/bias:0',
            'a/level_1/target/pi/fc0/kernel:0',
            'a/level_1/target/pi/fc1/bias:0',
            'a/level_1/target/pi/fc1/kernel:0',
            'a/level_1/target/pi/output/bias:0',
            'a/level_1/target/pi/output/kernel:0',
            'a/level_1/target/qf_0/fc0/bias:0',
            'a/level_1/target/qf_0/fc0/kernel:0',
            'a/level_1/target/qf_0/fc1/bias:0',
            'a/level_1/target/qf_0/fc1/kernel:0',
            'a/level_1/target/qf_0/qf_output/bias:0',
            'a/level_1/target/qf_0/qf_output/kernel:0',
            'a/level_1/target/qf_1/fc0/bias:0',
            'a/level_1/target/qf_1/fc0/kernel:0',
            'a/level_1/target/qf_1/fc1/bias:0',
            'a/level_1/target/qf_1/fc1/kernel:0',
            'a/level_1/target/qf_1/qf_output/bias:0',
            'a/level_1/target/qf_1/qf_output/kernel:0',

            'b/level_0/target/pi/fc0/bias:0',
            'b/level_0/target/pi/fc0/kernel:0',
            'b/level_0/target/pi/fc1/bias:0',
            'b/level_0/target/pi/fc1/kernel:0',
            'b/level_0/target/pi/output/bias:0',
            'b/level_0/target/pi/output/kernel:0',
            'b/level_0/target/qf_0/fc0/bias:0',
            'b/level_0/target/qf_0/fc0/kernel:0',
            'b/level_0/target/qf_0/fc1/bias:0',
            'b/level_0/target/qf_0/fc1/kernel:0',
            'b/level_0/target/qf_0/qf_output/bias:0',
            'b/level_0/target/qf_0/qf_output/kernel:0',
            'b/level_0/target/qf_1/fc0/bias:0',
            'b/level_0/target/qf_1/fc0/kernel:0',
            'b/level_0/target/qf_1/fc1/bias:0',
            'b/level_0/target/qf_1/fc1/kernel:0',
            'b/level_0/target/qf_1/qf_output/bias:0',
            'b/level_0/target/qf_1/qf_output/kernel:0',

            'b/level_1/target/pi/fc0/bias:0',
            'b/level_1/target/pi/fc0/kernel:0',
            'b/level_1/target/pi/fc1/bias:0',
            'b/level_1/target/pi/fc1/kernel:0',
            'b/level_1/target/pi/output/bias:0',
            'b/level_1/target/pi/output/kernel:0',
            'b/level_1/target/qf_0/fc0/bias:0',
            'b/level_1/target/qf_0/fc0/kernel:0',
            'b/level_1/target/qf_0/fc1/bias:0',
            'b/level_1/target/qf_0/fc1/kernel:0',
            'b/level_1/target/qf_0/qf_output/bias:0',
            'b/level_1/target/qf_0/qf_output/kernel:0',
            'b/level_1/target/qf_1/fc0/bias:0',
            'b/level_1/target/qf_1/fc0/kernel:0',
            'b/level_1/target/qf_1/fc1/bias:0',
            'b/level_1/target/qf_1/fc1/kernel:0',
            'b/level_1/target/qf_1/qf_output/bias:0',
            'b/level_1/target/qf_1/qf_output/kernel:0',
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
        policy = TD3MultiGoalConditionedPolicy(**policy_params)

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
            'level_1/target/qf_1/qf_output/kernel:0',
        ]

        for model, target in zip(model_var_list, target_var_list):
            with tf.compat.v1.variable_scope(
                    tf.compat.v1.get_variable_scope(), reuse=True):
                model_val = policy.sess.run(model)
                target_val = policy.sess.run(target)
            np.testing.assert_almost_equal(model_val, target_val)


class TestSACMultiGoalConditionedPolicy(unittest.TestCase):
    """Test MultiFeedForwardPolicy in hbaselines/multiagent/h_sac.py."""

    def setUp(self):
        self.sess = tf.compat.v1.Session()

        # Shared policy parameters
        self.policy_params_shared = {
            'sess': self.sess,
            'ac_space': Box(low=-1, high=1, shape=(1,)),
            'co_space': Box(low=-2, high=2, shape=(2,)),
            'ob_space': Box(low=-3, high=3, shape=(3,)),
            'all_ob_space': Box(low=-3, high=3, shape=(10,)),
            'verbose': 0,
        }
        self.policy_params_shared.update(SAC_PARAMS.copy())
        self.policy_params_shared.update(GOAL_CONDITIONED_PARAMS.copy())
        self.policy_params_shared.update(MULTIAGENT_PARAMS.copy())
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
            'verbose': 0,
        }
        self.policy_params_independent.update(SAC_PARAMS.copy())
        self.policy_params_independent.update(GOAL_CONDITIONED_PARAMS.copy())
        self.policy_params_independent.update(MULTIAGENT_PARAMS.copy())
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
        """
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = False
        policy = SACMultiGoalConditionedPolicy(**policy_params)

        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['a/level_0/model/log_alpha:0',
             'a/level_0/model/pi/fc0/bias:0',
             'a/level_0/model/pi/fc0/kernel:0',
             'a/level_0/model/pi/fc1/bias:0',
             'a/level_0/model/pi/fc1/kernel:0',
             'a/level_0/model/pi/log_std/bias:0',
             'a/level_0/model/pi/log_std/kernel:0',
             'a/level_0/model/pi/mean/bias:0',
             'a/level_0/model/pi/mean/kernel:0',
             'a/level_0/model/value_fns/qf1/fc0/bias:0',
             'a/level_0/model/value_fns/qf1/fc0/kernel:0',
             'a/level_0/model/value_fns/qf1/fc1/bias:0',
             'a/level_0/model/value_fns/qf1/fc1/kernel:0',
             'a/level_0/model/value_fns/qf1/qf_output/bias:0',
             'a/level_0/model/value_fns/qf1/qf_output/kernel:0',
             'a/level_0/model/value_fns/qf2/fc0/bias:0',
             'a/level_0/model/value_fns/qf2/fc0/kernel:0',
             'a/level_0/model/value_fns/qf2/fc1/bias:0',
             'a/level_0/model/value_fns/qf2/fc1/kernel:0',
             'a/level_0/model/value_fns/qf2/qf_output/bias:0',
             'a/level_0/model/value_fns/qf2/qf_output/kernel:0',
             'a/level_0/model/value_fns/vf/fc0/bias:0',
             'a/level_0/model/value_fns/vf/fc0/kernel:0',
             'a/level_0/model/value_fns/vf/fc1/bias:0',
             'a/level_0/model/value_fns/vf/fc1/kernel:0',
             'a/level_0/model/value_fns/vf/vf_output/bias:0',
             'a/level_0/model/value_fns/vf/vf_output/kernel:0',
             'a/level_0/target/value_fns/vf/fc0/bias:0',
             'a/level_0/target/value_fns/vf/fc0/kernel:0',
             'a/level_0/target/value_fns/vf/fc1/bias:0',
             'a/level_0/target/value_fns/vf/fc1/kernel:0',
             'a/level_0/target/value_fns/vf/vf_output/bias:0',
             'a/level_0/target/value_fns/vf/vf_output/kernel:0',
             'a/level_1/model/log_alpha:0',
             'a/level_1/model/pi/fc0/bias:0',
             'a/level_1/model/pi/fc0/kernel:0',
             'a/level_1/model/pi/fc1/bias:0',
             'a/level_1/model/pi/fc1/kernel:0',
             'a/level_1/model/pi/log_std/bias:0',
             'a/level_1/model/pi/log_std/kernel:0',
             'a/level_1/model/pi/mean/bias:0',
             'a/level_1/model/pi/mean/kernel:0',
             'a/level_1/model/value_fns/qf1/fc0/bias:0',
             'a/level_1/model/value_fns/qf1/fc0/kernel:0',
             'a/level_1/model/value_fns/qf1/fc1/bias:0',
             'a/level_1/model/value_fns/qf1/fc1/kernel:0',
             'a/level_1/model/value_fns/qf1/qf_output/bias:0',
             'a/level_1/model/value_fns/qf1/qf_output/kernel:0',
             'a/level_1/model/value_fns/qf2/fc0/bias:0',
             'a/level_1/model/value_fns/qf2/fc0/kernel:0',
             'a/level_1/model/value_fns/qf2/fc1/bias:0',
             'a/level_1/model/value_fns/qf2/fc1/kernel:0',
             'a/level_1/model/value_fns/qf2/qf_output/bias:0',
             'a/level_1/model/value_fns/qf2/qf_output/kernel:0',
             'a/level_1/model/value_fns/vf/fc0/bias:0',
             'a/level_1/model/value_fns/vf/fc0/kernel:0',
             'a/level_1/model/value_fns/vf/fc1/bias:0',
             'a/level_1/model/value_fns/vf/fc1/kernel:0',
             'a/level_1/model/value_fns/vf/vf_output/bias:0',
             'a/level_1/model/value_fns/vf/vf_output/kernel:0',
             'a/level_1/target/value_fns/vf/fc0/bias:0',
             'a/level_1/target/value_fns/vf/fc0/kernel:0',
             'a/level_1/target/value_fns/vf/fc1/bias:0',
             'a/level_1/target/value_fns/vf/fc1/kernel:0',
             'a/level_1/target/value_fns/vf/vf_output/bias:0',
             'a/level_1/target/value_fns/vf/vf_output/kernel:0',
             'b/level_0/model/log_alpha:0',
             'b/level_0/model/pi/fc0/bias:0',
             'b/level_0/model/pi/fc0/kernel:0',
             'b/level_0/model/pi/fc1/bias:0',
             'b/level_0/model/pi/fc1/kernel:0',
             'b/level_0/model/pi/log_std/bias:0',
             'b/level_0/model/pi/log_std/kernel:0',
             'b/level_0/model/pi/mean/bias:0',
             'b/level_0/model/pi/mean/kernel:0',
             'b/level_0/model/value_fns/qf1/fc0/bias:0',
             'b/level_0/model/value_fns/qf1/fc0/kernel:0',
             'b/level_0/model/value_fns/qf1/fc1/bias:0',
             'b/level_0/model/value_fns/qf1/fc1/kernel:0',
             'b/level_0/model/value_fns/qf1/qf_output/bias:0',
             'b/level_0/model/value_fns/qf1/qf_output/kernel:0',
             'b/level_0/model/value_fns/qf2/fc0/bias:0',
             'b/level_0/model/value_fns/qf2/fc0/kernel:0',
             'b/level_0/model/value_fns/qf2/fc1/bias:0',
             'b/level_0/model/value_fns/qf2/fc1/kernel:0',
             'b/level_0/model/value_fns/qf2/qf_output/bias:0',
             'b/level_0/model/value_fns/qf2/qf_output/kernel:0',
             'b/level_0/model/value_fns/vf/fc0/bias:0',
             'b/level_0/model/value_fns/vf/fc0/kernel:0',
             'b/level_0/model/value_fns/vf/fc1/bias:0',
             'b/level_0/model/value_fns/vf/fc1/kernel:0',
             'b/level_0/model/value_fns/vf/vf_output/bias:0',
             'b/level_0/model/value_fns/vf/vf_output/kernel:0',
             'b/level_0/target/value_fns/vf/fc0/bias:0',
             'b/level_0/target/value_fns/vf/fc0/kernel:0',
             'b/level_0/target/value_fns/vf/fc1/bias:0',
             'b/level_0/target/value_fns/vf/fc1/kernel:0',
             'b/level_0/target/value_fns/vf/vf_output/bias:0',
             'b/level_0/target/value_fns/vf/vf_output/kernel:0',
             'b/level_1/model/log_alpha:0',
             'b/level_1/model/pi/fc0/bias:0',
             'b/level_1/model/pi/fc0/kernel:0',
             'b/level_1/model/pi/fc1/bias:0',
             'b/level_1/model/pi/fc1/kernel:0',
             'b/level_1/model/pi/log_std/bias:0',
             'b/level_1/model/pi/log_std/kernel:0',
             'b/level_1/model/pi/mean/bias:0',
             'b/level_1/model/pi/mean/kernel:0',
             'b/level_1/model/value_fns/qf1/fc0/bias:0',
             'b/level_1/model/value_fns/qf1/fc0/kernel:0',
             'b/level_1/model/value_fns/qf1/fc1/bias:0',
             'b/level_1/model/value_fns/qf1/fc1/kernel:0',
             'b/level_1/model/value_fns/qf1/qf_output/bias:0',
             'b/level_1/model/value_fns/qf1/qf_output/kernel:0',
             'b/level_1/model/value_fns/qf2/fc0/bias:0',
             'b/level_1/model/value_fns/qf2/fc0/kernel:0',
             'b/level_1/model/value_fns/qf2/fc1/bias:0',
             'b/level_1/model/value_fns/qf2/fc1/kernel:0',
             'b/level_1/model/value_fns/qf2/qf_output/bias:0',
             'b/level_1/model/value_fns/qf2/qf_output/kernel:0',
             'b/level_1/model/value_fns/vf/fc0/bias:0',
             'b/level_1/model/value_fns/vf/fc0/kernel:0',
             'b/level_1/model/value_fns/vf/fc1/bias:0',
             'b/level_1/model/value_fns/vf/fc1/kernel:0',
             'b/level_1/model/value_fns/vf/vf_output/bias:0',
             'b/level_1/model/value_fns/vf/vf_output/kernel:0',
             'b/level_1/target/value_fns/vf/fc0/bias:0',
             'b/level_1/target/value_fns/vf/fc0/kernel:0',
             'b/level_1/target/value_fns/vf/fc1/bias:0',
             'b/level_1/target/value_fns/vf/fc1/kernel:0',
             'b/level_1/target/value_fns/vf/vf_output/bias:0',
             'b/level_1/target/value_fns/vf/vf_output/kernel:0']
        )

        # Check observation/action/context spaces of the agents
        self.assertEqual(
            policy.agents['a'].policy[0].ac_space.shape[0],
            self.policy_params_independent['ob_space']['a'].shape[0]
        )
        self.assertEqual(
            policy.agents['a'].policy[0].ob_space.shape[0],
            self.policy_params_independent['ob_space']['a'].shape[0]
        )
        self.assertEqual(
            policy.agents['a'].policy[0].co_space.shape[0],
            self.policy_params_independent['co_space']['a'].shape[0]
        )

        self.assertEqual(
            policy.agents['a'].policy[1].ac_space.shape[0],
            self.policy_params_independent['ac_space']['a'].shape[0]
        )
        self.assertEqual(
            policy.agents['a'].policy[1].ob_space.shape[0],
            self.policy_params_independent['ob_space']['a'].shape[0]
        )
        self.assertEqual(
            policy.agents['a'].policy[1].co_space.shape[0],
            self.policy_params_independent['ob_space']['a'].shape[0]
        )

        self.assertEqual(
            policy.agents['b'].policy[0].ac_space.shape[0],
            self.policy_params_independent['ob_space']['b'].shape[0]
        )
        self.assertEqual(
            policy.agents['b'].policy[0].ob_space.shape[0],
            self.policy_params_independent['ob_space']['b'].shape[0]
        )
        self.assertEqual(
            policy.agents['b'].policy[0].co_space.shape[0],
            self.policy_params_independent['co_space']['b'].shape[0]
        )

        self.assertEqual(
            policy.agents['b'].policy[1].ac_space.shape[0],
            self.policy_params_independent['ac_space']['b'].shape[0]
        )
        self.assertEqual(
            policy.agents['b'].policy[1].ob_space.shape[0],
            self.policy_params_independent['ob_space']['b'].shape[0]
        )
        self.assertEqual(
            policy.agents['b'].policy[1].co_space.shape[0],
            self.policy_params_independent['ob_space']['b'].shape[0]
        )

        # Check the instantiation of the class attributes.
        self.assertTrue(not policy.shared)
        self.assertTrue(not policy.maddpg)

    def test_init_2(self):
        policy_params = self.policy_params_shared.copy()
        policy_params["maddpg"] = False
        policy = SACMultiGoalConditionedPolicy(**policy_params)

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
             'level_1/target/value_fns/vf/vf_output/kernel:0']
        )

        # Check observation/action/context spaces of the agents
        self.assertEqual(
            policy.agents['policy'].policy[0].ac_space.shape[0],
            self.policy_params_shared['ob_space'].shape[0]
        )
        self.assertEqual(
            policy.agents['policy'].policy[0].ob_space.shape[0],
            self.policy_params_shared['ob_space'].shape[0]
        )
        self.assertEqual(
            policy.agents['policy'].policy[0].co_space.shape[0],
            self.policy_params_shared['co_space'].shape[0]
        )

        self.assertEqual(
            policy.agents['policy'].policy[1].ac_space.shape[0],
            self.policy_params_shared['ac_space'].shape[0]
        )
        self.assertEqual(
            policy.agents['policy'].policy[1].ob_space.shape[0],
            self.policy_params_shared['ob_space'].shape[0]
        )
        self.assertEqual(
            policy.agents['policy'].policy[1].co_space.shape[0],
            self.policy_params_shared['ob_space'].shape[0]
        )

        # Check the instantiation of the class attributes.
        self.assertTrue(policy.shared)
        self.assertTrue(not policy.maddpg)

    def test_initialize_1(self):
        """Check the functionality of the initialize() method.

        This test validates that the target variables are properly initialized
        when initialize is called.

        This is done for the following cases:

        1. maddpg = False, shared = False
        2. maddpg = False, shared = True
        """
        policy_params = self.policy_params_independent.copy()
        policy_params["maddpg"] = False
        policy = SACMultiGoalConditionedPolicy(**policy_params)

        # Initialize the variables of the policy.
        policy.sess.run(tf.compat.v1.global_variables_initializer())

        # Run the initialize method.
        policy.initialize()

        model_var_list = [
            'a/level_0/model/value_fns/vf/fc0/kernel:0',
            'a/level_0/model/value_fns/vf/fc0/bias:0',
            'a/level_0/model/value_fns/vf/fc1/kernel:0',
            'a/level_0/model/value_fns/vf/fc1/bias:0',
            'a/level_0/model/value_fns/vf/vf_output/kernel:0',
            'a/level_0/model/value_fns/vf/vf_output/bias:0',

            'a/level_1/model/value_fns/vf/fc0/kernel:0',
            'a/level_1/model/value_fns/vf/fc0/bias:0',
            'a/level_1/model/value_fns/vf/fc1/kernel:0',
            'a/level_1/model/value_fns/vf/fc1/bias:0',
            'a/level_1/model/value_fns/vf/vf_output/kernel:0',
            'a/level_1/model/value_fns/vf/vf_output/bias:0',

            'b/level_0/model/value_fns/vf/fc0/kernel:0',
            'b/level_0/model/value_fns/vf/fc0/bias:0',
            'b/level_0/model/value_fns/vf/fc1/kernel:0',
            'b/level_0/model/value_fns/vf/fc1/bias:0',
            'b/level_0/model/value_fns/vf/vf_output/kernel:0',
            'b/level_0/model/value_fns/vf/vf_output/bias:0',

            'b/level_1/model/value_fns/vf/fc0/kernel:0',
            'b/level_1/model/value_fns/vf/fc0/bias:0',
            'b/level_1/model/value_fns/vf/fc1/kernel:0',
            'b/level_1/model/value_fns/vf/fc1/bias:0',
            'b/level_1/model/value_fns/vf/vf_output/kernel:0',
            'b/level_1/model/value_fns/vf/vf_output/bias:0',
        ]

        target_var_list = [
            'a/level_0/target/value_fns/vf/fc0/kernel:0',
            'a/level_0/target/value_fns/vf/fc0/bias:0',
            'a/level_0/target/value_fns/vf/fc1/kernel:0',
            'a/level_0/target/value_fns/vf/fc1/bias:0',
            'a/level_0/target/value_fns/vf/vf_output/kernel:0',
            'a/level_0/target/value_fns/vf/vf_output/bias:0',

            'a/level_1/target/value_fns/vf/fc0/kernel:0',
            'a/level_1/target/value_fns/vf/fc0/bias:0',
            'a/level_1/target/value_fns/vf/fc1/kernel:0',
            'a/level_1/target/value_fns/vf/fc1/bias:0',
            'a/level_1/target/value_fns/vf/vf_output/kernel:0',
            'a/level_1/target/value_fns/vf/vf_output/bias:0',

            'b/level_0/target/value_fns/vf/fc0/kernel:0',
            'b/level_0/target/value_fns/vf/fc0/bias:0',
            'b/level_0/target/value_fns/vf/fc1/kernel:0',
            'b/level_0/target/value_fns/vf/fc1/bias:0',
            'b/level_0/target/value_fns/vf/vf_output/kernel:0',
            'b/level_0/target/value_fns/vf/vf_output/bias:0',

            'b/level_1/target/value_fns/vf/fc0/kernel:0',
            'b/level_1/target/value_fns/vf/fc0/bias:0',
            'b/level_1/target/value_fns/vf/fc1/kernel:0',
            'b/level_1/target/value_fns/vf/fc1/bias:0',
            'b/level_1/target/value_fns/vf/vf_output/kernel:0',
            'b/level_1/target/value_fns/vf/vf_output/bias:0',
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
        policy = SACMultiGoalConditionedPolicy(**policy_params)

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


if __name__ == '__main__':
    unittest.main()
