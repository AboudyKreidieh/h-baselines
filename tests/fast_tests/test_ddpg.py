"""Contains tests for the model abstractions and different models."""
import unittest
import numpy as np

from hbaselines.algs.ddpg import as_scalar, DDPG
from hbaselines.utils.exp_replay import GenericMemory, RecurrentMemory
from hbaselines.utils.exp_replay import HierarchicalRecurrentMemory


class TestAuxiliaryMethods(unittest.TestCase):
    """Tests the auxiliary methods in algs/ddpg.py"""

    def test_as_scalar(self):
        # test if input is a single element
        test_scalar = 3.4
        self.assertAlmostEqual(test_scalar, as_scalar(test_scalar))

        # test if input is an np.ndarray with a single element
        test_scalar = np.array([3.4])
        self.assertAlmostEqual(test_scalar[0], as_scalar(test_scalar))

        # test if input is an np.ndarray with multiple elements
        test_scalar = [3.4, 1]
        self.assertRaises(ValueError, as_scalar, scalar=test_scalar)

    def test_get_target_updates(self):
        pass


class TestDDPG(unittest.TestCase):
    """Test the components of the DDPG algorithm."""

    def setUp(self):
        self.env = 'HalfCheetah-v2'

        self.init_parameters = {
            'policy': None,
            'env': None,
            'recurrent': False,
            'hierarchical': False,
            'gamma': 0.99,
            'memory_policy': None,
            'nb_train_steps': 50,
            'nb_rollout_steps': 100,
            'action_noise': None,
            'normalize_observations': False,
            'tau': 0.001,
            'batch_size': 128,
            'normalize_returns': False,
            'observation_range': (-5, 5),
            'critic_l2_reg': 0.,
            'return_range': (-np.inf, np.inf),
            'actor_lr': 1e-4,
            'critic_lr': 1e-3,
            'clip_norm': None,
            'reward_scale': 1.,
            'render': False,
            'memory_limit': 100,
            'verbose': 0,
            'tensorboard_log': None,
            '_init_setup_model': True
        }

    def test_init(self):
        """Ensure that the parameters at init are as expected."""
        # Part 1. Fully Connected Network
        policy_params = self.init_parameters.copy()
        policy_params['env'] = self.env
        policy_params['_init_setup_model'] = False
        policy = DDPG(**policy_params)
        self.assertEqual(policy.gamma, policy_params['gamma'])
        self.assertEqual(policy.tau, policy_params['tau'])
        self.assertEqual(policy.normalize_observations,
                         policy_params['normalize_observations'])
        self.assertEqual(policy.normalize_returns,
                         policy_params['normalize_returns'])
        self.assertEqual(policy.action_noise, policy_params['action_noise'])
        self.assertEqual(policy.return_range, policy_params['return_range'])
        self.assertEqual(policy.observation_range,
                         policy_params['observation_range'])
        self.assertEqual(policy.actor_lr, policy_params['actor_lr'])
        self.assertEqual(policy.critic_lr, policy_params['critic_lr'])
        self.assertEqual(policy.clip_norm, policy_params['clip_norm'])
        self.assertEqual(policy.reward_scale, policy_params['reward_scale'])
        self.assertEqual(policy.batch_size, policy_params['batch_size'])
        self.assertEqual(policy.critic_l2_reg, policy_params['critic_l2_reg'])
        self.assertEqual(policy.render, policy_params['render'])
        self.assertEqual(policy.nb_train_steps,
                         policy_params['nb_train_steps'])
        self.assertEqual(policy.nb_rollout_steps,
                         policy_params['nb_rollout_steps'])
        self.assertEqual(policy.memory_limit, policy_params['memory_limit'])
        self.assertEqual(policy.tensorboard_log,
                         policy_params['tensorboard_log'])
        self.assertEqual(policy.memory_policy, GenericMemory)
        self.assertEqual(policy.recurrent, False)
        self.assertEqual(policy.hierarchical, False)

        # Part 2. Recurrent Policies
        policy_params = self.init_parameters.copy()
        policy_params['env'] = self.env
        policy_params['recurrent'] = True
        policy_params['_init_setup_model'] = False
        policy = DDPG(**policy_params)
        self.assertEqual(policy.memory_policy, RecurrentMemory)
        self.assertEqual(policy.recurrent, True)
        self.assertEqual(policy.hierarchical, False)

        # Part 3. Hierarchical Policies
        policy_params = self.init_parameters.copy()
        policy_params['env'] = self.env
        policy_params['hierarchical'] = True
        policy_params['_init_setup_model'] = False
        policy = DDPG(**policy_params)
        self.assertEqual(policy.memory_policy, HierarchicalRecurrentMemory)
        self.assertEqual(policy.recurrent, False)
        self.assertEqual(policy.hierarchical, True)

    def test_setup_model_stats(self):
        """Ensure that the correct policies were generated."""
        # Part 1. Fully Connected Network

        # Part 2. Recurrent Policies

        # Part 3. Hierarchical Policies

        pass

    def test_normalize_observations(self):
        pass

    def test_action_noise(self):
        pass


if __name__ == '__main__':
    unittest.main()
