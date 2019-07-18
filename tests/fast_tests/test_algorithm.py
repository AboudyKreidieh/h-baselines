"""Contains tests for the model abstractions and different models."""
import unittest
import numpy as np

from hbaselines.hiro.algorithm import as_scalar  # , TD3
# from hbaselines.hiro.replay_buffer import ReplayBuffer


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


class TestTD3(unittest.TestCase):
    """Test the components of the TD3 algorithm."""

    def setUp(self):
        self.env = 'MountainCarContinuous-v0'

        self.init_parameters = {
            'policy': None,
            'env': None,
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

        # alg = TD3(**policy_params)
        pass

    def test_setup_model_feedforward(self):
        pass

    def test_setup_model_goal_directed(self):
        pass


if __name__ == '__main__':
    unittest.main()
