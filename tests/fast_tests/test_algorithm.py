"""Contains tests for the model abstractions and different models."""
import unittest
import numpy as np
import random
import shutil

from hbaselines.hiro.algorithm import as_scalar, TD3
from hbaselines.hiro.tf_util import get_trainable_vars
from hbaselines.hiro.policy import FeedForwardPolicy, GoalDirectedPolicy


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
            'env': 'MountainCarContinuous-v0',
            'gamma': 0.99,
            'eval_env': None,
            'nb_train_steps': 50,
            'nb_rollout_steps': 100,
            'nb_eval_episodes': 50,
            'normalize_observations': False,
            'tau': 0.001,
            'batch_size': 128,
            'normalize_returns': False,
            'observation_range': (-5., 5.),
            'critic_l2_reg': 0.,
            'return_range': (-np.inf, np.inf),
            'actor_lr': 1e-4,
            'critic_lr': 1e-3,
            'clip_norm': None,
            'reward_scale': 1.,
            'render': False,
            'render_eval': False,
            'memory_limit': None,
            'buffer_size': 50000,
            'random_exploration': 0.0,
            'verbose': 0,
            '_init_setup_model': True
        }

    def test_init(self):
        """Ensure that the parameters at init are as expected."""
        # Create the algorithm object.
        policy_params = self.init_parameters.copy()
        policy_params['_init_setup_model'] = False
        alg = TD3(**policy_params)

        # Test the attribute values.
        self.assertEqual(alg.gamma, self.init_parameters['gamma'])
        self.assertEqual(alg.nb_train_steps,
                         self.init_parameters['nb_train_steps'])
        self.assertEqual(alg.nb_rollout_steps,
                         self.init_parameters['nb_rollout_steps'])
        self.assertEqual(alg.nb_eval_episodes,
                         self.init_parameters['nb_eval_episodes'])
        self.assertEqual(alg.normalize_observations,
                         self.init_parameters['normalize_observations'])
        self.assertEqual(alg.tau, self.init_parameters['tau'])
        self.assertEqual(alg.batch_size, self.init_parameters['batch_size'])
        self.assertEqual(alg.normalize_returns,
                         self.init_parameters['normalize_returns'])
        self.assertEqual(alg.observation_range,
                         self.init_parameters['observation_range'])
        self.assertEqual(alg.critic_l2_reg,
                         self.init_parameters['critic_l2_reg'])
        self.assertEqual(alg.return_range,
                         self.init_parameters['return_range'])
        self.assertEqual(alg.actor_lr, self.init_parameters['actor_lr'])
        self.assertEqual(alg.critic_lr, self.init_parameters['critic_lr'])
        self.assertEqual(alg.clip_norm, self.init_parameters['clip_norm'])
        self.assertEqual(alg.reward_scale,
                         self.init_parameters['reward_scale'])
        self.assertEqual(alg.render, self.init_parameters['render'])
        self.assertEqual(alg.render_eval, self.init_parameters['render_eval'])
        self.assertEqual(alg.memory_limit,
                         self.init_parameters['memory_limit'])
        self.assertEqual(alg.buffer_size, self.init_parameters['buffer_size'])
        self.assertEqual(alg.random_exploration,
                         self.init_parameters['random_exploration'])
        self.assertEqual(alg.verbose, self.init_parameters['verbose'])

    def test_setup_model_feedforward(self):
        # Create the algorithm object.
        policy_params = self.init_parameters.copy()
        policy_params['policy'] = FeedForwardPolicy
        policy_params['_init_setup_model'] = True
        alg = TD3(**policy_params)

        with alg.graph.as_default():
            expected_vars = sorted([var.name for var in get_trainable_vars()])

        # Check that all trainable variables have been created in the
        # TensorFlow graph.
        self.assertListEqual(
            expected_vars,
            ['model/pi/fc0/bias:0',
             'model/pi/fc0/kernel:0',
             'model/pi/fc1/bias:0',
             'model/pi/fc1/kernel:0',
             'model/pi/pi/bias:0',
             'model/pi/pi/kernel:0',
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
             'target/pi/pi/bias:0',
             'target/pi/pi/kernel:0',
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

    def test_setup_model_goal_directed(self):
        # Create the algorithm object.
        policy_params = self.init_parameters.copy()
        policy_params['policy'] = GoalDirectedPolicy
        policy_params['_init_setup_model'] = True
        alg = TD3(**policy_params)

        with alg.graph.as_default():
            expected_vars = sorted([var.name for var in get_trainable_vars()])

        # Check that all trainable variables have been created in the
        # TensorFlow graph.
        self.assertListEqual(
            expected_vars,
            ['Manager/model/pi/fc0/bias:0',
             'Manager/model/pi/fc0/kernel:0',
             'Manager/model/pi/fc1/bias:0',
             'Manager/model/pi/fc1/kernel:0',
             'Manager/model/pi/pi/bias:0',
             'Manager/model/pi/pi/kernel:0',
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
             'Manager/target/pi/pi/bias:0',
             'Manager/target/pi/pi/kernel:0',
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
             'Worker/model/pi/pi/bias:0',
             'Worker/model/pi/pi/kernel:0',
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
             'Worker/target/pi/pi/bias:0',
             'Worker/target/pi/pi/kernel:0',
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

    def test_learn_init(self):
        """Test the non-loop components of the `learn` method."""
        # Create the algorithm object.
        policy_params = self.init_parameters.copy()
        policy_params['policy'] = GoalDirectedPolicy
        policy_params['_init_setup_model'] = True
        alg = TD3(**policy_params)

        # Run the learn operation for zero timesteps.
        alg.learn(0, log_dir='results')
        self.assertEqual(alg.episode_reward, 0)
        self.assertEqual(alg.episode_step, 0)
        self.assertEqual(alg.episodes, 0)
        self.assertEqual(alg.total_steps, 0)
        self.assertEqual(alg.epoch, 0)
        self.assertEqual(len(alg.episode_rewards_history), 0)
        self.assertEqual(alg.epoch_episodes, 0)
        self.assertEqual(len(alg.epoch_actions), 0)
        self.assertEqual(len(alg.epoch_qs), 0)
        self.assertEqual(len(alg.epoch_actor_losses), 0)
        self.assertEqual(len(alg.epoch_critic_losses), 0)
        self.assertEqual(len(alg.epoch_episode_rewards), 0)
        self.assertEqual(len(alg.epoch_episode_steps), 0)
        shutil.rmtree('results')

        # Test the seeds.
        alg.learn(0, log_dir='results', seed=1)
        self.assertEqual(np.random.sample(), 0.417022004702574)
        self.assertEqual(random.uniform(0, 1), 0.13436424411240122)
        shutil.rmtree('results')

    def test_fingerprints(self):
        """Validate the functionality of the fingerprints.

        When the fingerprint functionality is turned on, the observation within
        the algorithm (stored under self.obs) should always include the
        fingerprint element.

        Policy-specific features of the fingerprint implementation are tested
        under test_policy.py
        """
        # Create the algorithm.  # TODO
        pass

        # Validate that observations include the fingerprints elements upon
        # initializing the `learn` procedure.  # TODO
        pass

        # Validate that observations include the fingerprints elements during
        # a step in collect_samples.  # TODO
        pass

        # Validate that observations include the fingerprints elements during
        # a reset in collect_samples.  # TODO
        pass


if __name__ == '__main__':
    unittest.main()
