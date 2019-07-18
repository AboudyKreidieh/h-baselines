"""Contains tests for the model abstractions and different models."""
import unittest
import numpy as np
import tensorflow as tf

from hbaselines.hiro.algorithm import as_scalar, TD3
from hbaselines.hiro.replay_buffer import ReplayBuffer


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


class TestTD3(unittest.TestCase):
    """Test the components of the TD3 algorithm."""

    def setUp(self):
        self.env = 'MountainCarContinuous-v0'

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

        alg = TD3(**policy_params)
        self.assertEqual(alg.gamma, policy_params['gamma'])
        self.assertEqual(alg.tau, policy_params['tau'])
        self.assertEqual(alg.normalize_observations,
                         policy_params['normalize_observations'])
        self.assertEqual(alg.normalize_returns,
                         policy_params['normalize_returns'])
        self.assertEqual(alg.return_range, policy_params['return_range'])
        self.assertEqual(alg.observation_range,
                         policy_params['observation_range'])
        self.assertEqual(alg.actor_lr, policy_params['actor_lr'])
        self.assertEqual(alg.critic_lr, policy_params['critic_lr'])
        self.assertEqual(alg.clip_norm, policy_params['clip_norm'])
        self.assertEqual(alg.reward_scale, policy_params['reward_scale'])
        self.assertEqual(alg.batch_size, policy_params['batch_size'])
        self.assertEqual(alg.critic_l2_reg, policy_params['critic_l2_reg'])
        self.assertEqual(alg.render, policy_params['render'])
        self.assertEqual(alg.nb_train_steps, policy_params['nb_train_steps'])
        self.assertEqual(alg.nb_rollout_steps,
                         policy_params['nb_rollout_steps'])
        self.assertEqual(alg.memory_limit, policy_params['memory_limit'])

    def test_setup_model_feedforward(self):
        """Ensure that the correct policies were generated in the
        non-recurrent, non-hierarchical case."""
        policy_params = self.init_parameters.copy()
        policy_params['env'] = self.env
        # policy_params['policy'] = FullyConnectedPolicy
        policy_params['_init_setup_model'] = True
        alg = TD3(**policy_params)

        # Check the primary policy.
        with alg.graph.as_default():
            # get the training variable of the policy
            with tf.variable_scope('model'):
                tv = tf.trainable_variables()

            # check that the training variables of the policy are as expected
            # (note that these include the policy and and target)
            expected_vars = ['model/pi/fc_0/kernel:0',
                             'model/pi/fc_0/bias:0',
                             'model/pi/fc_1/kernel:0',
                             'model/pi/fc_1/bias:0',
                             'model/pi/fc_output/kernel:0',
                             'model/pi/fc_output/bias:0',
                             'model/qf/normalized_critic_0/kernel:0',
                             'model/qf/normalized_critic_0/bias:0',
                             'model/qf/normalized_critic_1/kernel:0',
                             'model/qf/normalized_critic_1/bias:0',
                             'model/qf/normalized_critic_output/kernel:0',
                             'model/qf/normalized_critic_output/bias:0',
                             'target/pi/fc_0/kernel:0',
                             'target/pi/fc_0/bias:0',
                             'target/pi/fc_1/kernel:0',
                             'target/pi/fc_1/bias:0',
                             'target/pi/fc_output/kernel:0',
                             'target/pi/fc_output/bias:0',
                             'target/qf/normalized_critic_0/kernel:0',
                             'target/qf/normalized_critic_0/bias:0',
                             'target/qf/normalized_critic_1/kernel:0',
                             'target/qf/normalized_critic_1/bias:0',
                             'target/qf/normalized_critic_output/kernel:0',
                             'target/qf/normalized_critic_output/bias:0']
            actual_vars = [tv_i.name for tv_i in tv]
            self.assertCountEqual(expected_vars, actual_vars)

            # check that the shapes of the policy and target match
            policy_names = [tv_i.name[6:] for tv_i in
                            tv if "model" in tv_i.name]

            with tf.variable_scope('model', reuse=True):
                for name in policy_names:
                    t1 = next(tv_i for tv_i in tv
                              if tv_i.name == 'model/{}'.format(name))
                    t2 = next(tv_i for tv_i in tv
                              if tv_i.name == 'target/{}'.format(name))
                    self.assertEqual(t1.shape, t2.shape)

        # Check the stats.
        expected_stats = [
            'reference_Q_mean', 'reference_Q_std', 'reference_actor_Q_mean',
            'reference_actor_Q_std', 'reference_action_mean',
            'reference_action_std'
        ]
        self.assertCountEqual(alg.stats_names, expected_stats)

    def test_setup_model_goal_directed(self):
        policy_params = self.init_parameters.copy()
        policy_params['env'] = self.env
        # policy_params['policy'] = LSTMPolicy
        policy_params['recurrent'] = True
        policy_params['_init_setup_model'] = True
        alg = TD3(**policy_params)
        self.assertEqual(alg.memory_policy, ReplayBuffer)
        self.assertEqual(alg.recurrent, True)
        self.assertEqual(alg.hierarchical, False)

        # Check the primary policy.
        with alg.graph.as_default():
            # get the training variable of the policy
            with tf.variable_scope('model'):
                tv = tf.trainable_variables()

            # check that the training variables of the policy are as expected
            # (note that these include the policy and and target)
            expected_vars = ['model/pi/rnn/basic_lstm_cell/kernel:0',
                             'model/pi/rnn/basic_lstm_cell/bias:0',
                             'model/pi/lstm_output/kernel:0',
                             'model/pi/lstm_output/bias:0',
                             'model/qf/normalized_critic_0/kernel:0',
                             'model/qf/normalized_critic_0/bias:0',
                             'model/qf/normalized_critic_1/kernel:0',
                             'model/qf/normalized_critic_1/bias:0',
                             'model/qf/normalized_critic_output/kernel:0',
                             'model/qf/normalized_critic_output/bias:0',
                             'target/pi/rnn/basic_lstm_cell/kernel:0',
                             'target/pi/rnn/basic_lstm_cell/bias:0',
                             'target/pi/lstm_output/kernel:0',
                             'target/pi/lstm_output/bias:0',
                             'target/qf/normalized_critic_0/kernel:0',
                             'target/qf/normalized_critic_0/bias:0',
                             'target/qf/normalized_critic_1/kernel:0',
                             'target/qf/normalized_critic_1/bias:0',
                             'target/qf/normalized_critic_output/kernel:0',
                             'target/qf/normalized_critic_output/bias:0']

            actual_vars = [tv_i.name for tv_i in tv]
            self.assertCountEqual(expected_vars, actual_vars)

            # check that the shapes of the policy and target match
            policy_names = [tv_i.name[6:] for tv_i in
                            tv if "model" in tv_i.name]

            with tf.variable_scope('model', reuse=True):
                for name in policy_names:
                    t1 = next(tv_i for tv_i in tv
                              if tv_i.name == 'model/{}'.format(name))
                    t2 = next(tv_i for tv_i in tv
                              if tv_i.name == 'target/{}'.format(name))
                    self.assertEqual(t1.shape, t2.shape)


if __name__ == '__main__':
    unittest.main()
