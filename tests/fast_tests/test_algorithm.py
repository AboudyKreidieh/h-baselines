"""Contains tests for the model abstractions and different models."""
import unittest
import numpy as np
import random
import shutil

from hbaselines.hiro.algorithm import as_scalar, TD3
from hbaselines.hiro.tf_util import get_trainable_vars
from hbaselines.hiro.policy import FeedForwardPolicy, GoalDirectedPolicy
from hbaselines.hiro.algorithm import FEEDFORWARD_POLICY_KWARGS
from hbaselines.hiro.algorithm import GOAL_DIRECTED_POLICY_KWARGS


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
            'eval_env': None,
            'num_cpus': 1,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1.,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'policy_kwargs': None,
            '_init_setup_model': True
        }

    def test_init(self):
        """Ensure that the parameters at init are as expected."""
        # Create the algorithm object.
        policy_params = self.init_parameters.copy()
        policy_params['_init_setup_model'] = False
        alg = TD3(**policy_params)

        # Test the attribute values.
        self.assertEqual(alg.policy, self.init_parameters['policy'])
        self.assertEqual(alg.eval_env, self.init_parameters['eval_env'])
        self.assertEqual(alg.num_cpus, self.init_parameters['num_cpus'])
        self.assertEqual(alg.nb_train_steps,
                         self.init_parameters['nb_train_steps'])
        self.assertEqual(alg.nb_rollout_steps,
                         self.init_parameters['nb_rollout_steps'])
        self.assertEqual(alg.nb_eval_episodes,
                         self.init_parameters['nb_eval_episodes'])
        self.assertEqual(alg.reward_scale,
                         self.init_parameters['reward_scale'])
        self.assertEqual(alg.render, self.init_parameters['render'])
        self.assertEqual(alg.render_eval, self.init_parameters['render_eval'])
        self.assertEqual(alg.verbose, self.init_parameters['verbose'])

    def test_setup_model_feedforward(self):
        # Create the algorithm object.
        policy_params = self.init_parameters.copy()
        policy_params['policy'] = FeedForwardPolicy
        policy_params['_init_setup_model'] = True
        alg = TD3(**policy_params)

        # check the policy_kwargs term
        policy_kwargs = FEEDFORWARD_POLICY_KWARGS.copy()
        policy_kwargs['verbose'] = self.init_parameters['verbose']
        self.assertDictEqual(alg.policy_kwargs, policy_kwargs)

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

    def test_setup_model_goal_directed(self):
        # Create the algorithm object.
        policy_params = self.init_parameters.copy()
        policy_params['policy'] = GoalDirectedPolicy
        policy_params['_init_setup_model'] = True
        alg = TD3(**policy_params)

        # check the policy_kwargs term
        policy_kwargs = GOAL_DIRECTED_POLICY_KWARGS.copy()
        policy_kwargs['verbose'] = self.init_parameters['verbose']
        policy_kwargs['env_name'] = self.init_parameters['env']
        self.assertDictEqual(alg.policy_kwargs, policy_kwargs)

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
             'Manager/model/pi/output/bias:0',
             'Manager/model/pi/output/kernel:0',
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
             'Manager/target/pi/output/bias:0',
             'Manager/target/pi/output/kernel:0',
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
             'Worker/model/pi/output/bias:0',
             'Worker/model/pi/output/kernel:0',
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
             'Worker/target/pi/output/bias:0',
             'Worker/target/pi/output/kernel:0',
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
        alg.learn(0, log_dir='results', start_timesteps=0)
        self.assertEqual(alg.episode_reward, 0)
        self.assertEqual(alg.episode_step, 0)
        self.assertEqual(alg.episodes, 0)
        self.assertEqual(alg.total_steps, 0)
        self.assertEqual(alg.epoch, 0)
        self.assertEqual(len(alg.episode_rewards_history), 0)
        self.assertEqual(alg.epoch_episodes, 0)
        self.assertEqual(len(alg.epoch_actions), 0)
        self.assertEqual(len(alg.epoch_q1s), 0)
        self.assertEqual(len(alg.epoch_q2s), 0)
        self.assertEqual(len(alg.epoch_actor_losses), 0)
        self.assertEqual(len(alg.epoch_critic_losses), 0)
        self.assertEqual(len(alg.epoch_episode_rewards), 0)
        self.assertEqual(len(alg.epoch_episode_steps), 0)
        shutil.rmtree('results')

        # Test the seeds.
        alg.learn(0, log_dir='results', seed=1, start_timesteps=0)
        self.assertEqual(np.random.sample(), 0.417022004702574)
        self.assertEqual(random.uniform(0, 1), 0.13436424411240122)
        shutil.rmtree('results')

    def test_learn_start_timesteps(self):
        """TODO"""
        pass

    def test_collect_samples(self):
        """Validate the functionality of the _collect_samples method."""
        pass

    def test_evaluate(self):
        """Validate the functionality of the _evaluate method."""
        pass

    def test_fingerprints(self):
        """Validate the functionality of the fingerprints.

        When the fingerprint functionality is turned on, the observation within
        the algorithm (stored under self.obs) should always include the
        fingerprint element.

        Policy-specific features of the fingerprint implementation are also
        tested here. This feature should add a fingerprint dimension to the
        manager and worker observation spaces, but NOT the context space of the
        worker or the action space of the manager. The worker reward function
        should also be ignoring the fingerprint elements  during its
        computation. The fingerprint elements are passed by the algorithm, and
        tested under test_algorithm.py
        """
        # Create the algorithm.
        policy_params = self.init_parameters.copy()
        policy_params['policy'] = GoalDirectedPolicy
        policy_params['nb_rollout_steps'] = 1
        policy_params['policy_kwargs'] = {'use_fingerprints': True}
        alg = TD3(**policy_params)

        # Test the observation spaces of the manager and worker, as well as the
        # context space of the worker and action space of the manager.
        self.assertTupleEqual(alg.policy_tf.manager.ob_space.shape, (3,))
        self.assertTupleEqual(alg.policy_tf.manager.ac_space.shape, (2,))
        self.assertTupleEqual(alg.policy_tf.worker.ob_space.shape, (3,))
        self.assertTupleEqual(alg.policy_tf.worker.co_space.shape, (2,))

        # Test worker_reward method within the policy.
        self.assertAlmostEqual(
            alg.policy_tf.worker_reward(states=np.array([1, 2, 3]),
                                        goals=np.array([0, 0]),
                                        next_states=np.array([1, 2, 3])),
            -np.sqrt(1**2 + 2**2)
        )

        # Validate that observations include the fingerprints elements upon
        # initializing the `learn` procedure and  during a step in the
        # `_collect_samples` method.
        alg.learn(1, log_dir='results', log_interval=1, start_timesteps=0)
        self.assertEqual(
            len(alg.obs),
            alg.env.observation_space.shape[0] + alg.fingerprint_dim[0])
        np.testing.assert_almost_equal(
            alg.obs[-alg.fingerprint_dim[0]:], np.array([0]))

        # Validate that observations include the fingerprints elements during
        # a reset in the `_collect_samples` method.
        alg.learn(500, log_dir='results', log_interval=500, start_timesteps=0)
        self.assertEqual(
            len(alg.obs),
            alg.env.observation_space.shape[0] + alg.fingerprint_dim[0])
        np.testing.assert_almost_equal(
            alg.obs[-alg.fingerprint_dim[0]:], np.array([4.99]))

        # Delete generated files.
        shutil.rmtree('results')


if __name__ == '__main__':
    unittest.main()
