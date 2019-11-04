"""Contains tests for the model abstractions and different models."""
import unittest
import numpy as np
from hbaselines.utils.train import parse_options, get_hyperparameters
from hbaselines.utils.reward_fns import negative_distance
from hbaselines.goal_conditioned.algorithm import GoalConditionedPolicy
from hbaselines.goal_conditioned.algorithm import FEEDFORWARD_POLICY_KWARGS
from hbaselines.goal_conditioned.algorithm import GOAL_DIRECTED_POLICY_KWARGS


class TestTrain(unittest.TestCase):
    """A simple test to get Travis running."""

    def test_parse_options(self):
        # Test the default case.
        args = parse_options("", "", args=["AntMaze"])
        expected_args = {
            'env_name': 'AntMaze',
            'evaluate': False,
            'n_training': 1,
            'total_steps': 1000000,
            'seed': 1,
            'num_cpus': 1,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'buffer_size': FEEDFORWARD_POLICY_KWARGS['buffer_size'],
            'batch_size': FEEDFORWARD_POLICY_KWARGS['batch_size'],
            'actor_lr': FEEDFORWARD_POLICY_KWARGS['actor_lr'],
            'critic_lr': FEEDFORWARD_POLICY_KWARGS['critic_lr'],
            'tau': FEEDFORWARD_POLICY_KWARGS['tau'],
            'gamma': FEEDFORWARD_POLICY_KWARGS['gamma'],
            'noise': FEEDFORWARD_POLICY_KWARGS['noise'],
            'target_policy_noise': FEEDFORWARD_POLICY_KWARGS[
                'target_policy_noise'],
            'target_noise_clip': FEEDFORWARD_POLICY_KWARGS[
                'target_noise_clip'],
            'layer_norm': False,
            'use_huber': False,
            'meta_period': GOAL_DIRECTED_POLICY_KWARGS['meta_period'],
            'relative_goals': False,
            'off_policy_corrections': False,
            'use_fingerprints': False,
            'centralized_value_functions': False,
            'connected_gradients': False,
            'cg_weights': GOAL_DIRECTED_POLICY_KWARGS['cg_weights'],
        }
        self.assertDictEqual(vars(args), expected_args)

        # Test custom cases.
        args = parse_options("", "", args=[
            "AntMaze",
            '--evaluate',
            '--n_training', '1',
            '--total_steps', '2',
            '--seed', '3',
            '--num_cpus', '4',
            '--nb_train_steps', '5',
            '--nb_rollout_steps', '6',
            '--nb_eval_episodes', '7',
            '--reward_scale', '8',
            '--render',
            '--render_eval',
            '--verbose', '9',
            '--actor_update_freq', '10',
            '--meta_update_freq', '11',
            '--buffer_size', '12',
            '--batch_size', '13',
            '--actor_lr', '14',
            '--critic_lr', '15',
            '--tau', '16',
            '--gamma', '17',
            '--noise', '18',
            '--target_policy_noise', '19',
            '--target_noise_clip', '20',
            '--layer_norm',
            '--use_huber',
            '--meta_period', '21',
            '--relative_goals',
            '--off_policy_corrections',
            '--use_fingerprints',
            '--centralized_value_functions',
            '--connected_gradients',
            '--cg_weights', '22',
        ])
        hp = get_hyperparameters(args, GoalConditionedPolicy)
        expected_hp = {
            'num_cpus': 4,
            'nb_train_steps': 5,
            'nb_rollout_steps': 6,
            'nb_eval_episodes': 7,
            'reward_scale': 8.0,
            'render': True,
            'render_eval': True,
            'verbose': 9,
            'actor_update_freq': 10,
            'meta_update_freq': 11,
            '_init_setup_model': True,
            'policy_kwargs': {
                'buffer_size': 12,
                'batch_size': 13,
                'actor_lr': 14.0,
                'critic_lr': 15.0,
                'tau': 16.0,
                'gamma': 17.0,
                'noise': 18.0,
                'target_policy_noise': 19.0,
                'target_noise_clip': 20.0,
                'layer_norm': True,
                'use_huber': True,
                'meta_period': 21,
                'relative_goals': True,
                'off_policy_corrections': True,
                'use_fingerprints': True,
                'centralized_value_functions': True,
                'connected_gradients': True,
                'cg_weights': 22,
            }
        }
        self.assertDictEqual(hp, expected_hp)


class TestRewardFns(unittest.TestCase):
    """Test the reward_fns method."""

    def test_negative_distance(self):
        a = np.array([1, 2, 10])
        b = np.array([1, 2])
        c = negative_distance(b, b, a, goal_indices=[1, 2])
        self.assertEqual(c, -8.062257748304752)


if __name__ == '__main__':
    unittest.main()
