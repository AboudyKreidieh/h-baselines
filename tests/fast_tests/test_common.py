"""Contains tests for the model abstractions and different models."""
import unittest
from hbaselines.common.train import parse_options, get_hyperparameters
from hbaselines.hiro.algorithm import GoalDirectedPolicy
from hbaselines.hiro.algorithm import FEEDFORWARD_POLICY_KWARGS
from hbaselines.hiro.algorithm import GOAL_DIRECTED_POLICY_KWARGS


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
            'sims_per_step': 1,
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
            'connected_gradients': False
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
            '--sims_per_step', '5',
            '--nb_train_steps', '6',
            '--nb_rollout_steps', '7',
            '--nb_eval_episodes', '8',
            '--reward_scale', '9',
            '--render',
            '--render_eval',
            '--verbose', '10',
            '--actor_update_freq', '11',
            '--meta_update_freq', '12',
            '--buffer_size', '13',
            '--batch_size', '14',
            '--actor_lr', '15',
            '--critic_lr', '16',
            '--tau', '17',
            '--gamma', '18',
            '--noise', '19',
            '--target_policy_noise', '20',
            '--target_noise_clip', '21',
            '--layer_norm',
            '--use_huber',
            '--meta_period', '22',
            '--relative_goals',
            '--off_policy_corrections',
            '--use_fingerprints',
            '--centralized_value_functions',
            '--connected_gradients',
        ])
        hp = get_hyperparameters(args, GoalDirectedPolicy)
        expected_hp = {
            'num_cpus': 4,
            'sims_per_step': 5,
            'nb_train_steps': 6,
            'nb_rollout_steps': 7,
            'nb_eval_episodes': 8,
            'reward_scale': 9.0,
            'render': True,
            'render_eval': True,
            'verbose': 10,
            'actor_update_freq': 11,
            'meta_update_freq': 12,
            '_init_setup_model': True,
            'policy_kwargs': {
                'buffer_size': 13,
                'batch_size': 14,
                'actor_lr': 15.0,
                'critic_lr': 16.0,
                'tau': 17.0,
                'gamma': 18.0,
                'noise': 19.0,
                'target_policy_noise': 20.0,
                'target_noise_clip': 21.0,
                'layer_norm': True,
                'use_huber': True,
                'meta_period': 22,
                'relative_goals': True,
                'off_policy_corrections': True,
                'use_fingerprints': True,
                'centralized_value_functions': True,
                'connected_gradients': True
            }
        }
        self.assertDictEqual(hp, expected_hp)


class TestStats(unittest.TestCase):
    """A simple test to get Travis running."""

    def test_reduce_var(self):
        pass

    def test_reduce_std(self):
        pass


if __name__ == '__main__':
    unittest.main()
