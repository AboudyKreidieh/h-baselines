"""Contains tests for the model abstractions and different models."""
import unittest
import numpy as np
from hbaselines.utils.train import parse_options, get_hyperparameters
from hbaselines.utils.reward_fns import negative_distance
from hbaselines.utils.misc import get_manager_ac_space, get_state_indices
from hbaselines.goal_conditioned.algorithm import GoalConditionedPolicy
from hbaselines.goal_conditioned.algorithm import FEEDFORWARD_PARAMS
from hbaselines.goal_conditioned.algorithm import GOAL_CONDITIONED_PARAMS


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
            'buffer_size': FEEDFORWARD_PARAMS['buffer_size'],
            'batch_size': FEEDFORWARD_PARAMS['batch_size'],
            'actor_lr': FEEDFORWARD_PARAMS['actor_lr'],
            'critic_lr': FEEDFORWARD_PARAMS['critic_lr'],
            'tau': FEEDFORWARD_PARAMS['tau'],
            'gamma': FEEDFORWARD_PARAMS['gamma'],
            'noise': FEEDFORWARD_PARAMS['noise'],
            'target_policy_noise': FEEDFORWARD_PARAMS['target_policy_noise'],
            'target_noise_clip': FEEDFORWARD_PARAMS['target_noise_clip'],
            'layer_norm': False,
            'use_huber': False,
            'meta_period': GOAL_CONDITIONED_PARAMS['meta_period'],
            'worker_reward_scale':
                GOAL_CONDITIONED_PARAMS['worker_reward_scale'],
            'relative_goals': False,
            'off_policy_corrections': False,
            'use_fingerprints': False,
            'centralized_value_functions': False,
            'connected_gradients': False,
            'cg_weights': GOAL_CONDITIONED_PARAMS['cg_weights'],
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
            '--worker_reward_scale', '22',
            '--relative_goals',
            '--off_policy_corrections',
            '--use_fingerprints',
            '--centralized_value_functions',
            '--connected_gradients',
            '--cg_weights', '23',
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
                'worker_reward_scale': 22.0,
                'relative_goals': True,
                'off_policy_corrections': True,
                'use_fingerprints': True,
                'centralized_value_functions': True,
                'connected_gradients': True,
                'cg_weights': 23,
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


class TestMisc(unittest.TestCase):
    """Test the the miscellaneous utility methods."""

    def test_manager_ac_space(self):
        # non-relevant parameters for most tests
        params = dict(
            ob_space=None,
            relative_goals=False,
            use_fingerprints=False,
            fingerprint_dim=1,
        )
        rel_params = params.copy()
        rel_params.update({"relative_goals": True})

        # test for AntMaze
        ac_space = get_manager_ac_space(env_name="AntMaze", **params)
        test_space(
            ac_space,
            expected_min=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3,
                                   -0.5, -0.3, -0.5, -0.3, -0.5, -0.3]),
            expected_max=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3,
                                   0.5, 0.3, 0.5, 0.3]),
            expected_size=15,
        )

        # test for AntGather
        ac_space = get_manager_ac_space(env_name="AntGather", **params)
        test_space(
            ac_space,
            expected_min=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3,
                                   -0.5, -0.3, -0.5, -0.3, -0.5, -0.3]),
            expected_max=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3,
                                   0.5, 0.3, 0.5, 0.3]),
            expected_size=15,
        )

        # test for AntPush
        ac_space = get_manager_ac_space(env_name="AntPush", **params)
        test_space(
            ac_space,
            expected_min=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3,
                                   -0.5, -0.3, -0.5, -0.3, -0.5, -0.3]),
            expected_max=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3,
                                   0.5, 0.3, 0.5, 0.3]),
            expected_size=15,
        )

        # test for AntFall
        ac_space = get_manager_ac_space(env_name="AntFall", **params)
        test_space(
            ac_space,
            expected_min=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3,
                                   -0.5, -0.3, -0.5, -0.3, -0.5, -0.3]),
            expected_max=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3,
                                   0.5, 0.3, 0.5, 0.3]),
            expected_size=15,
        )

        # test for UR5
        ac_space = get_manager_ac_space(env_name="UR5", **params)
        test_space(
            ac_space,
            expected_min=np.array([-2*np.pi, -2*np.pi, -2*np.pi, -4, -4, -4]),
            expected_max=np.array([2*np.pi, 2*np.pi, 2*np.pi, 4, 4, 4]),
            expected_size=6,
        )

        # test for Pendulum
        ac_space = get_manager_ac_space(env_name="Pendulum", **params)
        test_space(
            ac_space,
            expected_min=np.array([-np.pi, -15]),
            expected_max=np.array([np.pi, 15]),
            expected_size=2,
        )

        # test for ring0
        ac_space = get_manager_ac_space(env_name="ring0", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(1)]),
            expected_max=np.array([1 for _ in range(1)]),
            expected_size=1,
        )

        ac_space = get_manager_ac_space(env_name="ring0", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-0.5 for _ in range(1)]),
            expected_max=np.array([0.5 for _ in range(1)]),
            expected_size=1,
        )

        # test for ring1
        ac_space = get_manager_ac_space(env_name="ring1", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(1)]),
            expected_max=np.array([1 for _ in range(1)]),
            expected_size=1,
        )

        ac_space = get_manager_ac_space(env_name="ring1", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-0.5 for _ in range(1)]),
            expected_max=np.array([0.5 for _ in range(1)]),
            expected_size=1,
        )

        # test for merge0
        ac_space = get_manager_ac_space(env_name="merge0", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(5)]),
            expected_max=np.array([1 for _ in range(5)]),
            expected_size=5,
        )

        ac_space = get_manager_ac_space(env_name="merge0", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-0.5 for _ in range(5)]),
            expected_max=np.array([0.5 for _ in range(5)]),
            expected_size=5,
        )

        # test for merge1
        ac_space = get_manager_ac_space(env_name="merge1", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(13)]),
            expected_max=np.array([1 for _ in range(13)]),
            expected_size=13,
        )

        ac_space = get_manager_ac_space(env_name="merge1", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-0.5 for _ in range(13)]),
            expected_max=np.array([0.5 for _ in range(13)]),
            expected_size=13,
        )

        # test for merge2
        ac_space = get_manager_ac_space(env_name="merge2", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(17)]),
            expected_max=np.array([1 for _ in range(17)]),
            expected_size=17,
        )

        ac_space = get_manager_ac_space(env_name="merge2", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-0.5 for _ in range(17)]),
            expected_max=np.array([0.5 for _ in range(17)]),
            expected_size=17,
        )

        # test for figureeight0
        ac_space = get_manager_ac_space(env_name="figureeight0", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(1)]),
            expected_max=np.array([1 for _ in range(1)]),
            expected_size=1,
        )

        ac_space = get_manager_ac_space(env_name="figureeight0", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-0.5 for _ in range(1)]),
            expected_max=np.array([0.5 for _ in range(1)]),
            expected_size=1,
        )

        # test for figureeight1
        ac_space = get_manager_ac_space(env_name="figureeight1", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(7)]),
            expected_max=np.array([1 for _ in range(7)]),
            expected_size=7,
        )

        ac_space = get_manager_ac_space(env_name="figureeight1", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-0.5 for _ in range(7)]),
            expected_max=np.array([0.5 for _ in range(7)]),
            expected_size=7,
        )

        # test for figureeight2
        ac_space = get_manager_ac_space(env_name="figureeight2", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(14)]),
            expected_max=np.array([1 for _ in range(14)]),
            expected_size=14,
        )

        ac_space = get_manager_ac_space(env_name="figureeight2", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-0.5 for _ in range(14)]),
            expected_max=np.array([0.5 for _ in range(14)]),
            expected_size=14,
        )

        # test for grid0
        ac_space = get_manager_ac_space(env_name="grid0", **params)
        del ac_space  # TODO

        # test for grid1
        ac_space = get_manager_ac_space(env_name="grid1", **params)
        del ac_space  # TODO

        # test for bottleneck0
        ac_space = get_manager_ac_space(env_name="bottleneck0", **params)
        del ac_space  # TODO

        # test for bottleneck1
        ac_space = get_manager_ac_space(env_name="bottleneck1", **params)
        del ac_space  # TODO

        # test for bottleneck2
        ac_space = get_manager_ac_space(env_name="bottleneck2", **params)
        del ac_space  # TODO

    def test_state_indices(self):
        # non-relevant parameters for most tests
        params = dict(
            ob_space=None,
            use_fingerprints=False,
            fingerprint_dim=1,
        )

        # test for AntMaze
        self.assertListEqual(
            get_state_indices(env_name="AntMaze", **params),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        )

        # test for AntGather
        self.assertListEqual(
            get_state_indices(env_name="AntGather", **params),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        )

        # test for AntPush
        self.assertListEqual(
            get_state_indices(env_name="AntPush", **params),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        )

        # test for AntFall
        self.assertListEqual(
            get_state_indices(env_name="AntFall", **params),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        )

        # test for UR5
        self.assertIsNone(get_state_indices(env_name="UR5", **params))

        # test for Pendulum
        self.assertListEqual(
            get_state_indices(env_name="Pendulum", **params),
            [0, 2]
        )

        # test for ring0
        self.assertListEqual(
            get_state_indices(env_name="ring0", **params),
            [0]
        )

        # test for ring1
        self.assertListEqual(
            get_state_indices(env_name="ring1", **params),
            [0]
        )

        # test for merge0
        self.assertListEqual(
            get_state_indices(env_name="merge0", **params),
            [0, 5, 10, 15, 20]
        )

        # test for merge1
        self.assertListEqual(
            get_state_indices(env_name="merge1", **params),
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        )

        # test for merge2
        self.assertListEqual(
            get_state_indices(env_name="merge2", **params),
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        )

        # test for figureeight0
        self.assertListEqual(
            get_state_indices(env_name="figureeight0", **params),
            [13]  # FIXME: correct?
        )

        # test for figureeight1
        self.assertListEqual(
            get_state_indices(env_name="figureeight1", **params),
            [1, 3, 5, 7, 9, 11, 13]  # FIXME: correct?
        )

        # test for figureeight2
        self.assertListEqual(
            get_state_indices(env_name="figureeight2", **params),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        )

        # test for grid0
        state_indices = get_state_indices(env_name="grid0", **params)
        del state_indices  # TODO

        # test for grid1
        state_indices = get_state_indices(env_name="grid1", **params)
        del state_indices  # TODO

        # test for bottleneck0
        state_indices = get_state_indices(env_name="bottleneck0", **params)
        del state_indices  # TODO

        # test for bottleneck1
        state_indices = get_state_indices(env_name="bottleneck1", **params)
        del state_indices  # TODO

        # test for bottleneck2
        state_indices = get_state_indices(env_name="bottleneck2", **params)
        del state_indices  # TODO


def test_space(gym_space, expected_size, expected_min, expected_max):
    """Test the shape and bounds of an action or observation space.

    Parameters
    ----------
    gym_space : gym.spaces.Box
        gym space object to be tested
    expected_size : int
        expected size
    expected_min : float or array_like
        expected minimum value(s)
    expected_max : float or array_like
        expected maximum value(s)
    """
    assert gym_space.shape[0] == expected_size, \
        "{}, {}".format(gym_space.shape[0], expected_size)
    np.testing.assert_almost_equal(gym_space.high, expected_max, decimal=4)
    np.testing.assert_almost_equal(gym_space.low, expected_min, decimal=4)


if __name__ == '__main__':
    unittest.main()
