"""Contains tests for the model abstractions and different models."""
import unittest
import tensorflow as tf
import numpy as np
from gym.spaces import Box

from hbaselines.utils.train import parse_options, get_hyperparameters
from hbaselines.utils.reward_fns import negative_distance
from hbaselines.utils.misc import get_manager_ac_space, get_state_indices
from hbaselines.utils.tf_util import gaussian_likelihood
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy
from hbaselines.multi_fcnet.td3 import MultiFeedForwardPolicy
from hbaselines.algorithms.off_policy import TD3_PARAMS
from hbaselines.algorithms.off_policy import SAC_PARAMS
from hbaselines.algorithms.off_policy import FEEDFORWARD_PARAMS
from hbaselines.algorithms.off_policy import GOAL_CONDITIONED_PARAMS


class TestTrain(unittest.TestCase):
    """A simple test to get Travis running."""

    def test_parse_options(self):
        # Test the default case.
        args = parse_options("", "", args=["AntMaze"])
        expected_args = {
            'env_name': 'AntMaze',
            'alg': 'TD3',
            'evaluate': False,
            'n_training': 1,
            'total_steps': 1000000,
            'seed': 1,
            'log_interval': 2000,
            'eval_interval': 50000,
            'save_interval': 50000,
            'initial_exploration_steps': 10000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'noise': TD3_PARAMS['noise'],
            'target_policy_noise': TD3_PARAMS['target_policy_noise'],
            'target_noise_clip': TD3_PARAMS['target_noise_clip'],
            'target_entropy': SAC_PARAMS['target_entropy'],
            'buffer_size': FEEDFORWARD_PARAMS['buffer_size'],
            'batch_size': FEEDFORWARD_PARAMS['batch_size'],
            'actor_lr': FEEDFORWARD_PARAMS['actor_lr'],
            'critic_lr': FEEDFORWARD_PARAMS['critic_lr'],
            'tau': FEEDFORWARD_PARAMS['tau'],
            'gamma': FEEDFORWARD_PARAMS['gamma'],
            'layer_norm': False,
            'use_huber': False,
            'meta_period': GOAL_CONDITIONED_PARAMS['meta_period'],
            'worker_reward_scale':
                GOAL_CONDITIONED_PARAMS['worker_reward_scale'],
            'relative_goals': False,
            'off_policy_corrections': False,
            'hindsight': False,
            'subgoal_testing_rate':
                GOAL_CONDITIONED_PARAMS['subgoal_testing_rate'],
            'use_fingerprints': False,
            'centralized_value_functions': False,
            'connected_gradients': False,
            'cg_weights': GOAL_CONDITIONED_PARAMS['cg_weights'],
            'shared': False,
            'maddpg': False,
        }
        self.assertDictEqual(vars(args), expected_args)

        # Test custom cases.
        args = parse_options("", "", args=[
            "AntMaze",
            '--evaluate',
            '--n_training', '1',
            '--total_steps', '2',
            '--seed', '3',
            '--log_interval', '4',
            '--eval_interval', '5',
            '--save_interval', '6',
            '--nb_train_steps', '7',
            '--nb_rollout_steps', '8',
            '--nb_eval_episodes', '9',
            '--reward_scale', '10',
            '--render',
            '--render_eval',
            '--verbose', '11',
            '--actor_update_freq', '12',
            '--meta_update_freq', '13',
            '--buffer_size', '14',
            '--batch_size', '15',
            '--actor_lr', '16',
            '--critic_lr', '17',
            '--tau', '18',
            '--gamma', '19',
            '--noise', '20',
            '--target_policy_noise', '21',
            '--target_noise_clip', '22',
            '--layer_norm',
            '--use_huber',
            '--meta_period', '23',
            '--worker_reward_scale', '24',
            '--relative_goals',
            '--off_policy_corrections',
            '--hindsight',
            '--subgoal_testing_rate', '25',
            '--use_fingerprints',
            '--centralized_value_functions',
            '--connected_gradients',
            '--cg_weights', '26',
            '--shared',
            '--maddpg',
        ])
        hp = get_hyperparameters(args, GoalConditionedPolicy)
        expected_hp = {
            'nb_train_steps': 7,
            'nb_rollout_steps': 8,
            'nb_eval_episodes': 9,
            'reward_scale': 10.0,
            'render': True,
            'render_eval': True,
            'verbose': 11,
            'actor_update_freq': 12,
            'meta_update_freq': 13,
            '_init_setup_model': True,
            'policy_kwargs': {
                'buffer_size': 14,
                'batch_size': 15,
                'actor_lr': 16.0,
                'critic_lr': 17.0,
                'tau': 18.0,
                'gamma': 19.0,
                'noise': 20.0,
                'target_policy_noise': 21.0,
                'target_noise_clip': 22.0,
                'layer_norm': True,
                'use_huber': True,
                'meta_period': 23,
                'worker_reward_scale': 24.0,
                'relative_goals': True,
                'off_policy_corrections': True,
                'hindsight': True,
                'subgoal_testing_rate': 25.0,
                'use_fingerprints': True,
                'centralized_value_functions': True,
                'connected_gradients': True,
                'cg_weights': 26,
            }
        }
        self.assertDictEqual(hp, expected_hp)
        self.assertEqual(args.log_interval, 4)
        self.assertEqual(args.eval_interval, 5)

        hp = get_hyperparameters(args, MultiFeedForwardPolicy)
        expected_hp = {
            'nb_train_steps': 7,
            'nb_rollout_steps': 8,
            'nb_eval_episodes': 9,
            'actor_update_freq': 12,
            'meta_update_freq': 13,
            'reward_scale': 10.0,
            'render': True,
            'render_eval': True,
            'verbose': 11,
            '_init_setup_model': True,
            'policy_kwargs': {
                'buffer_size': 14,
                'batch_size': 15,
                'actor_lr': 16.0,
                'critic_lr': 17.0,
                'tau': 18.0,
                'gamma': 19.0,
                'layer_norm': True,
                'use_huber': True,
                'noise': 20.0,
                'target_policy_noise': 21.0,
                'target_noise_clip': 22.0,
                'shared': True,
                'maddpg': True,
            }
        }
        self.assertDictEqual(hp, expected_hp)
        self.assertEqual(args.log_interval, 4)
        self.assertEqual(args.eval_interval, 5)


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
            ob_space=Box(-1, 1, shape=(2,)),
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
            [13]
        )

        # test for figureeight1
        self.assertListEqual(
            get_state_indices(env_name="figureeight1", **params),
            [1, 3, 5, 7, 9, 11, 13]
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


class TestTFUtil(unittest.TestCase):

    def setUp(self):
        self.sess = tf.compat.v1.Session()

    def tearDown(self):
        self.sess.close()

    def test_gaussian_likelihood(self):
        """Check the functionality of the gaussian_likelihood() method."""
        input_ = tf.constant([[0, 1, 2]], dtype=tf.float32)
        mu_ = tf.constant([[0, 0, 0]], dtype=tf.float32)
        log_std = tf.constant([[-4, -3, -2]], dtype=tf.float32)
        val = gaussian_likelihood(input_, mu_, log_std)
        expected = -304.65784

        self.assertAlmostEqual(self.sess.run(val)[0], expected, places=4)

    def test_apply_squashing(self):
        """Check the functionality of the apply_squashing() method."""
        pass  # TODO


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
