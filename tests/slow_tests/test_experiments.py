"""Contains tests for the model abstractions and different models."""
import unittest
import os
import shutil
import ray
import numpy as np

from hbaselines.utils.train import parse_options as parse_train_options
from experiments.plot import parse_options as parse_plot_options
from experiments.plot import import_results
from experiments.plot import plot_fig
from experiments.run_fcnet import main as run_fcnet
from experiments.run_hrl import main as run_hrl
from experiments.run_multi_fcnet import main as run_multi_fcnet
from experiments.run_multi_hrl import main as run_multi_hrl

os.environ["TEST_FLAG"] = "True"


class TestPlot(unittest.TestCase):
    """Tests for the experiments/plot.py script."""

    def test_parse_options(self):
        """Test the parse_options method.

        This is done for the following cases:

        1. default case
        2. custom case
        """
        # test case 1
        args = parse_plot_options(["AntMaze"])
        expected_args = {
            'folders': ['AntMaze'],
            'names': None,
            'out': 'out.png',
            'show': False,
            'use_eval': False,
            'x': 'total/steps',
            'xlabel': None,
            'y': 'rollout/return_history',
            'ylabel': None,
        }
        self.assertDictEqual(vars(args), expected_args)

        # test case 2
        args = parse_plot_options([
            '1', '2', '3',
            '--names', '4', '5', '6',
            '--out', '7',
            '--show',
            '--use_eval',
            '--x', '8',
            '--xlabel', '9',
            '--y', '10',
            '--ylabel', '11',
        ])
        expected_args = {
            'folders': ['1', '2', '3'],
            'names': ['4', '5', '6'],
            'out': '7',
            'show': True,
            'use_eval': True,
            'x': '8',
            'xlabel': '9',
            'y': '10',
            'ylabel': '11',
        }
        self.assertDictEqual(vars(args), expected_args)

    def test_import_results(self):
        """Test the import_results method.

        This is done for the following cases:

        1. use_eval = False
        2. use_eval = True
        """
        # test case 1
        x, y_mean, y_std = import_results(
            folders=[os.path.join(os.path.dirname(
                os.path.realpath(__file__)), 'supplementary/HalfCheetah-v2')],
            x='total/steps',
            y='rollout/return_history',
            use_eval=False,
        )

        np.testing.assert_almost_equal(x, [2000, 4000, 6000, 8000, 10000])
        np.testing.assert_almost_equal(
            y_mean, [[-452.9650969, -438.1423037, -446.7406732, -438.3988754,
                      -394.1437327]])
        np.testing.assert_almost_equal(y_std, [[0, 0, 0, 0, 0]])

        # Test plot_fig to make sure it generates something.
        plot_fig(
            mean=y_mean,
            std=y_std,
            steps=x,
            y_lim=None,
            name="output.png",
            legend=None,
            show=False,
            xlabel='x',
            ylabel='y',
            save=True,
        )
        self.assertTrue(os.path.isfile('output.png'))
        os.remove('output.png')

        # test case 2
        x, y_mean, y_std = import_results(
            folders=[os.path.join(os.path.dirname(
                os.path.realpath(__file__)), 'supplementary/HalfCheetah-v2')],
            x='total_step',
            y='success_rate',
            use_eval=True,
        )

        np.testing.assert_almost_equal(x, [10000])
        np.testing.assert_almost_equal(y_mean, [[[0.]]])
        np.testing.assert_almost_equal(y_std, [[[0.]]])


class TestExperimentRunnerScripts(unittest.TestCase):
    """Tests the runner scripts in the experiments folder."""

    def test_run_fcent_td3(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "MountainCarContinuous-v0",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
                "--num_envs", "2",  # to test RaySampler
                "--nb_rollout_steps", "2",
            ],
            multiagent=False,
            hierarchical=False,
        )
        run_fcnet(args, 'data/fcnet')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(os.getcwd(), "data/fcnet/MountainCarContinuous-v0")))

        # Clear anything that was generated.
        ray.shutdown()
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_fcent_sac(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "MountainCarContinuous-v0",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
                "--alg", "SAC"
            ],
            multiagent=False,
            hierarchical=False,
        )
        run_fcnet(args, 'data/fcnet')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(os.getcwd(), "data/fcnet/MountainCarContinuous-v0")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_fcent_failure(self):
        # Run the script; verify it fails.
        args = parse_train_options(
            '', '',
            args=[
                "MountainCarContinuous-v0",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
                "--alg", "woops"
            ],
            multiagent=False,
            hierarchical=False,
        )
        self.assertRaises(ValueError, run_fcnet,
                          args=args, base_dir='data/fcnet')

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_hrl_td3(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "MountainCarContinuous-v0",
                "--initial_exploration_steps", "1",
                "--batch_size", "32",
                "--meta_period", "5",
                "--total_steps", "500",
                "--log_interval", "500",
            ],
            multiagent=False,
            hierarchical=True,
        )
        run_hrl(args, 'data/goal-conditioned')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(
                os.getcwd(),
                "data/goal-conditioned/MountainCarContinuous-v0")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_hrl_sac(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "MountainCarContinuous-v0",
                "--initial_exploration_steps", "1",
                "--batch_size", "32",
                "--meta_period", "5",
                "--total_steps", "500",
                "--log_interval", "500",
                "--alg", "SAC"
            ],
            multiagent=False,
            hierarchical=True,
        )
        run_hrl(args, 'data/goal-conditioned')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(
                os.getcwd(),
                "data/goal-conditioned/MountainCarContinuous-v0")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_hrl_failure(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "MountainCarContinuous-v0",
                "--initial_exploration_steps", "1",
                "--meta_period", "5",
                "--total_steps", "500",
                "--log_interval", "500",
                "--alg", "woops"
            ],
            multiagent=False,
            hierarchical=True,
        )

        self.assertRaises(ValueError, run_hrl,
                          args=args, base_dir='data/goal-conditioned')

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_fcnet_td3_independent(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "multiagent-ring_small",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
            ],
            multiagent=True,
            hierarchical=False,
        )
        run_multi_fcnet(args, 'data/multi-fcnet')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(
                os.getcwd(),
                "data/multi-fcnet/multiagent-ring_small")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_fcnet_sac_independent(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "multiagent-ring_small",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
                "--alg", "SAC"
            ],
            multiagent=True,
            hierarchical=False,
        )
        run_multi_fcnet(args, 'data/multi-fcnet')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(
                os.getcwd(),
                "data/multi-fcnet/multiagent-ring_small")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_fcnet_failure_independent(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "multiagent-ring_small",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
                "--alg", "woops"
            ],
            multiagent=True,
            hierarchical=False,
        )

        self.assertRaises(ValueError, run_multi_fcnet,
                          args=args, base_dir='data/multi-fcnet')

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_fcnet_td3_shared(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "multiagent-ring_small",
                "--shared",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
            ],
            multiagent=True,
            hierarchical=False,
        )
        run_multi_fcnet(args, 'data/multi-fcnet')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(
                os.getcwd(),
                "data/multi-fcnet/multiagent-ring_small")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_fcnet_sac_shared(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "multiagent-ring_small",
                "--shared",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
                "--alg", "SAC"
            ],
            multiagent=True,
            hierarchical=False,
        )
        run_multi_fcnet(args, 'data/multi-fcnet')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(
                os.getcwd(),
                "data/multi-fcnet/multiagent-ring_small")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_fcnet_failure_shared(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "MountainCarContinuous-v0",
                "--shared",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
                "--alg", "woops"
            ],
            multiagent=True,
            hierarchical=False,
        )

        self.assertRaises(ValueError, run_multi_fcnet,
                          args=args, base_dir='data/multi-fcnet')

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_fcnet_td3_maddpg_independent(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "multiagent-ring_small",
                "--maddpg",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
            ],
            multiagent=True,
            hierarchical=False,
        )
        run_multi_fcnet(args, 'data/multi-fcnet')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(
                os.getcwd(),
                "data/multi-fcnet/multiagent-ring_small")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_fcnet_sac_maddpg_independent(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "multiagent-ring_small",
                "--maddpg",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
                "--alg", "SAC"
            ],
            multiagent=True,
            hierarchical=False,
        )
        run_multi_fcnet(args, 'data/multi-fcnet')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(
                os.getcwd(),
                "data/multi-fcnet/multiagent-ring_small")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_fcnet_failure_maddpg_independent(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "multiagent-ring_small",
                "--maddpg",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
                "--alg", "woops"
            ],
            multiagent=True,
            hierarchical=False,
        )

        self.assertRaises(ValueError, run_multi_fcnet,
                          args=args, base_dir='data/multi-fcnet')

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_fcnet_td3_maddpg_shared(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "multiagent-ring_small",
                "--shared",
                "--maddpg",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
            ],
            multiagent=True,
            hierarchical=False,
        )
        run_multi_fcnet(args, 'data/multi-fcnet')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(
                os.getcwd(),
                "data/multi-fcnet/multiagent-ring_small")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_fcnet_sac_maddpg_shared(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "multiagent-ring_small",
                "--shared",
                "--maddpg",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
                "--alg", "SAC"
            ],
            multiagent=True,
            hierarchical=False,
        )
        run_multi_fcnet(args, 'data/multi-fcnet')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(
                os.getcwd(),
                "data/multi-fcnet/multiagent-ring_small")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_fcnet_failure_maddpg_shared(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "MountainCarContinuous-v0",
                "--shared",
                "--maddpg",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
                "--alg", "woops"
            ],
            multiagent=True,
            hierarchical=False,
        )

        self.assertRaises(ValueError, run_multi_fcnet,
                          args=args, base_dir='data/multi-fcnet')

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_hrl_td3_independent(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "multiagent-ring_small",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
            ],
            multiagent=True,
            hierarchical=True,
        )
        run_multi_hrl(args, 'data/multi-goal-conditioned')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(
                os.getcwd(),
                "data/multi-goal-conditioned/multiagent-ring_small")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_hrl_sac_independent(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "multiagent-ring_small",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
                "--alg", "SAC"
            ],
            multiagent=True,
            hierarchical=True,
        )
        run_multi_hrl(args, 'data/multi-goal-conditioned')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(
                os.getcwd(),
                "data/multi-goal-conditioned/multiagent-ring_small")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_hrl_failure_independent(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "multiagent-ring_small",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
                "--alg", "woops"
            ],
            multiagent=True,
            hierarchical=True,
        )

        self.assertRaises(ValueError, run_multi_hrl,
                          args=args, base_dir='data/multi-goal-conditioned')

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_hrl_td3_shared(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "multiagent-ring_small",
                "--shared",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
            ],
            multiagent=True,
            hierarchical=True,
        )
        run_multi_hrl(args, 'data/multi-goal-conditioned')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(
                os.getcwd(),
                "data/multi-goal-conditioned/multiagent-ring_small")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_hrl_sac_shared(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "multiagent-ring_small",
                "--shared",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
                "--alg", "SAC"
            ],
            multiagent=True,
            hierarchical=True,
        )
        run_multi_hrl(args, 'data/multi-goal-conditioned')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(
                os.getcwd(),
                "data/multi-goal-conditioned/multiagent-ring_small")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_multi_hrl_failure_shared(self):
        # Run the script; verify it executes without failure.
        args = parse_train_options(
            '', '',
            args=[
                "MountainCarContinuous-v0",
                "--shared",
                "--initial_exploration_steps", "1",
                "--total_steps", "500",
                "--log_interval", "500",
                "--alg", "woops"
            ],
            multiagent=True,
            hierarchical=True,
        )

        self.assertRaises(ValueError, run_multi_hrl,
                          args=args, base_dir='data/multi-goal-conditioned')

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))


if __name__ == '__main__':
    unittest.main()
