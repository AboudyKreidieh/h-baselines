"""Contains tests for the model abstractions and different models."""
import unittest
import os
import shutil
import ray

from hbaselines.common.train import parse_options
from experiments.fcnet_baseline import main as fcnet_baseline
from experiments.hiro_baseline import main as hiro_baseline

ray.init(num_cpus=1, redirect_output=True)


class TestExperimentRunnerScripts(unittest.TestCase):
    """Tests the runner scripts in the experiments folder."""

    def test_fcent_baseline(self):
        # Run the script; verify it executes without failure.
        args = parse_options('', '', args=["MountainCarContinuous-v0",
                                           "--n_cpus", "1", "--steps", "1000"])
        fcnet_baseline(args, 'data/fcnet')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(os.getcwd(), "data/fcnet/MountainCarContinuous-v0")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_hiro_baseline(self):
        # Run the script; verify it executes without failure.
        args = parse_options('', '', args=["MountainCarContinuous-v0",
                                           "--n_cpus", "1", "--steps", "1000"])
        hiro_baseline(args, 'data/goal-directed')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(
                os.getcwd(),
                "data/goal-directed/MountainCarContinuous-v0")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))


if __name__ == '__main__':
    unittest.main()
