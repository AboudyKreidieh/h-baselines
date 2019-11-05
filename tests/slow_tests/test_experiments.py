"""Contains tests for the model abstractions and different models."""
import unittest
import os
import shutil

from hbaselines.utils.train import parse_options
from experiments.run_fcnet import main as run_fcnet
from experiments.run_hrl import main as run_hrl


class TestExperimentRunnerScripts(unittest.TestCase):
    """Tests the runner scripts in the experiments folder."""

    def test_run_fcent(self):
        # Run the script; verify it executes without failure.
        args = parse_options('', '', args=["MountainCarContinuous-v0",
                                           "--n_cpus", "1",
                                           "--total_steps", "2000"])
        run_fcnet(args, 'data/fcnet')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(os.getcwd(), "data/fcnet/MountainCarContinuous-v0")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    def test_run_hrl(self):
        # Run the script; verify it executes without failure.
        args = parse_options('', '', args=["MountainCarContinuous-v0",
                                           "--n_cpus", "1",
                                           "--total_steps", "2000"])
        run_hrl(args, 'data/goal-conditioned')

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(
                os.getcwd(),
                "data/goal-conditioned/MountainCarContinuous-v0")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))


if __name__ == '__main__':
    unittest.main()
