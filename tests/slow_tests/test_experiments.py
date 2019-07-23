"""Contains tests for the model abstractions and different models."""
import unittest
import os
import shutil

from experiments.fcnet_baseline import main as fcnet_baselines


class TestExperimentRunnerScripts(unittest.TestCase):
    """Tests the runner scripts in the experiments folder."""

    def test_fcent_baseline(self):
        # Run the script; verify it executes without failure.
        fcnet_baselines(
            args=["MountainCarContinuous-v0", "--n_cpus", "1",
                  "--steps", "1000"]
        )

        # Check that the folders were generated.
        self.assertTrue(os.path.isdir(
            os.path.join(os.getcwd(), "data/fcnet/MountainCarContinuous-v0")))

        # Clear anything that was generated.
        shutil.rmtree(os.path.join(os.getcwd(), "data"))

    # def test_hiro_baseline(self):
    #     # Run the script; verify it executes without failure.
    #     hiro_baseline(
    #         args=["MountainCarContinuous-v0", "--n_cpus", "1",
    #               "--steps", "1000"]
    #     )
    #
    #     # Check that the folders were generated.
    #
    #     # Clear anything that was generated.


if __name__ == '__main__':
    unittest.main()
