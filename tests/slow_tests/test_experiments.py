"""Contains tests for the model abstractions and different models."""
import unittest

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

        # Check that the hyperparameter files contained the correct values.

        # Clear anything that was generated.

        # os._exit(1)
        return

    # def test_hiro_baseline(self):
    #     # Run the script; verify it executes without failure.
    #     hiro_baseline(
    #         args=["MountainCarContinuous-v0", "--n_cpus", "1",
    #               "--steps", "1000"]
    #     )
    #
    #     # Check that the folders were generated.
    #
    #     # Check that the hyperparameter files contained the correct values.
    #
    #     # Clear anything that was generated.


if __name__ == '__main__':
    unittest.main()
