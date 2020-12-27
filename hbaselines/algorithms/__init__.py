"""Init script for the algorithms submodule."""
from hbaselines.algorithms.off_policy import OffPolicyRLAlgorithm
from hbaselines.algorithms.dagger import DAggerAlgorithm
from hbaselines.algorithms.rl_algorithm import RLAlgorithm

__all__ = [
    "OffPolicyRLAlgorithm",
    "RLAlgorithm",
    "DAggerAlgorithm",
]
