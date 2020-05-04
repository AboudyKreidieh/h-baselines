"""Init script for the algorithms submodule."""
from hbaselines.algorithms.off_policy import OffPolicyRLAlgorithm
from hbaselines.algorithms.dagger import DAggerAlgorithm

__all__ = [
    "OffPolicyRLAlgorithm",
    "DAggerAlgorithm"
]
