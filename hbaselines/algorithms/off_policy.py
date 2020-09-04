"""Pending deprecation file.

To view the actual content, go to: hbaselines/algorithms/rl_algorithm.py
"""
from hbaselines.utils.misc import deprecated
from hbaselines.algorithms.rl_algorithm import RLAlgorithm


@deprecated('hbaselines.algorithms.off_policy',
            'hbaselines.algorithms.rl_algorithm')
class OffPolicyRLAlgorithm(RLAlgorithm):
    """See parent class."""

    pass
