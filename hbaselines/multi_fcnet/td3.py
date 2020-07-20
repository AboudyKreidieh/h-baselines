"""Pending deprecation file.

To view the actual content, go to: hbaselines/multiagent/td3.py
"""
from hbaselines.utils.misc import deprecated
from hbaselines.multiagent.td3 import FeedForwardPolicy as MultiTD3


@deprecated('hbaselines.multi_fcnet.td3.FeedForwardPolicy',
            'hbaselines.multiagent.td3.FeedForwardPolicy')
class FeedForwardPolicy(MultiTD3):
    """See parent class."""

    pass
