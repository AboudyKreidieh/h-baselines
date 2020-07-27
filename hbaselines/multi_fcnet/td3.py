"""Pending deprecation file.

To view the actual content, go to: hbaselines/multiagent/td3.py
"""
from hbaselines.utils.misc import deprecated
from hbaselines.multiagent.td3 import MultiFeedForwardPolicy as MultiTD3


@deprecated('hbaselines.multi_fcnet.td3', 'hbaselines.multiagent.td3')
class MultiFeedForwardPolicy(MultiTD3):
    """See parent class."""

    pass
