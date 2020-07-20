"""Pending deprecation file.

To view the actual content, go to: hbaselines/multiagent/sac.py
"""
from hbaselines.utils.misc import deprecated
from hbaselines.multiagent.sac import FeedForwardPolicy as MultiSAC


@deprecated('hbaselines.multi_fcnet.sac.FeedForwardPolicy',
            'hbaselines.multiagent.sac.FeedForwardPolicy')
class FeedForwardPolicy(MultiSAC):
    """See parent class."""

    pass
