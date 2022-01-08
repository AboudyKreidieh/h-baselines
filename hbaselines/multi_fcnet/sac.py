"""Pending deprecation file.

To view the actual content, go to: hbaselines/multiagent/sac.py
"""
from hbaselines.utils.misc import deprecated
from hbaselines.multiagent.sac import MultiFeedForwardPolicy as MultiSAC


@deprecated('hbaselines.multi_fcnet.sac', 'hbaselines.multiagent.sac')
class MultiFeedForwardPolicy(MultiSAC):
    """See parent class."""

    pass
