"""
##############################################
# Feudal Network batch data processor script #
##############################################
"""


import numpy as np
from collections import namedtuple


def cosine_similarity(u, v):
    """
    Cosine similarity function as is defined here:
    https://reference.wolfram.com/language/ref/CosineDistance.html

    Parameters
    ----------
    u : vector
        value vector
    v : vector
        value vector
    """
    return np.dot(np.squeeze(u),
                  np.squeeze(v)) / (np.linalg.norm(u) * np.linalg.norm(v))


Batch = namedtuple("Batch",
                   ["obs",
                    "a",
                    "returns",
                    "s_diff",
                    "ri",
                    "gsum",
                    "features"])


class FeudalBatch(object):
    """
    Class to be used for batch data efficient processing in the Feudal Network

    """

    def __init__(self):
        """
        Instantiate a feudal batch data processor

        """
        self.obs = []
        self.a = []
        self.returns = []
        self.s_diff = []
        self.ri = []
        self.gsum = []
        self.features = None

    def add(self, obs, a, returns, s_diff, ri, gsum, features):
        """
        Create a Gym environment by passing environment id.

        Parameters
        ----------
        obs : object
            new observations to be added to feudal network
        a : object
            new actions to be added to feudal network
        returns : object
            new returns to be added to feudal network
        s_diff : object
            BLANK
        ri : object
            BLANK
        gsum : object
            BLANK
        features : object
            BLANK
        """
        self.obs += [obs]
        self.a += [a]
        self.returns += [returns]
        self.s_diff += [s_diff]
        self.ri += [ri]
        self.gsum += [gsum]
        if not self.features:
            self.features = features

    def get_batch(self):
        """
        Function used to retrieve batch of data

        """
        batch_obs = np.asarray(self.obs)
        batch_a = np.asarray(self.a)
        batch_r = np.asarray(self.returns)
        batch_sd = np.squeeze(np.asarray(self.s_diff))
        batch_ri = np.asarray(self.ri)
        batch_gs = np.asarray(self.gsum)
        return Batch(batch_obs,
                     batch_a,
                     batch_r,
                     batch_sd,
                     batch_ri,
                     batch_gs,
                     self.features)


class FeudalBatchProcessor(object):
    """
    Class to processor batch of data in Feudal Network


    This class adapts the batch of PolicyOptimizer to a batch usable by
    the FeudalPolicy.
    """
    def __init__(self, c):
        """
        Instantiate a feudal batch data processor

        Parameters
        ----------
        c : object
            BLANK
        """
        self.c = c
        self.last_terminal = True

    def _extend(self, batch):
        """
        Private function to help extend the Feudal Network.

        Parameters
        ----------
        batch : object
            batch object containing states,
            goals, observations, actions, and features
        """
        if self.last_terminal:
            self.last_terminal = False
            self.s = [batch.s[0] for _ in range(self.c)]
            self.g = [batch.g[0] for _ in range(self.c)]
            # prepend with dummy values so indexing is the same
            self.obs = [None for _ in range(self.c)]
            self.a = [None for _ in range(self.c)]
            self.returns = [None for _ in range(self.c)]
            self.features = [None for _ in range(self.c)]

        # extend with the actual values
        self.obs.extend(batch.obs)
        self.a.extend(batch.a)
        self.returns.extend(batch.returns)
        self.s.extend(batch.s)
        self.g.extend(batch.g)
        self.features.extend(batch.features)

        # if this is a terminal batch, then append the final s and g c times
        # note that both this and the above case can occur at the same time
        if batch.terminal:
            self.s.extend([batch.s[-1] for _ in range(self.c)])
            self.g.extend([batch.g[-1] for _ in range(self.c)])

    def process_batch(self, batch):
        """
        Function to process the batch of data in the Feudal Network


        Converts a normal batch into one used by the FeudalPolicy update.
        FeudalPolicy requires a batch of the form:
        c previous timesteps - batch size timesteps - c future timesteps
        This class handles the tracking the leading and
        following timesteps over
        time. Additionally, it also computes values across timesteps from the
        batch to provide to FeudalPolicy.

        Parameters
        ----------
        batch : object
            batch object containing states,
            goals, observations, actions, and features
        """
        # extend with current batch
        self._extend(batch)

        # unpack and compute bounds
        length = len(self.obs)
        c = self.c

        # normally we cannot compute samples for the last c elements, but
        # in the terminal case, we halluciante values where necessary
        end = length if batch.terminal else length - c

        # collect samples to return in a FeudalBatch
        feudal_batch = FeudalBatch()
        for t in range(c, end):

            # state difference
            s_diff = self.s[t + c] - self.s[t]

            # intrinsic reward
            ri = 0
            # note that this for loop considers s and g values
            # 1 timestep to c timesteps (inclusively) ago
            for i in range(1, c + 1):
                ri_s_diff = self.s[t] - self.s[t - i]
                if np.linalg.norm(ri_s_diff) != 0:
                    ri += cosine_similarity(ri_s_diff, self.g[t - i])
            ri /= c

            # sum of g values used to derive w, input to the linear transform
            gsum = np.zeros_like(self.g[t - c])
            for i in range(t - c, t + 1):
                gsum += self.g[i]

            # add to the batch
            feudal_batch.add(self.obs[t],
                             self.a[t],
                             self.returns[t],
                             s_diff,
                             ri,
                             gsum,
                             self.features[t])

        # in the terminal case, set reset flag
        if batch.terminal:
            self.last_terminal = True
        # in the general case, forget all but the last 2 * c elements
        # reason being that the first c of those we have already computed
        # a batch for, and the second c need those first c
        else:
            twoc = 2 * self.c
            self.obs = self.obs[-twoc:]
            self.a = self.a[-twoc:]
            self.returns = self.returns[-twoc:]
            self.s = self.s[-twoc:]
            self.g = self.g[-twoc:]
            self.features = self.features[-twoc:]

        return feudal_batch.get_batch()
