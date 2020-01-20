"""Script containing the HierReplayBuffer object."""
import random
import numpy as np

from hbaselines.fcnet.replay_buffer import ReplayBuffer


class HierReplayBuffer(ReplayBuffer):
    """Hierarchical variant of ReplayBuffer."""

    def __init__(self,
                 buffer_size,
                 batch_size,
                 meta_period,
                 meta_obs_dim,
                 meta_ac_dim,
                 worker_obs_dim,
                 worker_ac_dim):
        """Instantiate the hierarchical replay buffer.

        Parameters
        ----------
        buffer_size : int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        batch_size : int
            number of elements that are to be returned as a batch
        meta_obs_dim : int
            number of elements in the Manager observations
        meta_ac_dim : int
            number of elements in the Manager actions
        worker_obs_dim : int
            number of elements in the Worker observations
        worker_ac_dim : int
            number of elements in the Worker actions
        """
        super(HierReplayBuffer, self).__init__(
            buffer_size, batch_size, worker_obs_dim, worker_ac_dim)

        self._meta_period = meta_period
        self._worker_ob_dim = worker_obs_dim
        self._worker_ac_dim = worker_ac_dim

        # Used to store buffer data.
        self._storage = [None for _ in range(buffer_size)]

        # Variables that are used when returning samples
        self.meta_obs0 = np.zeros(
            (batch_size, meta_obs_dim), dtype=np.float32)
        self.meta_obs1 = np.zeros(
            (batch_size, meta_obs_dim), dtype=np.float32)
        self.meta_act = np.zeros(
            (batch_size, meta_ac_dim), dtype=np.float32)
        self.meta_rew = np.zeros(
            batch_size, dtype=np.float32)
        self.meta_done = np.zeros(
            batch_size, dtype=np.float32)
        self.worker_obs0 = np.zeros(
            (batch_size, worker_obs_dim), dtype=np.float32)
        self.worker_obs1 = np.zeros(
            (batch_size, worker_obs_dim), dtype=np.float32)
        self.worker_act = np.zeros(
            (batch_size, worker_ac_dim), dtype=np.float32)
        self.worker_rew = np.zeros(
            batch_size, dtype=np.float32)
        self.worker_done = np.zeros(
            batch_size, dtype=np.float32)

    def add(self, obs_t, goal_t, action_t, reward_t, done, **kwargs):
        """Add a new transition to the buffer.

        Parameters
        ----------
        obs_t : array_like
            list of all worker observations for a given meta period
        action_t : array_like
            list of all worker actions for a given meta period
        goal_t : array_like
            the meta action
        reward_t : list of float
            list of all worker rewards for a given meta period
        done : list of float or list of bool
            list of done masks
        kwargs : Any
            additional parameters, including:

            * meta_obs_t: a tuple of the manager observation and next
              observation
            * meta_reward_t: the reward of the manager
        """
        # Store the manager samples, then the worker samples.
        data = (kwargs["meta_obs_t"], goal_t, kwargs["meta_reward_t"],
                obs_t, action_t, reward_t, done)

        # Add the element to the list. If the list is already the max size of
        # the replay buffer, then replace the oldest sample with this one.
        self._storage[self._next_idx] = data

        # Increment the next index and size terms
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self._size = min(self._size + 1, self._maxsize)

    def _encode_sample(self, idxes, **kwargs):
        """Return a sample from the replay buffer based on indices.

        Parameters
        ----------
        idxes : list of int
            list of random indices
        kwargs : dict
            additional parameters, including:

            * with_additional (bool): specifies whether to remove additional
              data from the replay buffer sampling procedure. Since only a
              subset of algorithms use additional data, removing it can speedup
              the other algorithms.

        Returns
        -------
        numpy.ndarray
            (batch_size, meta_obs) matrix of meta observations
        numpy.ndarray
            (batch_size, meta_obs) matrix of next meta-period meta observations
        numpy.ndarray
            (batch_size, meta_ac) matrix of meta actions
        numpy.ndarray
            (batch_size,) vector of meta rewards
        numpy.ndarray
            (batch_size,) vector of meta done masks
        numpy.ndarray
            (batch_size, worker_obs) matrix of worker observations
        numpy.ndarray
            (batch_size, worker_obs) matrix of next step worker observations
        numpy.ndarray
            (batch_size, worker_ac) matrix of worker actions
        numpy.ndarray
            (batch_size,) vector of worker rewards
        numpy.ndarray
            (batch_size,) vector of worker done masks
        dict
            additional information; used for features such as the off-policy
            corrections or centralized value functions
        """
        # Do not encode additional information information in samples if it is
        # not needed. Waste of compute resources.
        if kwargs.get("with_additional", True):
            additional = {
                "worker_obses": np.zeros(
                    (self._batch_size, self._worker_ob_dim,
                     self._meta_period + 1),
                    dtype=np.float32),
                "worker_actions": np.zeros(
                    (self._batch_size, self._worker_ac_dim, self._meta_period),
                    dtype=np.float32),
            }
        else:
            additional = None

        for i, indx in enumerate(idxes):
            # Extract the elements of the sample.
            meta_obs, meta_action, meta_reward, worker_obses, worker_actions, \
                worker_rewards, worker_dones = self._storage[indx]

            # Separate the current and next step meta observations.
            meta_obs0, meta_obs1 = meta_obs

            # The meta done value corresponds to the last done value.
            meta_done = worker_dones[-1]

            # Sample one obs0/obs1/action/reward from the list of per-meta-
            # period variables.
            indx_val = random.randint(0, len(worker_obses)-2)
            worker_obs0 = worker_obses[indx_val]
            worker_obs1 = worker_obses[indx_val + 1]
            worker_action = worker_actions[indx_val]
            worker_reward = worker_rewards[indx_val]
            worker_done = 0  # see docstring

            # Add the new sample to the list of returned samples.
            self.meta_obs0[i, :] = np.array(meta_obs0, copy=False)
            self.meta_obs1[i, :] = np.array(meta_obs1, copy=False)
            self.meta_act[i, :] = np.array(meta_action, copy=False)
            self.meta_rew[i] = np.array(meta_reward, copy=False)
            self.meta_done[i] = np.array(meta_done, copy=False)
            self.worker_obs0[i, :] = np.array(worker_obs0, copy=False)
            self.worker_obs1[i, :] = np.array(worker_obs1, copy=False)
            self.worker_act[i, :] = np.array(worker_action, copy=False)
            self.worker_rew[i] = np.array(worker_reward, copy=False)
            self.worker_done[i] = np.array(worker_done, copy=False)

            if kwargs.get("with_additional", True):
                for j in range(len(worker_obses)):
                    additional["worker_obses"][i, :, j] = worker_obses[j]
                for j in range(len(worker_actions)):
                    additional["worker_actions"][i, :, j] = worker_actions[j]

        return self.meta_obs0, \
            self.meta_obs1, \
            self.meta_act, \
            self.meta_rew, \
            self.meta_done, \
            self.worker_obs0, \
            self.worker_obs1, \
            self.worker_act, \
            self.worker_rew, \
            self.worker_done, \
            additional
