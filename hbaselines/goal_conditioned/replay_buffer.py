"""Script contain the ReplayBuffer object.

This script is adapted largely from: https://github.com/hill-a/stable-baselines
"""
import random
import numpy as np


class ReplayBuffer(object):
    """Experience replay buffer."""

    def __init__(self, buffer_size, batch_size, obs_dim, ac_dim):
        """Instantiate a ring buffer (FIFO).

        Parameters
        ----------
        buffer_size : int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        batch_size : int
            number of elements that are to be returned as a batch
        obs_dim : int
            number of elements in the observations
        ac_dim : int
            number of elements in the actions
        """
        self._maxsize = buffer_size
        self._size = 0
        self._next_idx = 0
        self._batch_size = batch_size

        self.obs_t = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.action = np.zeros((buffer_size, ac_dim), dtype=np.float32)
        self.reward = np.zeros(buffer_size, dtype=np.float32)
        self.obs_tp1 = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.done = np.zeros(buffer_size, dtype=np.float32)

    def __len__(self):
        """Return the number of elements stored."""
        return self._size

    @property
    def buffer_size(self):
        """Return the (float) max capacity of the buffer."""
        return self._maxsize

    def can_sample(self):
        """Check if n_samples samples can be sampled from the buffer.

        Returns
        -------
        bool
            True if enough sample exist, False otherwise
        """
        return len(self) >= self._batch_size

    def is_full(self):
        """Check whether the replay buffer is full or not.

        Returns
        -------
        bool
            True if it is full, False otherwise
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done):
        """Add a new transition to the buffer.

        Parameters
        ----------
        obs_t : Any
            the last observation
        action : array_like
            the action
        reward : float
            the reward of the transition
        obs_tp1 : Any
            the current observation
        done : float
            is the episode done
        """
        self.obs_t[self._next_idx, :] = obs_t
        self.action[self._next_idx, :] = action
        self.reward[self._next_idx] = reward
        self.obs_tp1[self._next_idx, :] = obs_tp1
        self.done[self._next_idx] = done

        # Increment the next index and size terms
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self._size = min(self._size + 1, self._maxsize)

    def _encode_sample(self, idxes):
        """Convert the indices to appropriate samples."""
        return self.obs_t[idxes, :], self.action[idxes, :], \
            self.reward[idxes], self.obs_tp1[idxes, :], self.done[idxes]

    def sample(self, **_kwargs):
        """Sample a batch of experiences.

        Returns
        -------
        np.ndarray
            batch of observations
        numpy float
            batch of actions executed given obs_batch
        numpy float
            rewards received as results of executing act_batch
        np.ndarray
            next set of observations seen after executing act_batch
        numpy bool
            done_mask[i] = 1 if executing act_batch[i] resulted in the end of
            an episode and 0 otherwise.
        """
        indices = np.random.randint(0, self._size, size=self._batch_size)
        return self._encode_sample(indices)


class HierReplayBuffer(ReplayBuffer):
    """Hierarchical variant of ReplayBuffer."""

    def __init__(self,
                 buffer_size,
                 batch_size,
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

    def _encode_sample(self, idxes):
        """Return a sample from the replay buffer based on indices.

        Parameters
        ----------
        idxes : list of int
            list of random indices

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
        """
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

        return self.meta_obs0, \
            self.meta_obs1, \
            self.meta_act, \
            self.meta_rew, \
            self.meta_done, \
            self.worker_obs0, \
            self.worker_obs1, \
            self.worker_act, \
            self.worker_rew, \
            self.worker_done
