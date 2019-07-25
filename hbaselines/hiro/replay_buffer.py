"""Script contain the ReplayBuffer object.

This script is adapted largely from: https://github.com/hill-a/stable-baselines
"""
import random
import numpy as np


class ReplayBuffer(object):
    """Experience replay buffer."""

    def __init__(self, size):
        """Instantiate a ring buffer (FIFO).

        Parameters
        ----------
        size : int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        """Return the number of elements stored."""
        return len(self._storage)

    @property
    def storage(self):
        """Return the content of the replay buffer.

        Of the form: [(np.ndarray, float, float, np.ndarray, bool)]
        """
        return self._storage

    @property
    def buffer_size(self):
        """Return the (float) max capacity of the buffer."""
        return self._maxsize

    def can_sample(self, n_samples):
        """Check if n_samples samples can be sampled from the buffer.

        Parameters
        ----------
        n_samples : int
            number of requested samples

        Returns
        -------
        bool
            True if enough sample exist, False otherwise
        """
        return len(self) >= n_samples

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
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)

        return np.array(obses_t), np.array(actions), np.array(rewards), \
            np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size, **_kwargs):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size : int
            How many transitions to sample.

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
        indices = [random.randint(0, len(self._storage) - 1)
                   for _ in range(batch_size)]
        return self._encode_sample(indices)


class HierReplayBuffer(ReplayBuffer):
    """Hierarchical variant of ReplayBuffer."""

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
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data

        # Increment the index of the oldest element.
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        """Return a sample from the replay buffer based on indices.

        Parameters
        ----------
        idxes : list of int
            list of random indices

        Returns
        -------
        list of (numpy.ndarray, numpy.ndarray)
            the previous and next manager observations for each meta period
        list of numpy.ndarray
            the meta action (goal) for each meta period
        list of float  FIXME: maybe numpy.ndarray
            the meta reward for each meta period
        list of list of numpy.ndarray  FIXME: maybe list of numpy.ndarray
            all observations for the worker for each meta period
        list of list of numpy.ndarray  FIXME: maybe list of numpy.ndarray
            all actions for the worker for each meta period
        list of list of float  FIXME: maybe list of numpy.ndarray
            all rewards for the worker for each meta period
        list of list of float  FIXME: maybe list of numpy.ndarray
            all done masks for the worker for each meta period. The last done
            mask corresponds to the done mask of the manager
        """
        meta_obs0_all = []
        meta_obs1_all = []
        meta_act_all = []
        meta_rew_all = []
        meta_done_all = []
        worker_obs0_all = []
        worker_obs1_all = []
        worker_act_all = []
        worker_rew_all = []
        worker_done_all = []

        for i in idxes:
            # Extract the elements of the sample.
            meta_obs, meta_action, meta_reward, worker_obses, worker_actions, \
                worker_rewards, worker_dones = self._storage[i]

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
            worker_done = worker_dones[indx_val]

            # Add the new sample to the list of returned samples.
            meta_obs0_all.append(np.array(meta_obs0, copy=False))
            meta_obs1_all.append(np.array(meta_obs1, copy=False))
            meta_act_all.append(np.array(meta_action, copy=False))
            meta_rew_all.append(np.array(meta_reward, copy=False))
            meta_done_all.append(np.array(meta_done, copy=False))
            worker_obs0_all.append(np.array(worker_obs0, copy=False))
            worker_obs1_all.append(np.array(worker_obs1, copy=False))
            worker_act_all.append(np.array(worker_action, copy=False))
            worker_rew_all.append(np.array(worker_reward, copy=False))
            worker_done_all.append(np.array(worker_done, copy=False))

        return np.array(meta_obs0_all), \
            np.array(meta_obs1_all), \
            np.array(meta_act_all), \
            np.array(meta_rew_all), \
            np.array(meta_done_all), \
            np.array(worker_obs0_all), \
            np.array(worker_obs1_all), \
            np.array(worker_act_all), \
            np.array(worker_rew_all), \
            np.array(worker_done_all)
