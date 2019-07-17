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
        return len(self._storage)

    @property
    def storage(self):
        """Return the content of the replay buffer.

        Of the form: [(np.ndarray, float, float, np.ndarray, bool)]
        """
        return self._storage

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
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
        """Add a new transition to the buffer

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

    def new_add(self,
                obs_t,
                goal_t,
                action_t,
                reward_t,
                done_,
                h_t,
                goal_updated):
        """
        Add a new transition to the buffer

        Parameters
        ----------
        obs_t: Any
            the last observation
        action_t: array_like
            the action
        goal_t: array_like
            the goal
        reward_t: float
            the reward of the transition
        done_: float
            is the episode done
        h_t: array_like
            the next goal
        goal_updated: int boolean
            is the goal updated or no
        """
        data = (obs_t,
                goal_t,
                action_t,
                reward_t,
                done_,
                h_t,
                goal_updated)

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

    def _new_sample_encoder(self, idxes):
        """
        Returns a sample from the replay buffer based on indices

        Parameters
        ----------
        idxes: list
            list of random indices

        Returns
        -------
        sample from replay buffer
        (s_t, g_t, a_t, r, s_t+1, done, h(s,g,s'))
        """
        obses_t, goals, actions, rewards = [], [], [], []
        obses_tp1, done, h_t = [], [], []

        for i in idxes:
            data = self._storage[i]
            obs_t, goals_t, actions_t, rewards_t, obs_tp1, d, h = data
            obses_t.append(np.array(obs_t, copy=False))
            goals.append(np.array(goals_t, copy=False))
            actions.append(np.array(actions_t, copy=False))
            rewards.append(np.array(rewards_t, copy=False))
            done.append(d)
            h_t.append(np.array(h, copy=False))

        return np.array(obses_t), np.array(goals), np.array(actions), \
            np.array(rewards), np.array(obses_tp1), np.array(done), \
            np.array(h_t)

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

    def replace(self, d, goal_updated):
        """
        Call this function to replace the old goal
        with the new one when off-policy correction
        takes place.

        Prerequisite
        ------------
        pass the data
        tuple in the form of a list to allow
        it to be mutable.

        Postrequisite
        -------------
        convert list of data back to tuple

        * Note: function assumes that there exists no
                duplication of an experience tuple in
                the replay buffer. Other wise it will
                update the first occurrence tuple.

        Parameters
        ----------
        d: list
            experience tuple as a list
        goal_updated: Any
            new updated goal
        """
        assert type(d) is list

        # find the location of the experience
        location_of_experience = self._storage.index(tuple(d))

        # update the experience
        """
            First index is following old add method.
            New index should be == 3
        """
        d[1][0] = goal_updated

        # update it in the replay buffer
        self._storage[location_of_experience] = tuple(d)
