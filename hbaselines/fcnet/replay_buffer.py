"""Script containing the ReplayBuffer object."""
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
        self._current_idx = 0
        self._next_idx = 0
        self._batch_size = batch_size

        self.obs_t = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.action_t = np.zeros((buffer_size, ac_dim), dtype=np.float32)
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
        self.action_t[self._next_idx, :] = action
        self.reward[self._next_idx] = reward
        self.obs_tp1[self._next_idx, :] = obs_tp1
        self.done[self._next_idx] = done

        # Increment the next index and size terms
        self._current_idx = self._next_idx
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self._size = min(self._size + 1, self._maxsize)

    def _encode_sample(self, idxes, **kwargs):
        """Convert the indices to appropriate samples."""
        return self.obs_t[idxes, :], self.action_t[idxes, :], \
            self.reward[idxes], self.obs_tp1[idxes, :], self.done[idxes]

    def sample(self, **kwargs):
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
        return self._encode_sample(indices, **kwargs)
