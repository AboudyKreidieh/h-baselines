import numpy as np
from stable_baselines.ddpg.memory import Memory


class RecurrentMemory(Memory):

    def __init__(self, limit, action_shape, observation_shape):
        """The replay buffer object.

        Parameters
        ----------
        limit : int
            the max number of transitions to store
        action_shape : tuple
            the action shape
        observation_shape : tuple
            the observation shape
        """
        super(RecurrentMemory, self).__init__(
            limit, action_shape, observation_shape)

        self.limit = limit

        self.observations0 = [[]]
        self.actions = [[]]
        self.rewards = [[]]
        self.terminals1 = [[]]
        self.observations1 = [[]]

        # TODO: add to inputs
        self.trace_length = 8

    def sample(self, batch_size):
        """Sample a random batch from the buffer.

        Parameters
        ----------
        batch_size : int
            the number of element to sample for the batch

        Returns
        -------
        dict
            the sampled batch
        """
        trace_length = self.trace_length

        # draw different episodes
        batch_idxs = np.random.randint(low=0, high=self.nb_entries-1,
                                       size=batch_size)

        # from each episode, draw a series of mini-episodes of length
        obs0_batch = []
        obs1_batch = []
        action_batch = []
        reward_batch = []
        terminal1_batch = []
        for idx in batch_idxs:
            # get the starting time step of the next trace sample
            eps_length = len(self.observations0[idx])
            point = np.random.randint(0, eps_length + 1 - trace_length)
            # collect the next sample
            obs0_batch.append(
                self.observations0[idx][point:point+trace_length])
            obs1_batch.append(
                self.observations1[idx][point:point+trace_length])
            action_batch.append(
                self.actions[idx][point:point+trace_length])
            reward_batch.append(
                self.rewards[idx][point:point+trace_length])
            terminal1_batch.append(
                self.terminals1[idx][point:point+trace_length])

        result = {
            'obs0': np.concatenate(obs0_batch),
            'obs1': np.concatenate(obs1_batch),
            'rewards': np.concatenate(reward_batch).reshape(
                [batch_size * trace_length, 1]),
            'actions': np.concatenate(action_batch),
            'terminals1': np.concatenate(terminal1_batch).reshape(
                [batch_size * trace_length, 1]),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        """Append a transition to the buffer.

        This methods will store samples from the same episode in a single list.

        Parameters
        ----------
        obs0 : list of float or list of int
            the last observation
        action : list of float
            the action
        reward : float
            the reward
        obs1 : list of float or list of int
            the current observation
        terminal1 : bool
            is the episode done
        training : bool
            is the RL model training or not
        """
        if not training:
            return

        self.observations0[-1].append(obs0)
        self.actions[-1].append(action)
        self.rewards[-1].append(reward)
        self.observations1[-1].append(obs1)
        self.terminals1[-1].append(terminal1)

        if terminal1:
            # if the number of elements exceeds the batch size, remove the
            # first element
            if self.nb_entries == self.limit:
                del self.observations0[0]
                del self.actions[0]
                del self.rewards[0]
                del self.observations1[0]
                del self.terminals1[0]

            # check if the most recent batch is too small, and if so, delete
            if len(self.observations0[-1]) <= self.trace_length:
                print("oops:", self.observations0[-1])
                del self.observations0[-1]
                del self.actions[-1]
                del self.rewards[-1]
                del self.observations1[-1]
                del self.terminals1[-1]

            # if this a termination stage, create a new list item to fill with
            # rollout experience
            self.observations0.append([])
            self.actions.append([])
            self.rewards.append([])
            self.observations1.append([])
            self.terminals1.append([])

    @property
    def nb_entries(self):
        return len(self.observations0)
