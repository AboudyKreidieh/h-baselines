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

    def sample(self, batch_size, trace_length=8):
        """Sample a random batch from the buffer.

        Parameters
        ----------
        batch_size : int
            the number of element to sample for the batch
        trace_length : int
            number of time steps to include in each episode element

        Returns
        -------
        dict
            the sampled batch
        """
        # draw different episodes
        batch_idxs = np.random.randint(low=0, high=self.nb_entries,
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
            'obs0': np.asarray(obs0_batch),
            'obs1': np.asarray(obs1_batch),
            'rewards': np.asarray(reward_batch),
            'actions': np.asarray(action_batch),
            'terminals1': np.asarray(terminal1_batch),
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
