import numpy as np
from stable_baselines.ddpg.memory import Memory


class GenericMemory(Memory):
    """FIXME

    """

    def __init__(self, limit, action_shape, observation_shape):
        """See parent class."""
        super(GenericMemory, self).__init__(
            limit, action_shape, observation_shape)
        self.trace_length = 1

    def append(self,
               obs0,
               action,
               reward,
               obs1,
               terminal1,
               training=True,
               **kwargs):
        """See parent class.

        Slight modification to include kwargs.
        """
        return super(GenericMemory, self).append(
            obs0, action, reward, obs1, terminal1, training)


class RecurrentMemory(Memory):
    """Recurrent variant of replay buffer.

    Samples in this buffer are stored at a per-episode basis instead of a
    per-time step basis. In addition, samples are extracted as sub-episodes of
    length **trace_length**.
    """

    def __init__(self,
                 limit,
                 action_shape,
                 observation_shape,
                 trace_length=8):
        """The replay buffer object.

        Parameters
        ----------
        limit : int
            the max number of transitions to store
        action_shape : tuple
            the action shape
        observation_shape : tuple
            the observation shape
        trace_length : int, optional
            number of time steps within each sample
        """
        super(RecurrentMemory, self).__init__(
            limit, action_shape, observation_shape)

        self.ob_shape = observation_shape
        self.ac_shape = action_shape

        self.limit = limit
        self.trace_length = trace_length

        self.observations0 = [[]]
        self.actions = [[]]
        self.rewards = [[]]
        self.terminals1 = [[]]
        self.observations1 = [[]]

    def sample(self, batch_size):
        """Sample a random batch from the buffer.

        Batches consist of samples of episodes of length self.trace_length.

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

    def append(self,
               obs0,
               action,
               reward,
               obs1,
               terminal1,
               training=True,
               **kwargs):
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


class HierarchicalRecurrentMemory(RecurrentMemory):
    """FIXME

    """

    def __init__(self, limit, action_shape, observation_shape, **kwargs):
        """Initialize the replay buffer object.

        Parameters
        ----------
        limit : int
            the max number of transitions to store
        action_shape : tuple
            the action shape
        observation_shape : tuple
            the observation shape
        """
        super(HierarchicalRecurrentMemory, self).__init__(
            limit, action_shape, observation_shape, **kwargs)

        self.limit = limit

        self.manager = {
            'observations0': [[]],
            'actions': [[]],
            'rewards': [[]],
            'terminals1': [[]],
            'observations1': [[]],
        }

        self.worker = {
            'observations0': [[]],
            'actions': [[]],
            'rewards': [[]],
            'terminals1': [[]],
            'observations1': [[]],
        }

    def sample(self, batch_size):
        """FIXME

        :param batch_size:
        :return:
        """
        trace_length = self.trace_length

        # draw different episodes
        batch_idxs = np.random.randint(low=0, high=self.nb_entries_manager-1,
                                       size=batch_size)

        # from each episode, draw a series of mini-episodes of length
        obs0_batch = []
        obs1_batch = []
        action_batch = []
        reward_batch = []
        terminal1_batch = []
        for idx in batch_idxs:
            # get the starting time step of the next trace sample
            eps_length = len(self.manager['observations0'][idx])
            point = np.random.randint(0, eps_length + 1 - trace_length)
            # collect the next sample
            obs0_batch.append(
                self.manager['observations0'][idx][point:point+trace_length])
            obs1_batch.append(
                self.manager['observations1'][idx][point:point+trace_length])
            action_batch.append(
                self.manager['actions'][idx][point:point+trace_length])
            reward_batch.append(
                self.manager['rewards'][idx][point:point+trace_length])
            terminal1_batch.append(
                self.manager['terminals1'][idx][point:point+trace_length])

        result_manager = {
            'obs0': np.concatenate(obs0_batch),
            'obs1': np.concatenate(obs1_batch),
            'rewards': np.concatenate(reward_batch).reshape(
                [batch_size * trace_length, 1]),
            'actions': np.concatenate(action_batch),
            'terminals1': np.concatenate(terminal1_batch).reshape(
                [batch_size * trace_length, 1]),
        }

        # draw different episodes
        batch_idxs = np.random.randint(low=0, high=self.nb_entries_worker-1,
                                       size=batch_size)

        # from each episode, draw a series of mini-episodes of length
        obs0_batch = []
        obs1_batch = []
        action_batch = []
        reward_batch = []
        terminal1_batch = []
        for idx in batch_idxs:
            # get the starting time step of the next trace sample
            eps_length = len(self.worker['observations0'][idx])
            point = np.random.randint(0, eps_length + 1 - trace_length)
            # collect the next sample
            obs0_batch.append(
                self.worker['observations0'][idx][point:point+trace_length])
            obs1_batch.append(
                self.worker['observations1'][idx][point:point+trace_length])
            action_batch.append(
                self.worker['actions'][idx][point:point+trace_length])
            reward_batch.append(
                self.worker['rewards'][idx][point:point+trace_length])
            terminal1_batch.append(
                self.worker['terminals1'][idx][point:point+trace_length])

        result_worker = {
            'obs0': np.concatenate(obs0_batch),
            'obs1': np.concatenate(obs1_batch),
            'rewards': np.concatenate(reward_batch).reshape(
                [batch_size * trace_length, 1]),
            'actions': np.concatenate(action_batch),
            'terminals1': np.concatenate(terminal1_batch).reshape(
                [batch_size * trace_length, 1]),
        }

        return result_manager, result_worker

    def append(self,
               obs0,
               action,
               reward,
               obs1,
               terminal1,
               training=True,
               **kwargs):
        """FIXME

        :param obs0:
        :param action:
        :param reward:
        :param obs1:
        :param terminal1:
        :param training:
        :return:
        """
        if not training:
            return

        # separate the manager and worker rewards
        obs0_manager, obs0_worker = obs0[:self.ob_shape[0]], obs0.copy()
        goal, action = action[self.ac_shape[0]:], action[:self.ac_shape[0]]
        reward_manager, reward_worker = reward
        obs1_manager, obs1_worker = obs1[:self.ob_shape[0]], obs1.copy()

        # append the list of samples from the worker
        self.worker['observations0'][-1].append(obs0_worker)
        self.worker['actions'][-1].append(action)
        self.worker['rewards'][-1].append(reward_worker)
        self.worker['observations1'][-1].append(obs1_worker)
        self.worker['terminals1'][-1].append(terminal1)

        if terminal1:
            # if the number of elements exceeds the batch size, remove the
            # first element
            if self.nb_entries_worker == self.limit:
                del self.worker['observations0'][0]
                del self.worker['actions'][0]
                del self.worker['rewards'][0]
                del self.worker['observations1'][0]
                del self.worker['terminals1'][0]

            # check if the most recent batch is too small, and if so, delete
            if len(self.worker['observations0'][-1]) <= self.trace_length:
                del self.worker['observations0'][-1]
                del self.worker['actions'][-1]
                del self.worker['rewards'][-1]
                del self.worker['observations1'][-1]
                del self.worker['terminals1'][-1]

            # if this a termination stage, create a new list item to fill with
            # rollout experience
            self.worker['observations0'].append([])
            self.worker['actions'].append([])
            self.worker['rewards'].append([])
            self.worker['observations1'].append([])
            self.worker['terminals1'].append([])

            # if the number of elements exceeds the batch size, remove the
            # first element
            if self.nb_entries_manager == self.limit:
                del self.manager['observations0'][0]
                del self.manager['actions'][0]
                del self.manager['rewards'][0]
                del self.manager['observations1'][0]
                del self.manager['terminals1'][0]

            # check if the most recent batch is too small, and if so,
            # delete
            if len(self.manager['observations0'][-1]) <= self.trace_length:
                del self.manager['observations0'][-1]
                del self.manager['actions'][-1]
                del self.manager['rewards'][-1]
                del self.manager['observations1'][-1]
                del self.manager['terminals1'][-1]

            # if this a termination stage, create a new list item to fill
            # with rollout experience
            self.manager['observations0'].append([])
            self.manager['actions'].append([])
            self.manager['rewards'].append([])
            self.manager['observations1'].append([])
            self.manager['terminals1'].append([])

        # append the list of samples from the manager
        if kwargs['apply_manager']:
            self.manager['observations0'][-1].append(obs0_manager)
            self.manager['actions'][-1].append(goal)
            self.manager['rewards'][-1].append(reward_manager)
            if len(self.manager['observations0'][-1]) > 1:
                # add the next observation only after the first observation has
                # already been added in the past
                self.manager['observations1'][-1].append(obs1_manager)
            self.manager['terminals1'][-1].append(0)
        elif not terminal1:
            self.manager['rewards'][-1][-1] += reward_manager
        else:
            # Add the last observation at the next observation for the manager.
            # This should complete the list
            self.manager['observations1'][-2].append(obs0_manager)
            self.manager['rewards'][-2][-1] += reward_manager
            self.manager['terminals1'][-2][-1] = 1

    @property
    def nb_entries_manager(self):
        return len(self.manager['observations0'])

    @property
    def nb_entries_worker(self):
        return len(self.worker['observations0'])

    @property
    def nb_entries(self):
        return min(self.nb_entries_worker, self.nb_entries_manager)
