import unittest
from hbaselines.hiro.replay_buffer import ReplayBuffer


class TestBuffer(unittest.TestCase):

    def setUp(self):
        self.replay_buffer = ReplayBuffer(2)

        self.obs_t, self.action, self.reward = 1, 2, 3
        self.obs_tp1, self.done = 4, 5.1

    def test_replay_buffer_size(self):
        """
            Simple test to assert buffer size of
            created ReplayBuffer instance.

        Parameters
        ----------
        replay_buffer: Object
            ReplayBuffer
        """
        assert self.replay_buffer.buffer_size == 2

    def test_replay_buffer_storage(self):
        """
        Test the storage taken up in the
        replay buffer (i.e: content of storage list).

        Parameters
        ----------
        replay_buffer: Object
            ReplayBuffer
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
        self.replay_buffer.add(self.obs_t,
                               self.action,
                               self.reward,
                               self.obs_tp1,
                               self.done)

        assert self.replay_buffer.storage == [(1, 2, 3, 4, 5.1)]

    def test_storage_length(self):
        """
        Test the length of the storage list.

        Parameters
        ----------
        replay_buffer: Object
            ReplayBuffer
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
        self.replay_buffer.add(self.obs_t,
                               self.action,
                               self.reward,
                               self.obs_tp1,
                               self.done)

        assert self.replay_buffer.__len__() == 1

    def test_can_sample(self):
        """
        Test whether a replay buffer can be sampled
        depending on the content it holds (or doesn't yet).

        Parameters
        ----------
        replay_buffer: Object
            ReplayBuffer
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
        # have yet to add something
        assert self.replay_buffer.can_sample(1) is False

        # added something
        self.replay_buffer.add(self.obs_t,
                               self.action,
                               self.reward,
                               self.obs_tp1,
                               self.done)

        # sample that one thing added
        assert self.replay_buffer.can_sample(1) is True

        # get greedy and sample beyond
        assert self.replay_buffer.can_sample(2) is False

    def test_is_full(self):
        """
        Test whether the replay buffer is full or not.

        Parameters
        ----------
        replay_buffer: Object
            ReplayBuffer
        obs_t, obs_t_2 : Any
            the last observation
        action, action_2 : array_like
            the action
        reward, reward_2 : float
            the reward of the transition
        obs_tp1, obs_tp1_2 : Any
            the current observation
        done, done_2 : float
            is the episode done
        """
        list_of_experiences = [(1, 2, 3, 4, 5.1),
                               (1, 2, 3, 4, 5.1)]

        for tuples in list_of_experiences:
            self.replay_buffer.add(tuples[0],
                                   tuples[1],
                                   tuples[2],
                                   tuples[3],
                                   tuples[4])

        assert self.replay_buffer.is_full() is True

    def test_sample(self):
        """
        Test whether the data is being sampled correctly.

        Parameters
        ----------
        replay_buffer: Object
            ReplayBuffer
        list_of_experiences: list of Objects
            list of experiences to be sampled from
        """
        sampled_data = []
        list_of_experiences = [(1, 2, 3, 4, 5.1),
                               (1, 2, 3, 4, 5.1)]

        for tuples in list_of_experiences:
            self.replay_buffer.add(tuples[0],
                                   tuples[1],
                                   tuples[2],
                                   tuples[3],
                                   tuples[4])

        tmp_sample = [_ for _ in self.replay_buffer.sample(1)]

        for index in range(5):
            sampled_data.append(tmp_sample[index][0])

        # Because we are sampling randomly, don't know what we get
        assert tuple(sampled_data) in list_of_experiences

    def test_replace(self):
        """
        Test if a goal in a replay experience is being updated.

        Parameters
        ----------
        replay_buffer: Object
            ReplayBuffer
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
        self.replay_buffer.add(self.obs_t,
                               self.action,
                               self.reward,
                               self.obs_tp1,
                               self.done)

        sampled_experience = [_ for _ in self.replay_buffer.sample(1)]

        self.replay_buffer.replace(sampled_experience, 10)

        new_sampled_experience = [_ for _ in self.replay_buffer.sample(1)]

        assert new_sampled_experience[1][0][0] == 10

    def tearDown(self):
        self.replay_buffer = ReplayBuffer(2)


"""
    Up until this point we have been testing using the old
    add function in the replay buffer; consider the above as
    a benchmark if you will. The below tests are going to use
    the new add and sampling functions in the replay buffer.
"""

if __name__ == '__main__':
    unittest.main()
