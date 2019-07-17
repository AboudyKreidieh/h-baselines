import pytest
from hbaselines.hiro.replay_buffer import ReplayBuffer


@pytest.fixture
def replay_buffer():
    """Returns a replay buffer of size x"""
    return ReplayBuffer(2)


def test_replay_buffer_size(replay_buffer):
    """
        Simple test to assert buffer size of
        created ReplayBuffer instance.

    Parameters
    ----------
    replay_buffer: Object
        ReplayBuffer
    """
    assert replay_buffer.buffer_size == 2


@pytest.mark.parametrize("obs_t, action, reward, obs_tp1, done",
                         [(1, 2, 3, 4, 5.1)])
def test_replay_buffer_storage(replay_buffer,
                               obs_t,
                               action,
                               reward,
                               obs_tp1,
                               done):
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
    replay_buffer.add(obs_t, action, reward, obs_tp1, done)

    assert replay_buffer.storage == [(1, 2, 3, 4, 5.1)]


@pytest.mark.parametrize("obs_t, action, reward, obs_tp1, done",
                         [(1, 2, 3, 4, 5.1)])
def test_storage_length(replay_buffer, obs_t, action, reward, obs_tp1, done):
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
    replay_buffer.add(obs_t, action, reward, obs_tp1, done)
    assert replay_buffer.__len__() == 1


@pytest.mark.parametrize("obs_t, action, reward, obs_tp1, done",
                         [(1, 2, 3, 4, 5.1)])
def test_can_sample(replay_buffer, obs_t, action, reward, obs_tp1, done):
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
    assert replay_buffer.can_sample(1) is False

    # added something
    replay_buffer.add(obs_t, action, reward, obs_tp1, done)

    # sample that one thing added
    assert replay_buffer.can_sample(1) is True

    # get greedy and sample beyond
    assert replay_buffer.can_sample(2) is False


@pytest.mark.parametrize("obs_t, action, reward, obs_tp1, done,"
                         "obs_t_2, action_2, reward_2, obs_tp1_2, done_2",
                         [(1, 2, 3, 4, 5.1,
                           1, 2, 3, 4, 5.1)])
def test_is_full(replay_buffer,
                 obs_t,
                 action,
                 reward,
                 obs_tp1,
                 done,
                 obs_t_2,
                 action_2,
                 reward_2,
                 obs_tp1_2,
                 done_2):
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
    replay_buffer.add(obs_t, action, reward, obs_tp1, done)
    replay_buffer.add(obs_t_2, action_2, reward_2, obs_tp1_2, done_2)

    assert replay_buffer.is_full() is True
