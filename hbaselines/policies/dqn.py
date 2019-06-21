from stable_baselines.deepq.policies import DQNPolicy


class FullyConnectedPolicy(DQNPolicy):
    """

    """

    def step(self, obs, state=None, mask=None, deterministic=True):
        """Return the q_values for a single step.

        Parameters
        ----------
        obs : np.ndarray of (float or int)
            The current observation of the environment
        state : np.ndarray of float
            The last states (used in recurrent policies)
        mask : np.ndarray of float
            The last masks (used in recurrent policies)
        deterministic : bool
            Whether or not to return deterministic actions.

        Returns
        -------
        np.ndarray of int
            actions
        np.ndarray of float
            q_values
        np.ndarray of float
            states
        """
        raise NotImplementedError

    def proba_step(self, obs, state=None, mask=None):
        """Return the action probability for a single step.

        obs : np.ndarray of (float or int)
            The current observation of the environment
        state : np.ndarray of float
            The last states (used in recurrent policies)
        mask : np.ndarray of float
                The last masks (used in recurrent policies)

        Returns
        -------
        np.ndarray of float
            the action probability
        """
        raise NotImplementedError


# TODO
class LSTMPolicy(DQNPolicy):
    pass


# TODO
class FeudalPolicy(DQNPolicy):
    pass


# TODO
class HIROPolicy(DQNPolicy):  # TODO: rename
    pass
