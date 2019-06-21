import tensorflow as tf
import numpy as np
import gym


class HRLAlgorithm(object):
    """

    """

    def __init__(self, env):
        """

        :param env:
        """
        # create a tf session
        self.sess = tf.Session()

        # store the environment of interest in a variable
        self.env = env

        # placeholder for the states
        self.s_t_ph = tf.placeholder(
            tf.float32,
            shape=[None, env.observation_space.shape[0]])

        # placeholder for the actions
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.a_t_ph = tf.placeholder(
                tf.float32,
                shape=[None, env.action_space.n]
            )
        elif isinstance(env.action_space, gym.spaces.Box):
            self.a_t_ph = tf.placeholder(
                tf.float32,
                shape=[None, env.action_space.shape[0]]
            )
        else:
            raise TypeError("Actions of type {} no supported.".format(
                type(env.action_space)))

        # create the policy model that is employed in the training procedure
        self.model = self.create_model()

        # initialize all tf variables
        self.sess.run(tf.global_variables_initializer())

    ###########################################################################
    #                        Policy-specific components                       #
    ###########################################################################

    def create_model(self):
        """

        :return:
        """
        raise NotImplementedError

    def compute_action(self, state):
        """

        :param state:
        :return:
        """
        self.model.get_action(state)

    def save_checkpoint(self):
        """Save the model parameters to a ckpt file.

        The name of the checkpoint file is based on the name of the training
        experiments, the initial time of running, and the iteration number.
        """
        raise NotImplementedError

    def load_checkpoint(self, filename):
        """Restore the model parameters from a ckpt file.

        Parameters
        ----------
        filename : str
            location of the checkpoint
        """
        raise NotImplementedError

    ###########################################################################
    #                      Training-specific components                       #
    ###########################################################################

    def train(self):
        """

        :return:
        """
        raise NotImplementedError

    def define_updates(self, **kwargs):
        """

        :return:
        """
        raise NotImplementedError

    def call_updates(self, samples, **kwargs):
        """

        :return:
        """
        raise NotImplementedError

    ###########################################################################
    #                     Environment-specific components                     #
    ###########################################################################

    def rollout(self):
        """Collect samples from one rollout of the policy.

        Returns
        -------
        dict
            dictionary conta.ining trajectory information for the rollout,
            specifically containing keys for "state", "action",
            "next_state", "reward", and "done"
        """
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []

        # start a new rollout by resetting the environment and
        # collecting the initial state
        state = self.env.reset()

        steps = 0
        while True:
            steps += 1

            # compute the action given the state
            action = self.compute_action(np.array([state]))
            action = action[0]

            # advance the environment once and collect the next state,
            # reward, done, and info parameters from the environment
            next_state, reward, done, info = self.env.step(action)

            # add to the samples list
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)

            state = next_state

            # if the environment returns a True for the done parameter,
            # end the rollout before the time horizon is met
            if done or steps > self.env.env_params.horizon:
                break

        # create the output trajectory
        trajectory = {"state": np.array(states, dtype=np.float32),
                      "reward": np.array(rewards, dtype=np.float32),
                      "action": np.array(actions, dtype=np.float32),
                      "next_state": np.array(next_states, dtype=np.float32),
                      "done": np.array(dones, dtype=np.float32)}

        return trajectory

    ###########################################################################
    #                       Logging-specific components                       #
    ###########################################################################

    def get_statistics(self, verbose):
        """

        Parameters
        ----------
        verbose : bool
            If True, then the statistics are also printed after every training
            iteration.

        :return:
        """
        raise NotImplementedError

    def log_statistics(self):
        """

        :return:
        """
        raise NotImplementedError
