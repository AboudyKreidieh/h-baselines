from stable_baselines.ddpg.policies import FeedForwardPolicy, nature_cnn
import tensorflow as tf


class FullyConnectedPolicy(FeedForwardPolicy):
    """Policy object that implements a DDPG-like actor critic, using MLPs.

    sess : tf.Session
        The current TensorFlow session
    ob_space : gym.space.*
        The observation space of the environment
    ac_space : gym.space.*
        The action space of the environment
    n_env : int
        The number of environments to run
    n_steps : int
        The number of steps to run for each environment
    n_batch : int
        The number of batch to run (n_envs * n_steps)
    reuse : bool
        If the policy is reusable or not
    layers : list of int
        The size of the Neural network for the policy (if None, default to
        [64, 64])
    cnn_extractor : function
        the CNN feature extraction
    feature_extraction : str
        The feature extraction type ("cnn" or "mlp")
    layer_norm : bool
        enable layer normalisation
    kwargs : dict
        Extra keyword arguments for the nature CNN feature extraction
    """
    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 n_env,
                 n_steps,
                 n_batch,
                 reuse=False,
                 layers=None,
                 cnn_extractor=nature_cnn,
                 feature_extraction="mlp",
                 layer_norm=False,
                 **kwargs):
        super(FullyConnectedPolicy, self).__init__(
            sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256,
            reuse=reuse, scale=(feature_extraction == "cnn"))

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        """Create an actor object.

        Parameters
        ----------
        obs : tf.Tensor
            The observation placeholder (can be None for default placeholder)
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor

        Returns
        -------
        tf.Tensor
            the output tensor
        """
        if obs is None:
            obs = self.processed_x

        hidden_size = [64, 64]
        nonlinearity = tf.nn.relu

        with tf.variable_scope(scope, reuse=reuse):
            # input to the system
            last_layer = tf.layers.flatten(obs)

            # create the hidden layers
            for i, hidden in enumerate(hidden_size):
                last_layer = tf.layers.dense(inputs=last_layer,
                                             units=hidden,
                                             name='fc_{}'.format(i),
                                             activation=nonlinearity)

            # create the output layer
            self.policy = tf.layers.dense(inputs=last_layer,
                                          units=self.ac_space.shape[0],
                                          name=scope,
                                          activation=None)

        return self.policy

    def make_critic(self, obs=None, action=None, reuse=False, scope="qf"):
        """Create a critic object.

        Parameters
        ----------
        obs : tf.Tensor
            The observation placeholder (can be None for default placeholder)
        action : tf.Tensor
            The action placeholder (can be None for default placeholder)
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the critic

        Returns
        -------
        tf.Tensor
            the output tensor
        """
        if obs is None:
            obs = self.processed_x
        if action is None:
            action = self.action_ph

        hidden_size = [64, 64]
        nonlinearity = tf.nn.relu

        with tf.variable_scope(scope, reuse=reuse):
            # input to the system
            vf_h = tf.layers.flatten(obs)
            last_layer = tf.concat([vf_h, action], axis=-1)

            # create the hidden layers
            for i, hidden in enumerate(hidden_size):
                last_layer = tf.layers.dense(inputs=last_layer,
                                             units=hidden,
                                             name='fc_{}'.format(i),
                                             activation=nonlinearity)

            # create the output layer
            value_fn = tf.layers.dense(inputs=last_layer,
                                       units=1,
                                       name=scope,
                                       activation=None)

            self.value_fn = value_fn
            self._value = value_fn[:, 0]

        return self.value_fn


class LSTMPolicy(FullyConnectedPolicy):
    """Policy object that implements a DDPG-like actor critic, using LSTMs.

    sess : tf.Session
        The current TensorFlow session
    ob_space : gym.space.*
        The observation space of the environment
    ac_space : gym.space.*
        The action space of the environment
    n_env : int
        The number of environments to run
    n_steps : int
        The number of steps to run for each environment
    n_batch : int
        The number of batch to run (n_envs * n_steps)
    reuse : bool
        If the policy is reusable or not
    layers : list of int
        The size of the Neural network for the policy (if None, default to
        [64, 64])
    cnn_extractor : function
        the CNN feature extraction
    feature_extraction : str
        The feature extraction type ("cnn" or "mlp")
    layer_norm : bool
        enable layer normalisation
    kwargs : dict
        Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 n_env,
                 n_steps,
                 n_batch,
                 reuse=False,
                 layers=None,
                 cnn_extractor=nature_cnn,
                 feature_extraction="mlp",
                 layer_norm=False,
                 **kwargs):
        super(LSTMPolicy, self).__init__(
            sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256,
            reuse=reuse, scale=(feature_extraction == "cnn"))

    # TODO: observations need to be entire trajectories!
    def make_actor(self, obs=None, reuse=False, scope="pi"):
        """Create an actor object.

        Parameters
        ----------
        obs : tf.Tensor
            The observation placeholder (can be None for default placeholder)
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor

        Returns
        -------
        tf.Tensor
            the output tensor
        """
        if obs is None:
            obs = self.processed_x

        hidden_size = [256]
        horizon = 1000

        # convert batch to sequence
        obs = tf.split(obs, horizon, 1)

        with tf.variable_scope(scope, reuse=reuse):
            # Part 1. create the LSTM hidden layers

            # create the hidden layers
            lstm_layers = [tf.contrib.rnn.BasicLSTMCell(size)
                           for size in hidden_size]

            # stack up multiple LSTM layers
            cell = tf.contrib.rnn.MultiRNNCell(lstm_layers)

            # getting an initial state of all zeros
            lstm_outputs, final_state = tf.contrib.rnn.static_rnn(
                cell, obs, dtype=tf.float32)

            # Part 2. Create the output from the LSTM model

            # initialize the weights and biases
            out_w = tf.get_variable(
                "W",
                [hidden_size[-1], self.ac_space],
                initializer=tf.truncated_normal_initializer())
            out_b = tf.get_variable(
                "b",
                [self.ac_space],
                initializer=tf.zeros_initializer())

            # compute the output from the neural network model
            self.policy = tf.matmul(lstm_outputs[-1], out_w) + out_b

        return self.policy


# TODO
class FeudalPolicy(FeedForwardPolicy):
    pass


# TODO
class HIROPolicy(FeedForwardPolicy):  # TODO: rename
    pass
