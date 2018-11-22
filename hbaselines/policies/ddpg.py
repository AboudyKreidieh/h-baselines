from stable_baselines.ddpg.policies import FeedForwardPolicy, nature_cnn
import tensorflow as tf


class FullyConnectedPolicy(FeedForwardPolicy):
    """Policy object that implements a DDPG-like actor critic, using MLPs."""
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
    """Policy object that implements a DDPG-like actor critic, using LSTMs."""

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
    """Policy object for the Feudal (FuN) hierarchical model.
    """
    pass


# TODO
class HIROPolicy(FeedForwardPolicy):  # TODO: rename
    """Policy object for the HIRO hierarchical model.

    In this policy, we consider two actors and two critic, one for the manager
    and another for the worker.

    The manager receives state information and returns goals, while the worker
    receives the states and goals and returns an action. Moreover, while the
    worker performs actions at every time step, the manager only does so every
    c time steps (he is informed when to act from the algorithm).

    The manager is trained to maximize the total expected return from the
    environment, while the worker is rewarded to achieving states that are
    similar to those requested by the manager.

    See: https://arxiv.org/abs/1805.08296
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
        super(HIROPolicy, self).__init__(
            sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256,
            reuse=reuse, scale=(feature_extraction == "cnn"))

        # manager policy
        self.manager = None

        # worker policy
        self.worker = None

        # manager value function
        self.manager_vf = None
        self._manager_vf = None

        # worker value function
        self.worker_vf = None
        self._worker_vf = None

        # a placeholder for goals that are communicated to the worker
        self.goal_ph = tf.placeholder(dtype=tf.float32,
                                      shape=self.ob_space.shape[0])

        # a variable that internally stores the most recent goal to the worker
        self.cur_goal = [0 for _ in range(ob_space.shape[0])]

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
            the output tensor from the manager
        tf.Tensor
            the output tensor from the worker
        """
        if obs is None:
            obs = self.processed_x

        hidden_size = [64, 64]
        nonlinearity = tf.nn.relu

        # create the policy for the manager
        with tf.variable_scope(scope, reuse=reuse):
            # TODO: more than the current obs?
            # input to the system
            last_layer = tf.layers.flatten(obs)

            # create the hidden layers
            for i, hidden in enumerate(hidden_size):
                last_layer = tf.layers.dense(inputs=last_layer,
                                             units=hidden,
                                             name='manager_{}'.format(i),
                                             activation=nonlinearity)

            # create the output layer
            self.manager = tf.layers.dense(inputs=last_layer,
                                           units=self.ob_space.shape[0],
                                           name=scope,
                                           activation=None)

        # create the policy for the worker
        with tf.variable_scope(scope, reuse=reuse):
            # input to the system
            last_layer = tf.concat([obs, self.goal_ph], axis=-1)

            # create the hidden layers
            for i, hidden in enumerate(hidden_size):
                last_layer = tf.layers.dense(inputs=last_layer,
                                             units=hidden,
                                             name='worker_{}'.format(i),
                                             activation=nonlinearity)

            # create the output layer
            self.worker = tf.layers.dense(inputs=last_layer,
                                          units=self.ac_space.shape[0],
                                          name=scope,
                                          activation=None)

        return self.manager, self.worker

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
            the output tensor for the manager
        tf.Tensor
            the output tensor for the worker
        """
        if obs is None:
            obs = self.processed_x
        if action is None:
            action = self.action_ph

        hidden_size = [64, 64]
        nonlinearity = tf.nn.relu

        # value function for the manager
        with tf.variable_scope(scope, reuse=reuse):
            # TODO: more than the current obs?
            # input to the system
            vf_h = tf.layers.flatten(obs)
            last_layer = tf.concat([vf_h, self.goal_ph], axis=-1)

            # create the hidden layers
            for i, hidden in enumerate(hidden_size):
                last_layer = tf.layers.dense(inputs=last_layer,
                                             units=hidden,
                                             name='manager_{}'.format(i),
                                             activation=nonlinearity)

            # create the output layer
            value_fn = tf.layers.dense(inputs=last_layer,
                                       units=1,
                                       name=scope,
                                       activation=None)

            self.manager_vf = value_fn
            self._manager_vf = value_fn[:, 0]

        # value function for the worker
        with tf.variable_scope(scope, reuse=reuse):
            # input to the system
            vf_h = tf.layers.flatten(obs)
            last_layer = tf.concat([vf_h, self.goal_ph, action], axis=-1)

            # create the hidden layers
            for i, hidden in enumerate(hidden_size):
                last_layer = tf.layers.dense(inputs=last_layer,
                                             units=hidden,
                                             name='manager_{}'.format(i),
                                             activation=nonlinearity)

            # create the output layer
            value_fn = tf.layers.dense(inputs=last_layer,
                                       units=1,
                                       name=scope,
                                       activation=None)

            self.worker_vf = value_fn
            self._worker_vf = value_fn[:, 0]

        return self.value_fn

    def step(self, obs, state=None, mask=None, **kwargs):
        """Return the policy for a single step.

        Parameters
        ----------
        obs : list of float or list of int
            The current observation of the environment
        state : list of float
            The last states (used in recurrent policies)
        mask : list of float
            The last masks (used in recurrent policies)

        Returns
        -------
        list of float
            the current goal from the manager
        list of float
            actions from the worker
        """
        if kwargs["apply_manager"]:
            # TODO: more than the current obs?
            self.cur_goal = self.sess.run(self.manager,
                                          feed_dict={self.obs_ph: obs})
        else:
            # TODO: the thing they do in the HIRO paper
            pass

        action = self.sess.run(self.worker,
                               feed_dict={self.obs_ph: obs,
                                          self.goal_ph: self.cur_goal})

        return self.cur_goal, action

    def value(self, obs, action, state=None, mask=None, **kwargs):
        """Return the value for a single step.

        Parameters
        ----------
        obs : list of float or list of int
            The current observation of the environment
        action : list of float or list of int
            The taken action
        state : list of float
            The last states (used in recurrent policies)
        mask : list of float
            The last masks (used in recurrent policies)

        Returns
        -------
        list of float
            The associated value of the goal from the manager
        list of float
            The associated value of the action from the worker
        """
        if kwargs["apply_manager"]:
            # TODO: more than the current obs?
            v_manager = self.sess.run(self._manager_vf,
                                      feed_dict={self.obs_ph: obs,
                                                 self.goal_ph: self.cur_goal})
        else:
            v_manager = None

        v_worker = self.sess.run(self._worker_vf,
                                 feed_dict={self.obs_ph: obs,
                                            self.goal_ph: self.cur_goal,
                                            self.action_ph: action})

        return v_manager, v_worker
