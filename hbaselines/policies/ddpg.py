from stable_baselines.ddpg.policies import FeedForwardPolicy, nature_cnn
from hbaselines.models import build_fcnet, build_lstm
import tensorflow as tf


class FullyConnectedPolicy(FeedForwardPolicy):
    """Policy object that implements a DDPG-like actor critic, using MLPs."""

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        """See parent class."""
        if obs is None:
            obs = self.processed_x

        self.policy = build_fcnet(
            inputs=tf.layers.flatten(obs),
            num_outputs=self.ac_space.shape[0],
            hidden_size=[128, 128],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
            scope=scope,
            reuse=reuse
        )

        return self.policy

    def make_critic(self, obs=None, action=None, reuse=False, scope="qf"):
        """See parent class."""
        if obs is None:
            obs = self.processed_x
        if action is None:
            action = self.action_ph

        vf_h = tf.layers.flatten(obs)

        value_fn = build_fcnet(
            inputs=tf.concat([vf_h, action], axis=-1),
            num_outputs=1,
            hidden_size=[128, 128],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
            scope=scope,
            reuse=reuse
        )

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

        # input placeholder to the internal states component of the actor
        self.states_ph = None

        # output internal states component from the actor
        self.state = None

        # initial internal state of the actor
        self.state_init = None

        # placeholders for the actor recurrent network
        self.train_length = tf.placeholder(dtype=tf.int32, name="train_length")
        self.batch_size = tf.placeholder(dtype=tf.int32, name="batch_size")

        # hidden size of the actor LSTM
        self.actor_size = 64

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        """See parent class."""
        if obs is None:
            obs = self.processed_x

        # convert batch to sequence
        obs = tf.reshape(
            tf.layers.flatten(obs),
            [self.batch_size, self.train_length, self.ob_space.shape[0]]
        )

        # create the LSTM model
        self.policy, self.state_init, self.states_ph, self.state = build_lstm(
            inputs=obs,
            num_outputs=self.ac_space.shape[0],
            hidden_size=self.actor_size,
            scope=scope,
            reuse=reuse,
            nonlinearity=tf.nn.tanh,
        )

        return self.policy

    def step(self, obs, state=None, mask=None):
        """See parent class."""
        return self.sess.run(
            [self.policy, self.state],
            feed_dict={self.obs_ph: obs, self.states_ph: state}
        )


# TODO
class FeudalPolicy(LSTMPolicy):
    """Policy object for the Feudal (FuN) hierarchical model.

    blank
    """
    pass


class HIROPolicy(LSTMPolicy):
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

        # number of steps after which the manager performs an action
        self.c = 0  # FIXME

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
        self.cur_goal = None  # [0 for _ in range(ob_space.shape[0])]

        # hidden size of the actor LSTM  # TODO: add to inputs
        self.actor_size = 64

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
        tuple of tf.Tensor
            the output tensor from the manager and the worker
        """
        if obs is None:
            obs = self.processed_x

        # input to the manager
        manager_in = tf.reshape(
            tf.layers.flatten(obs),
            [self.batch_size, self.train_length, self.ob_space.shape[0]]
        )

        # create the policy for the manager
        m_policy, m_state_init, m_states_ph, m_state = build_lstm(
            inputs=manager_in,
            num_outputs=self.ob_space.shape[0],
            hidden_size=self.actor_size,
            scope='manager_{}'.format(scope),
            reuse=reuse,
            nonlinearity=tf.nn.tanh,
        )

        # input to the worker
        worker_in = tf.reshape(
            tf.concat([tf.layers.flatten(obs), self.goal_ph], axis=-1),
            [self.batch_size, self.train_length, self.ob_space.shape[0]]
        )

        # create the policy for the worker
        w_policy, w_state_init, w_states_ph, w_state = build_lstm(
            inputs=worker_in,
            num_outputs=self.ac_space.shape[0],
            hidden_size=self.actor_size,
            scope='worker_{}'.format(scope),
            reuse=reuse,
            nonlinearity=tf.nn.tanh,
        )

        self.policy = (m_policy, w_policy)
        self.state_init = (m_state_init, w_state_init)
        self.states_ph = (m_states_ph, w_states_ph)
        self.state = (m_state, w_state)

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
        tuple of tf.Tensor
            the output tensor for the manager and the worker
        """
        if obs is None:
            obs = self.processed_x
        if action is None:
            action = self.action_ph

        vf_h = tf.layers.flatten(obs)

        # value function for the manager
        value_fn = build_fcnet(
            inputs=tf.concat([vf_h, self.goal_ph], axis=-1),
            num_outputs=1,
            hidden_size=[128, 128],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
            scope=scope,
            reuse=reuse,
            prefix='manager',
        )
        self.manager_vf = value_fn
        self._manager_vf = value_fn[:, 0]

        # value function for the worker
        value_fn = build_fcnet(
            inputs=tf.concat([vf_h, self.goal_ph, action], axis=-1),
            num_outputs=1,
            hidden_size=[128, 128],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
            scope=scope,
            reuse=reuse,
            prefix='worker',
        )
        self.worker_vf = value_fn
        self._worker_vf = value_fn[:, 0]

        self.value_fn = (self.manager_vf, self.worker_vf)

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
            self.cur_goal = self.sess.run(
                self.manager,
                feed_dict={
                    self.obs_ph: obs,
                    self.states_ph: state[0]  # TODO: make state tuple in this case
                }
            )
        else:
            # TODO: the thing they do in the HIRO paper
            self.cur_goal = self.update_goal(self.cur_goal)

        action = self.sess.run(
            self.worker,
            feed_dict={
                self.obs_ph: obs,
                self.states_ph: state[1],  # TODO: make state tuple in this case
                self.goal_ph: self.cur_goal
            }
        )

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
            v_manager = self.sess.run(self._manager_vf,
                                      feed_dict={self.obs_ph: obs,
                                                 self.goal_ph: self.cur_goal})
        else:
            v_manager = None

        v_worker = self.sess.run(self._worker_vf,
                                 feed_dict={self.obs_ph: obs,
                                            self.states_ph: state[1],  # TODO: make state tuple in this case
                                            self.goal_ph: self.cur_goal,
                                            self.action_ph: action})

        return v_manager, v_worker

    def update_goal(self, goal, obs, prev_obs):
        """Update the goal when the manager isn't issuing commands.

        Parameters
        ----------
        goal : list of float or np.ndarray
            previous step goals
        obs : list of float or np.ndarray
            observations from the current time step
        prev_obs : list of float or np.ndarray
            observations from the previous time step

        Returns
        -------
        list of float or list of int
            current step goals
        """
        return prev_obs + goal - obs
