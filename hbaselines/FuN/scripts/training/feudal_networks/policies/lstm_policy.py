"""
##############################
# LSTM Network policy script #
##############################
"""


import tensorflow as tf
from hbaselines.FuN.scripts.training.feudal_networks.models.models import \
    (linear, conv2d, build_lstm, normalized_columns_initializer)
import hbaselines.FuN.scripts.training.feudal_networks.policies.policy_utils \
    as policy_utils
from hbaselines.FuN.scripts.training.feudal_networks.policies.configs.lstm_config \
    import config


class LSTMPolicy(object):
    """
    Policy of the Feudal network architecture.

    """

    def __init__(self, obs_space, act_space, global_step):
        """
         Instantiate an LSTM policy object.

        Parameters
        ----------
        obs_space : object
            Observation space
        act_space : object
            Action space
        global_step : object
            Global step defined for Feudal Network
        """
        self.global_step = global_step
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config
        self.local_steps = 0
        # build placeholders
        self.obs = x = tf.placeholder(tf.float32,
                                      [None] + list(obs_space),
                                      name='state')
        self.adv = tf.placeholder(tf.float32,
                                  [None],
                                  name="adv")
        self.ac = tf.placeholder(tf.float32,
                                 [None, act_space],
                                 name="ac")
        self.r = tf.placeholder(tf.float32,
                                [None],
                                name="r")

        print(self.r)
        # build perception
        for i in range(config.n_percept_hidden_layer):
            x = tf.nn.elu(conv2d(x, config.n_percept_filters,
                                 "l{}".format(i + 1), [3, 3], [2, 2]))

        # introduce a "fake" batch dimension of 1 after flatten so that we
        # can do LSTM over time dim
        x = tf.expand_dims(policy_utils.flatten(x), [0])
        x, self.state_init, self.state_in, self.state_out = build_lstm(
            x, config.size, 'lstm', tf.shape(self.obs)[:1])

        # on the lstm to output values for both the policy and value function
        # add hidden layer to value output so that less of a burden is placed
        vfhid = tf.nn.elu(linear(x,
                                 config.size,
                                 "value_hidden",
                                 normalized_columns_initializer(0.01)))
        self.vf = tf.reshape(
            linear(vfhid,
                   1,
                   "value",
                   normalized_columns_initializer(1.0)),
            [-1])

        # retrieve logits, sampling op
        self.logits = linear(x,
                             act_space,
                             "action",
                             normalized_columns_initializer(0.01))
        self.sample = policy_utils.categorical_sample(
            self.logits, act_space)[0, :]
        self.var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        # build loss
        log_prob_tf = tf.nn.log_softmax(self.logits)
        prob_tf = tf.nn.softmax(self.logits)
        pi_loss = - tf.reduce_sum(
                    tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)
        entropy = - tf.reduce_sum(prob_tf * log_prob_tf)
        vf_loss = 0.5 * tf.reduce_sum(tf.square(self.vf - self.r))
        beta = tf.train.polynomial_decay(config.beta_start, self.global_step,
                                         end_learning_rate=config.beta_end,
                                         decay_steps=config.decay_steps,
                                         power=1)
        self.loss = pi_loss + 0.5 * vf_loss - entropy * beta

        # summaries
        bs = tf.to_float(tf.shape(self.obs)[0])
        tf.summary.scalar("model/policy_loss", pi_loss / bs)
        tf.summary.scalar("model/value_mean", tf.reduce_mean(self.vf))
        tf.summary.scalar("model/value_loss", vf_loss / bs)
        tf.summary.scalar("model/value_loss_scaled", vf_loss / bs * .5)
        tf.summary.scalar("model/entropy", entropy / bs)
        tf.summary.scalar("model/entropy_loss_scaleed", -entropy / bs * beta)
        # tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
        tf.summary.scalar(
            "model/var_global_norm",
            tf.global_norm(self.var_list))
        tf.summary.scalar("model/beta", beta)
        tf.summary.image("model/state", self.obs)
        self.summary_op = tf.summary.merge_all()

    def get_initial_features(self):
        """
        Function to get the initial state features of the LSTM network

        """
        return self.state_init

    def act(self, ob, c, h):
        """
        Function to allow the LSTM Network to
        start acting based on the environmental
        observations.

        Parameters
        ----------
        ob : object
            Observation object
        c : object
            Cost
        h : object
            Internal state of the LSTM network
        """
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.obs: [ob],
                         self.state_in[0]: c,
                         self.state_in[1]: h})

    def value(self, ob, c, h):
        """
        Value function for the LSTM Network

        Parameters
        ----------
        ob : object
            Observation object
        c : object
            BLANK
        h : object
            Internal state of the LSTM Network
        """
        sess = tf.get_default_session()
        return sess.run(self.vf,
                        {self.obs: [ob],
                         self.state_in[0]: c,
                         self.state_in[1]: h})[0]

    def update_batch(self, batch):
        """
        Function to efficiently update the batch of data in the LSTM Network.
        It does so by just directly returning the batch
        object parameter passed.

        Parameters
        ----------
        batch : object
            Batch object
        """
        return batch
