"""SAC-compatible multi-agent feedforward policy."""
import tensorflow as tf
import numpy as np

from hbaselines.multi_fcnet.base import MultiFeedForwardPolicy as BasePolicy
from hbaselines.fcnet.sac import FeedForwardPolicy
from hbaselines.multi_fcnet.replay_buffer import MultiReplayBuffer


# Stabilizing term to avoid NaN (prevents division by zero or log of zero)
EPS = 1e-6
# Cap the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class MultiFeedForwardPolicy(BasePolicy):
    """SAC-compatible multi-agent feedforward neural.

    The attributes described in this docstring are only used if the `maddpg`
    parameter is set to True. The attributes are dictionaries of their
    described form for each agent if `shared` is set to False.

    See the docstring of the parent class for a further description of this
    class.

    Attributes
    ----------
    target_entropy : float
        target entropy used when learning the entropy coefficient
    replay_buffer : hbaselines.multi_fcnet.replay_buffer.MultiReplayBuffer
        the replay buffer for each agent
    terminals1 : tf.compat.v1.placeholder
        placeholder for the next step terminals for each agent
    rew_ph : tf.compat.v1.placeholder
        placeholder for the rewards for each agent
    action_ph : tf.compat.v1.placeholder
        placeholder for the actions for each agent
    obs_ph : tf.compat.v1.placeholder
        placeholder for the observations for each agent
    obs1_ph : tf.compat.v1.placeholder
        placeholder for the next step observations for each agent
    all_obs_ph : tf.compat.v1.placeholder
        placeholder for the last step full state observations
    all_obs1_ph : tf.compat.v1.placeholder
        placeholder for the current step full state observations
    all_action_ph : tf.compat.v1.placeholder
        placeholder for the actions of all agents
    deterministic_action : tf.Variable
        the output from the deterministic actor
    policy_out : tf.Variable
        the output from the stochastic actor
    logp_pi : tf.Variable
        the log-probability of a given observation given the output action from
        the policy
    logp_action : tf.Variable
        the log-probability of a given observation given a fixed action. Used
        by the hierarchical policy to perform off-policy corrections.
    qf1 : tf.Variable
        the output from the first Q-function
    qf2 : tf.Variable
        the output from the second Q-function
    value_fn : tf.Variable
        the output from the value function
    log_alpha : tf.Variable
        the log of the entropy coefficient
    alpha : tf.Variable
        the entropy coefficient
    value_target : tf.Variable
        the output from the target value function. Takes as input the next-step
        observations
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 co_space,
                 buffer_size,
                 batch_size,
                 actor_lr,
                 critic_lr,
                 verbose,
                 tau,
                 gamma,
                 layer_norm,
                 layers,
                 act_fun,
                 use_huber,
                 target_entropy,
                 shared,
                 maddpg,
                 all_ob_space=None,
                 n_agents=1,
                 scope=None,
                 zero_fingerprint=False,
                 fingerprint_dim=2):
        """Instantiate a multi-agent feed-forward neural network policy.

        Parameters
        ----------
        sess : tf.compat.v1.Session
            the current TensorFlow session
        ob_space : gym.spaces.*
            the observation space of the environment
        ac_space : gym.spaces.*
            the action space of the environment
        co_space : gym.spaces.*
            the context space of the environment
        buffer_size : int
            the max number of transitions to store
        batch_size : int
            SGD batch size
        actor_lr : float
            actor learning rate
        critic_lr : float
            critic learning rate
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        tau : float
            target update rate
        gamma : float
            discount factor
        layer_norm : bool
            enable layer normalisation
        layers : list of int or None
            the size of the Neural network for the policy
        act_fun : tf.nn.*
            the activation function to use in the neural network
        use_huber : bool
            specifies whether to use the huber distance function as the loss
            for the critic. If set to False, the mean-squared error metric is
            used instead
        target_entropy : float
            target entropy used when learning the entropy coefficient. If set
            to None, a heuristic value is used.
        shared : bool
            whether to use a shared policy for all agents
        maddpg : bool
            whether to use an algorithm-specific variant of the MADDPG
            algorithm
        all_ob_space : gym.spaces.*
            the observation space of the full state space. Used by MADDPG
            variants of the policy.
        n_agents : int
            the number of agents in the networks. This is needed if using
            MADDPG with a shared policy to compute the length of the full
            action space. Otherwise, it is not used.
        scope : str
            an upper-level scope term. Used by policies that call this one.
        zero_fingerprint : bool
            whether to zero the last two elements of the observations for the
            actor and critic computations. Used for the worker policy when
            fingerprints are being implemented.
        fingerprint_dim : bool
            the number of fingerprint elements in the observation. Used when
            trying to zero the fingerprint elements.
        """
        # Instantiate a few terms (needed if MADDPG is used).
        if target_entropy is None:
            if isinstance(ac_space, dict):
                self.target_entropy = {
                    key: -np.prod(ac_space[key].shape)
                    for key in ac_space.keys()
                }
            else:
                self.target_entropy = -np.prod(ac_space.shape)
        else:
            self.target_entropy = target_entropy

        # variables to be initialized later (if MADDPG is used)
        self.replay_buffer = None
        self.terminals1 = None
        self.rew_ph = None
        self.action_ph = None
        self.obs_ph = None
        self.obs1_ph = None
        self.all_obs_ph = None
        self.all_obs1_ph = None
        self.all_action_ph = None
        self.deterministic_action = None
        self.policy_out = None
        self.logp_pi = None
        self.logp_action = None
        self.qf1 = None
        self.qf2 = None
        self.value_fn = None
        self.log_alpha = None
        self.alpha = None
        self.value_target = None

        super(MultiFeedForwardPolicy, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            co_space=co_space,
            buffer_size=buffer_size,
            batch_size=batch_size,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            verbose=verbose,
            tau=tau,
            gamma=gamma,
            layer_norm=layer_norm,
            layers=layers,
            act_fun=act_fun,
            use_huber=use_huber,
            shared=shared,
            maddpg=maddpg,
            all_ob_space=all_ob_space,
            n_agents=n_agents,
            base_policy=FeedForwardPolicy,
            scope=scope,
            zero_fingerprint=zero_fingerprint,
            fingerprint_dim=fingerprint_dim,
            additional_params=dict(
                target_entropy=target_entropy,
            ),
        )

    def _setup_maddpg(self, scope):
        """See setup."""
        # Create an input placeholder for the full state observations.
        self.all_obs_ph = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None,) + self.all_ob_space.shape,
            name='all_obs')

        if self.shared:
            # Create an input placeholder for the full actions.
            self.all_action_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + (self.all_ob_space.shape[0] * self.n_agents,),
                name='all_actions')

            # Create actor and critic networks for the shared policy.
            replay_buffer, terminals1, rew_ph, action_ph, obs_ph, obs1_ph, \
                deterministic_action, policy_out, logp_pi, logp_action, qf1, \
                qf2, value_fn, log_alpha, alpha, value_target = \
                self._setup_agent(
                    ob_space=self.ob_space,
                    ac_space=self.ac_space,
                    co_space=self.co_space,
                )

            # Store the new objects in their respective attributes.
            self.replay_buffer = replay_buffer
            self.terminals1 = terminals1
            self.rew_ph = rew_ph
            self.action_ph = action_ph
            self.obs_ph = obs_ph
            self.obs1_ph = obs1_ph
            self.deterministic_action = deterministic_action
            self.policy_out = policy_out
            self.logp_pi = logp_pi
            self.logp_action = logp_action
            self.qf1 = qf1
            self.qf2 = qf2
            self.value_fn = value_fn
            self.log_alpha = log_alpha
            self.alpha = alpha
            self.value_target = value_target
        else:
            # Create an input placeholder for the full actions.
            all_ac_dim = sum(self.ac_space[key].shape[0]
                             for key in self.ac_space.keys())

            self.all_action_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, all_ac_dim),
                name='all_actions')

            self.replay_buffer = {}
            self.terminals1 = {}
            self.rew_ph = {}
            self.action_ph = {}
            self.obs_ph = {}
            self.obs1_ph = {}
            self.deterministic_action = {}
            self.policy_out = {}
            self.logp_pi = {}
            self.logp_action = {}
            self.qf1 = {}
            self.qf2 = {}
            self.value_fn = {}
            self.log_alpha = {}
            self.alpha = {}
            self.value_target = {}

            # We move through the keys in a sorted fashion so that we may
            # collect the observations and actions for the full state in a
            # sorted manner as well.
            for key in sorted(self.ob_space.keys()):
                # Create actor and critic networks for the the individual
                # policies.
                with tf.compat.v1.variable_scope(key, reuse=False):
                    replay_buffer, terminals1, rew_ph, action_ph, obs_ph, \
                        obs1_ph, deterministic_action, policy_out, logp_pi, \
                        logp_action, qf1, qf2, value_fn, log_alpha, alpha, \
                        value_target = \
                        self._setup_agent(
                            ob_space=self.ob_space[key],
                            ac_space=self.ac_space[key],
                            co_space=self.co_space[key],
                        )

                # Store the new objects in their respective attributes.
                self.replay_buffer[key] = replay_buffer
                self.terminals1[key] = terminals1
                self.rew_ph[key] = rew_ph
                self.action_ph[key] = action_ph
                self.obs_ph[key] = obs_ph
                self.obs1_ph[key] = obs1_ph
                self.deterministic_action[key] = deterministic_action
                self.policy_out[key] = policy_out
                self.logp_pi[key] = logp_pi
                self.logp_action[key] = logp_action
                self.qf1[key] = qf1
                self.qf2[key] = qf2
                self.value_fn[key] = value_fn
                self.log_alpha[key] = log_alpha
                self.alpha[key] = alpha
                self.value_target[key] = value_target

    def _setup_agent(self, ob_space, ac_space, co_space):
        """Create the components for an individual agent.

        Parameters
        ----------
        ob_space : gym.spaces.*
            the observation space of the individual agent
        ac_space : gym.spaces.*
            the action space of the individual agent
        co_space : gym.spaces.*
            the context space of the individual agent

        Returns
        -------
        MultiReplayBuffer
            the replay buffer object for each agent
        tf.compat.v1.placeholder
            placeholder for the next step terminals for each agent
        tf.compat.v1.placeholder
            placeholder for the rewards for each agent
        tf.compat.v1.placeholder
            placeholder for the actions for each agent
        tf.compat.v1.placeholder
            placeholder for the observations for each agent
        tf.compat.v1.placeholder
            placeholder for the next step observations for each agent
        tf.Variable
            the output from the deterministic actor
        tf.Variable
            the output from the stochastic actor
        tf.Variable
            the log-probability of a given observation given the output action
            from the policy
        tf.Variable
            the log-probability of a given observation given a fixed action.
            Used by the hierarchical policy to perform off-policy corrections.
        tf.Variable
            the output from the first Q-function
        tf.Variable
            the output from the second Q-function
        tf.Variable
            the output from the value function
        tf.Variable
            the log of the entropy coefficient
        tf.Variable
            the entropy coefficient
        tf.Variable
            the output from the target value function. Takes as input the
            next-step observations
        """
        # Compute the shape of the input observation space, which may include
        # the contextual term.
        ob_dim = self._get_ob_dim(ob_space, co_space)

        # =================================================================== #
        # Step 1: Create a replay buffer object.                              #
        # =================================================================== #

        replay_buffer = MultiReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            obs_dim=ob_dim[0],
            ac_dim=ac_space.shape[0],
            all_obs_dim=self.all_obs_ph.shape[-1],
            all_ac_dim=self.all_action_ph.shape[-1],
        )

        # =================================================================== #
        # Step 2: Create input variables.                                     #
        # =================================================================== #

        with tf.compat.v1.variable_scope("input", reuse=False):
            terminals1 = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='terminals1')
            rew_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='rewards')
            action_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ac_space.shape,
                name='actions')
            obs_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ob_dim,
                name='obs0')
            obs1_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ob_dim,
                name='obs1')

        # logging of rewards to tensorboard
        with tf.compat.v1.variable_scope("input_info", reuse=False):
            tf.compat.v1.summary.scalar('rewards', tf.reduce_mean(rew_ph))

        # =================================================================== #
        # Step 3: Create actor and critic variables.                          #
        # =================================================================== #

        # Create networks and core TF parts that are shared across setup parts.
        with tf.compat.v1.variable_scope("model", reuse=False):
            deterministic_action, policy_out, logp_pi, logp_action = \
                self.make_actor(obs_ph, ac_space, action_ph)
            qf1, qf2, value_fn = self.make_critic(
                self.all_obs_ph, self.all_action_ph,
                scope="centralized_value_fns", create_qf=True, create_vf=True)

            # The entropy coefficient or entropy can be learned automatically,
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            log_alpha = tf.compat.v1.get_variable(
                'log_alpha',
                dtype=tf.float32,
                initializer=0.0)
            alpha = tf.exp(log_alpha)

        with tf.compat.v1.variable_scope("target", reuse=False):
            # Create the value network
            _, _, value_target = self.make_critic(
                self.all_obs_ph,
                scope="centralized_value_fns", create_qf=False, create_vf=True)

        return replay_buffer, terminals1, rew_ph, action_ph, obs_ph, obs1_ph, \
            deterministic_action, policy_out, logp_pi, logp_action, qf1, qf2, \
            value_fn, log_alpha, alpha, value_target

    def make_actor(self, obs, ac_space, action, reuse=False, scope="pi"):
        """Create the actor variables.

        Parameters
        ----------
        obs : tf.compat.v1.placeholder
            the input observation placeholder
        ac_space : gym.spaces.*
            the action space of an agent
        action : tf.compat.v1.placeholder
            the input action placeholder
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor

        Returns
        -------
        tf.Variable
            the output from the deterministic actor
        tf.Variable
            the output from the stochastic actor
        tf.Variable
            the log-probability of a given observation given the output action
            from the policy
        tf.Variable
            the log-probability of a given observation given a fixed action
        """
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            pi_h = obs

            # create the hidden layers
            for i, layer_size in enumerate(self.layers):
                pi_h = self._layer(
                    pi_h,  layer_size, 'fc{}'.format(i),
                    act_fun=self.act_fun,
                    layer_norm=self.layer_norm
                )

            # create the output mean
            policy_mean = self._layer(
                pi_h, ac_space.shape[0], 'mean',
                act_fun=None,
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3)
            )

            # create the output log_std
            log_std = self._layer(
                pi_h, ac_space.shape[0], 'log_std',
                act_fun=None,
            )

        # OpenAI Variation to cap the standard deviation
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        std = tf.exp(log_std)

        # Reparameterization trick
        policy = policy_mean + tf.random.normal(tf.shape(policy_mean)) * std
        logp_pi = self._gaussian_likelihood(policy, policy_mean, log_std)
        logp_ac = self._gaussian_likelihood(action, policy_mean, log_std)

        # Apply squashing and account for it in the probability
        _, _, logp_ac = self._apply_squashing_func(
            policy_mean, action, logp_ac)
        deterministic_policy, policy, logp_pi = self._apply_squashing_func(
            policy_mean, policy, logp_pi)

        return deterministic_policy, policy, logp_pi, logp_ac

    def make_critic(self,
                    obs,
                    action=None,
                    reuse=False,
                    scope="value_fns",
                    create_qf=True,
                    create_vf=True):
        """Create the critic variables.

        Parameters
        ----------
        obs : tf.compat.v1.placeholder
            the input observation placeholder
        action : tf.compat.v1.placeholder
            the input action placeholder
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor
        create_qf : bool
            whether to create the Q-functions
        create_vf : bool
            whether to create the value function

        Returns
        -------
        tf.Variable
            the output from the first Q-function. Set to None if `create_qf` is
            False.
        tf.Variable
            the output from the second Q-function. Set to None if `create_qf`
            is False.
        tf.Variable
            the output from the value function. Set to None if `create_vf` is
            False.
        """
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Value function
            if create_vf:
                with tf.compat.v1.variable_scope("vf", reuse=reuse):
                    vf_h = obs

                    # create the hidden layers
                    for i, layer_size in enumerate(self.layers):
                        vf_h = self._layer(
                            vf_h, layer_size, 'fc{}'.format(i),
                            act_fun=self.act_fun,
                            layer_norm=self.layer_norm
                        )

                    # create the output layer
                    value_fn = self._layer(
                        vf_h, 1, 'vf_output',
                        kernel_initializer=tf.random_uniform_initializer(
                            minval=-3e-3, maxval=3e-3)
                    )
            else:
                value_fn = None

            # Double Q values to reduce overestimation
            if create_qf:
                with tf.compat.v1.variable_scope('qf1', reuse=reuse):
                    # concatenate the observations and actions
                    qf1_h = tf.concat([obs, action], axis=-1)

                    # create the hidden layers
                    for i, layer_size in enumerate(self.layers):
                        qf1_h = self._layer(
                            qf1_h, layer_size, 'fc{}'.format(i),
                            act_fun=self.act_fun,
                            layer_norm=self.layer_norm
                        )

                    # create the output layer
                    qf1 = self._layer(
                        qf1_h, 1, 'qf_output',
                        kernel_initializer=tf.random_uniform_initializer(
                            minval=-3e-3, maxval=3e-3)
                    )

                with tf.compat.v1.variable_scope('qf2', reuse=reuse):
                    # concatenate the observations and actions
                    qf2_h = tf.concat([obs, action], axis=-1)

                    # create the hidden layers
                    for i, layer_size in enumerate(self.layers):
                        qf2_h = self._layer(
                            qf2_h, layer_size, 'fc{}'.format(i),
                            act_fun=self.act_fun,
                            layer_norm=self.layer_norm
                        )

                    # create the output layer
                    qf2 = self._layer(
                        qf2_h, 1, 'qf_output',
                        kernel_initializer=tf.random_uniform_initializer(
                            minval=-3e-3, maxval=3e-3)
                    )
            else:
                qf1, qf2 = None, None

        return qf1, qf2, value_fn

    @staticmethod
    def _gaussian_likelihood(input_, mu_, log_std):
        """Compute log likelihood of a gaussian.

        Here we assume this is a Diagonal Gaussian.

        Parameters
        ----------
        input_ : tf.Variable
            the action by the policy
        mu_ : tf.Variable
            the policy mean
        log_std : tf.Variable
            the policy log std

        Returns
        -------
        tf.Variable
            the log-probability of a given observation given the output action
            from the policy
        """
        pre_sum = -0.5 * (((input_ - mu_) / (
                    tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(
            2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    @staticmethod
    def _apply_squashing_func(mu_, pi_, logp_pi):
        """Squash the output of the Gaussian distribution.

        This method also accounts for that in the log probability. The squashed
        mean is also returned for using deterministic actions.

        Parameters
        ----------
        mu_ : tf.Variable
            mean of the gaussian
        pi_ : tf.Variable
            output of the policy (or action) before squashing
        logp_pi : tf.Variable
            log probability before squashing

        Returns
        -------
        tf.Variable
            the output from the squashed deterministic policy
        tf.Variable
            the output from the squashed stochastic policy
        tf.Variable
            the log probability of a given squashed action
        """
        # Squash the output
        deterministic_policy = tf.nn.tanh(mu_)
        policy = tf.nn.tanh(pi_)

        # Squash correction (from original implementation)
        logp_pi -= tf.reduce_sum(tf.math.log(1 - policy ** 2 + EPS), axis=1)

        return deterministic_policy, policy, logp_pi

    def _initialize_maddpg(self):
        """See initialize."""
        pass  # TODO

    def _update_maddpg(self, update_actor=True, **kwargs):
        """See update."""
        pass  # TODO

    def _get_action_maddpg(self, obs, context, apply_noise, random_actions):
        """See get_action."""
        pass  # TODO

    def _value_maddpg(self, obs, context, action):
        """See value."""
        pass  # TODO

    def _store_transition_maddpg(self,
                                 obs0,
                                 context0,
                                 action,
                                 reward,
                                 obs1,
                                 context1,
                                 done,
                                 all_obs0,
                                 all_obs1,
                                 evaluate):
        """See store_transition."""
        pass  # TODO

    def _get_td_map_maddpg(self):
        """See get_td_map."""
        pass  # TODO
