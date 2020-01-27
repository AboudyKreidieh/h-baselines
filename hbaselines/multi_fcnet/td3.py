"""TD3-compatible multi-agent feedforward policy."""
import tensorflow as tf

from hbaselines.multi_fcnet.base import MultiFeedForwardPolicy as BasePolicy
from hbaselines.fcnet.td3 import FeedForwardPolicy
from hbaselines.multi_fcnet.replay_buffer import MultiReplayBuffer


class MultiFeedForwardPolicy(BasePolicy):
    """TD3-compatible multi-agent feedforward neural.

    The attributes described in this docstring are only used if the `maddpg`
    parameter is set to True. The attributes are dictionaries of their
    described form for each agent if `shared` is set to False.

    See the docstring of the parent class for a further description of this
    class.

    Attributes
    ----------
    target_policy_noise : float
        standard deviation term to the noise from the output of the target
        actor policy. See TD3 paper for more.
    target_noise_clip : float
        clipping term for the noise injected in the target actor policy
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
    actor_tf : tf.Variable
        the output from the actor network
    critic_tf : list of tf.Variable
        the output from the critic networks. Two networks are used to stabilize
        training.
    critic_target : list of tf.Variable
        the output from the critic target networks. Two networks are used to
        stabilize training.
    critic_loss : tf.Operation
        the operation that returns the loss of the critic
    critic_optimizer : tf.Operation
        the operation that updates the trainable parameters of the critic
    target_init_updates : tf.Operation
        an operation that sets the values of the trainable parameters of the
        target actor/critic to match those actual actor/critic
    target_soft_updates : tf.Operation
        soft target update function
    actor_loss : tf.Operation
        the operation that returns the loss of the actor
    actor_optimizer : tf.Operation
        the operation that updates the trainable parameters of the actor
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
                 noise,
                 target_policy_noise,
                 target_noise_clip,
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
        noise : float
            scaling term to the range of the action space, that is subsequently
            used as the standard deviation of Gaussian noise added to the
            action if `apply_noise` is set to True in `get_action`.
        target_policy_noise : float
            standard deviation term to the noise from the output of the target
            actor policy. See TD3 paper for more.
        target_noise_clip : float
            clipping term for the noise injected in the target actor policy
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
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip

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
        self.actor_tf = None
        self.critic_tf = None
        self.critic_target = None
        self.critic_loss = None
        self.critic_optimizer = None
        self.target_init_updates = None
        self.target_soft_updates = None
        self.actor_loss = None
        self.actor_optimizer = None

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
                noise=noise,
                target_policy_noise=target_policy_noise,
                target_noise_clip=target_noise_clip,
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
            replay_buffer, terminals1, rew_ph, action_ph, obs_ph, \
                obs1_ph, actor_tf, critic_tf, noisy_actor_target = \
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
            self.actor_tf = actor_tf
            self.critic_tf = critic_tf

            # Setup the target critic and critic update procedure.
            self.critic_target, self.critic_loss, self.critic_optimizer = \
                self._setup_critic_updates_shared(noisy_actor_target)

            # Create the target update operations.
            init, soft = self._setup_target_updates_shared(scope)
            self.target_init_updates = init
            self.target_soft_updates = soft

            # Setup the actor update procedure.
            loss, optimizer = self._setup_actor_updates_shared()
            self.actor_loss = loss
            self.actor_optimizer = optimizer

            # Setup the running means and standard deviations of the model
            # inputs and outputs.
            self._setup_stats_shared(scope or "Model")
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
            self.actor_tf = {}
            self.critic_tf = {}
            actor_targets = []

            # We move through the keys in a sorted fashion so that we may
            # collect the observations and actions for the full state in a
            # sorted manner as well.
            for key in sorted(self.ob_space.keys()):
                # Create actor and critic networks for the the individual
                # policies.
                with tf.compat.v1.variable_scope(key, reuse=False):
                    replay_buffer, terminals1, rew_ph, action_ph, obs_ph, \
                        obs1_ph, actor_tf, critic_tf, noisy_actor_target = \
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
                self.actor_tf[key] = actor_tf
                self.critic_tf[key] = critic_tf
                actor_targets.append(noisy_actor_target)

            # Setup the target critic and critic update procedure.
            self.critic_target, self.critic_loss, self.critic_optimizer = \
                self._setup_critic_updates_nonshared(actor_targets)

            # Create the target update operations.
            init, soft = self._setup_target_updates_nonshared(scope)
            self.target_init_updates = init
            self.target_soft_updates = soft

            # Setup the actor update procedure.
            loss, optimizer = self._setup_actor_updates_nonshared()
            self.actor_loss = loss
            self.actor_optimizer = optimizer

            # Setup the running means and standard deviations of the model
            # inputs and outputs.
            self._setup_stats_nonshared(scope or "Model")

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
            the output from the actor network
        list of tf.Variable
            the output from the critic networks. Two networks are used to
            stabilize training.
        tf.Variable
            the output from a noise target actor network
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
            shared=self.shared,
            n_agents=self.n_agents,
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

        with tf.compat.v1.variable_scope("model", reuse=False):
            actor_tf = self.make_actor(obs_ph, ac_space)
            critic_tf = [
                self.make_critic(self.all_obs_ph, self.all_action_ph,
                                 scope="centralized_qf_{}".format(i))
                for i in range(2)
            ]

        with tf.compat.v1.variable_scope("target", reuse=False):
            # create the target actor policy
            actor_target = self.make_actor(obs1_ph, ac_space)

            # smooth target policy by adding clipped noise to target actions
            target_noise = tf.random.normal(
                tf.shape(actor_target), stddev=self.target_policy_noise)
            target_noise = tf.clip_by_value(
                target_noise, -self.target_noise_clip, self.target_noise_clip)

            # clip the noisy action to remain in the bounds
            noisy_actor_target = tf.clip_by_value(
                actor_target + target_noise,
                ac_space.low,
                ac_space.high
            )

        return replay_buffer, terminals1, rew_ph, action_ph, obs_ph, obs1_ph, \
            actor_tf, critic_tf, noisy_actor_target

    def make_actor(self, obs, ac_space, reuse=False, scope="pi"):
        """Create an actor tensor.

        Parameters
        ----------
        obs : tf.compat.v1.placeholder
            the input observation placeholder of the individual agent
        ac_space : gym.space.*
            the action space of the individual agent
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor

        Returns
        -------
        tf.Variable
            the output from the actor
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

            # create the output layer
            policy = self._layer(
                pi_h, ac_space.shape[0], 'output',
                act_fun=tf.nn.tanh,
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3)
            )

            # scaling terms to the output from the policy
            ac_means = (ac_space.high + ac_space.low) / 2.
            ac_magnitudes = (ac_space.high - ac_space.low) / 2.

            policy = ac_means + ac_magnitudes * tf.to_float(policy)

        return policy

    def make_critic(self, obs, action, reuse=False, scope="qf"):
        """Create a critic tensor.

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

        Returns
        -------
        tf.Variable
            the output from the critic
        """
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # concatenate the observations and actions
            qf_h = tf.concat([obs, action], axis=-1)

            # create the hidden layers
            for i, layer_size in enumerate(self.layers):
                qf_h = self._layer(
                    qf_h,  layer_size, 'fc{}'.format(i),
                    act_fun=self.act_fun,
                    layer_norm=self.layer_norm
                )

            # create the output layer
            qvalue_fn = self._layer(
                qf_h, 1, 'qf_output',
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3)
            )

        return qvalue_fn

    def _setup_critic_updates_shared(self, actor_target):
        """TODO

        Parameters
        ----------
        actor_target : tf.Variable
            the output from the shared target actor

        Returns
        -------
        tf.Variable
            output from the centralized critic target with inputs from the
            actor target
        tf.Operation
            the operation that returns the loss of the critic
        tf.Operation
            the operation that updates the trainable parameters of the critic
        """
        pass  # TODO

    def _setup_critic_updates_nonshared(self, actor_targets):
        """TODO

        Parameters
        ----------
        actor_targets : list of tf.Variable
            the output from all target actors, indexed by their ordered IDs

        Returns
        -------
        dict <str, tf.Variable>
            output from the centralized critic target with inputs from the
            actor target
        dict <str, tf.Operation>
            the operation that returns the loss of the critic
        dict <str, tf.Operation>
            the operation that updates the trainable parameters of the critic
        """
        pass  # TODO

    def _setup_target_updates_shared(self, scope):
        """TODO

        Parameters
        ----------
        scope : str
            an upper-level scope term

        Returns
        -------
        tf.Operation
            an operation that sets the values of the trainable parameters of
            the target actor/critic to match those actual actor/critic
        tf.Operation
            soft target update function
        """
        pass  # TODO

    def _setup_target_updates_nonshared(self, scope):
        """TODO

        Parameters
        ----------
        scope : str
            an upper-level scope term

        Returns
        -------
        dict <str, tf.Operation>
            an operation that sets the values of the trainable parameters of
            the target actor/critic to match those actual actor/critic
        dict <str, tf.Operation>
            soft target update function
        """
        pass  # TODO

    def _setup_actor_updates_shared(self):
        """TODO

        Returns
        -------
        tf.Operation
            the operation that returns the loss of the actor
        tf.Operation
            the operation that updates the trainable parameters of the actor
        """
        pass  # TODO

    def _setup_actor_updates_nonshared(self):
        """TODO

        Returns
        -------
        dict <str, tf.Operation>
            the operation that returns the loss of the actor
        dict <str, tf.Operation>
            the operation that updates the trainable parameters of the actor
        """
        pass  # TODO

    def _setup_stats_shared(self, base):
        """TODO

        Parameters
        ----------
        base : str
            an upper-level scope term
        """
        pass  # TODO

    def _setup_stats_nonshared(self, base):
        """TODO

        Parameters
        ----------
        base : str
            an upper-level scope term
        """
        pass  # TODO

    def _initialize_maddpg(self):
        """See initialize.

        This method initializes the target parameters to match the model
        parameters.
        """
        self.sess.run(self.target_init_updates)

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
