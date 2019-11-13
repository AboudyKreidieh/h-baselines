"""Script containing multiagent variants of the policies."""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from gym.spaces import Box

from hbaselines.goal_conditioned.policy import ActorCriticPolicy
from hbaselines.goal_conditioned.policy import FeedForwardPolicy
from hbaselines.multi_goal_conditioned.replay_buffer import MultiReplayBuffer
from hbaselines.utils.misc import get_manager_ac_space

# name of Flow environments. Used to assign appropriate Worker ob/ac/co spaces.
FLOW_ENV_NAMES = [
    "ringroad0",
    "ringroad1",
    "figureeight0",
    "figureeight1",
    "figureeight2",
    "merge0",
    "merge1",
    "merge2",
    "grid0",
    "grid1",
    "bottleneck0",
    "bottleneck1",
    "bottleneck2"
]


class MultiFeedForwardPolicy(ActorCriticPolicy):
    """Multi-agent fully connected neural network policy.

    TODO: description

    Attributes
    ----------
    sess : tf.compat.v1.Session
        the current TensorFlow session
    ob_space : gym.spaces.*
        the observation space of the environment
    ac_space : gym.spaces.*
        the action space of the environment
    co_space : gym.spaces.*
        the context space of the environment
    shared : bool
        whether to use a shared policy for all agents
    centralized_vfs : bool
        Whether to use a centralized value function for all agents. This is
        done in one of two ways:

        * If shared is set to True, the following technique is followed:
          https://arxiv.org/pdf/1705.08926.pdf
        * If shared is set to False, the following technique is followed:
          https://arxiv.org/pdf/1706.02275.pdf
    buffer_size : int
        the max number of transitions to store
    batch_size : int
        SGD batch size
    actor_lr : float
        actor learning rate
    critic_lr ; float
        critic learning rate
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    tau : float
        target update rate
    gamma : float
        discount factor
    noise : float
        scaling term to the range of the action space, that is subsequently
        used as the standard deviation of Gaussian noise added to the action if
        `apply_noise` is set to True in `get_action`
    target_policy_noise : float
        standard deviation term to the noise from the output of the target
        actor policy. See TD3 paper for more.
    target_noise_clip : float
        clipping term for the noise injected in the target actor policy
    layer_norm : bool
        enable layer normalisation
    layers : list of int
        the size of the Neural network for the policy
    act_fun : tf.nn.*
        the activation function to use in the neural network
    use_huber : bool
        specifies whether to use the huber distance function as the loss for
        the critic. If set to False, the mean-squared error metric is used
        instead
    replay_buffer : MultiReplayBuffer
        a centralized replay buffer object. Used only when `centralized_vfs` is
        set to True.
    agents : dict of FeedForwardPolicy
        Policy for each agent in the network.
    central_q : TODO
        TODO
    optimizer : TODO
        TODO
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
                 noise,
                 target_policy_noise,
                 target_noise_clip,
                 layer_norm,
                 layers,
                 act_fun,
                 use_huber,
                 shared,
                 centralized_vfs,
                 all_ob_space,
                 use_fingerprints=False,
                 zero_fingerprint=False):
        """Instantiate the multiagent feed-forward neural network policy.

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
        noise : float
            scaling term to the range of the action space, that is subsequently
            used as the standard deviation of Gaussian noise added to the
            action if `apply_noise` is set to True in `get_action`
        target_policy_noise : float
            standard deviation term to the noise from the output of the target
            actor policy. See TD3 paper for more.
        target_noise_clip : float
            clipping term for the noise injected in the target actor policy
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
        shared : bool
            whether to use a shared policy for all agents
        centralized_vfs : bool
            Whether to use a centralized value function for all agents. This is
            done in one of two ways:

            * If shared is set to True, the following technique is followed:
              https://arxiv.org/pdf/1705.08926.pdf
            * If shared is set to False, the following technique is followed:
              https://arxiv.org/pdf/1706.02275.pdf
        all_ob_space : gym.spaces.*
            the observation space of the full state space. Used for centralized
            value functions
        """
        super(MultiFeedForwardPolicy, self).__init__(
            sess, ob_space, ac_space, co_space)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.verbose = verbose
        self.tau = tau
        self.gamma = gamma
        self.noise = noise
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.layer_norm = layer_norm
        self.layers = layers
        self.act_fun = act_fun
        self.use_huber = use_huber
        self.use_fingerprints = use_fingerprints
        self.zero_fingerprint = zero_fingerprint
        self.shared = shared
        self.centralized_vfs = centralized_vfs

        # variables that are defined by the _setup* procedures
        self.replay_buffer = None
        self.agents = None
        self.central_q = None
        self.optimizer = None

        # Setup the agents and the necessary objects and operations needed to
        # support the training procedure.
        if centralized_vfs:
            self._setup_centralized_vfs(buffer_size, batch_size, all_ob_space)
        else:
            self._setup_independent_learners()

    def _setup_independent_learners(self):
        """Setup independent learners.

        In this case, the policy consists of separate (or shared) policies for
        the individual agents that are subsequently trained in a decentralized
        manner.

        Then agents in this case are created using the `FeedForwardPolicy`
        class. No separate replay buffers, centralized value functions, or
        optimization operations are created.
        """
        policy_parameters = dict(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            verbose=self.verbose,
            tau=self.tau,
            gamma=self.gamma,
            noise=self.noise,
            target_policy_noise=self.target_policy_noise,
            target_noise_clip=self.target_noise_clip,
            layer_norm=self.layer_norm,
            layers=self.layers,
            act_fun=self.act_fun,
            use_huber=self.use_huber,
            use_fingerprints=self.use_fingerprints,
            zero_fingerprint=self.zero_fingerprint,
            reuse=False
        )

        # Create the actor and critic networks for each agent.
        self.agents = {}
        if self.shared:
            # One policy shared by all agents.
            with tf.compat.v1.variable_scope("agent"):
                self.agents["agent"] = FeedForwardPolicy(
                    sess=self.sess,
                    ob_space=self.ob_space,
                    ac_space=self.ac_space,
                    co_space=self.co_space,
                    scope="agent",
                    **policy_parameters
                )
        else:
            # Each agent requires a new feed-forward policy.
            for key in self.ob_space.keys():
                with tf.compat.v1.variable_scope(key):
                    self.agents[key] = FeedForwardPolicy(
                        sess=self.sess,
                        ob_space=self.ob_space[key],
                        ac_space=self.ac_space[key],
                        co_space=self.co_space[key],
                        scope=key,
                        **policy_parameters
                    )

    def _setup_centralized_vfs(self, buffer_size, batch_size, all_ob_space):
        """Setup centralized value function variants of the policy.

        This method creates a new replay buffer object to store full state
        observation samples that can be used when training a centralized value
        functions.

        If the agents utilize a shared policy, the COMA [1] algorithm is used
        to train the policy; otherwise, a TD3 variant of the MADDPG [2]
        algorithm is used.

        [1] Foerster, Jakob N., et al. "Counterfactual multi-agent policy
            gradients." Thirty-Second AAAI Conference on Artificial
            Intelligence. 2018.
        [2] Lowe, Ryan, et al. "Multi-agent actor-critic for mixed cooperative-
            competitive environments." Advances in Neural Information
            Processing Systems. 2017.

        Parameters
        ----------
        buffer_size : int
            the max number of transitions to store
        batch_size : int
            SGD batch size
        all_ob_space : gym.spaces.*
            the observation space of the full state space. Used for centralized
            value functions
        """
        # Compute the observation dimensions, taking into account the possible
        # use of contextual spaces as well.
        obs_dim = [self.ob_space[key].shape[0]
                   for key in sorted(self.ob_space.keys())]
        if self.co_space is not None:
            obs_dim = [obs_dim_i + self.co_space[key].shape[0]
                       for obs_dim_i, key
                       in zip(obs_dim, sorted(self.co_space.keys()))]

        # Create the replay buffer object if centralized value functions are
        # used. This is to allow the policy to stored relative and full state
        # information for training the actor and critic networks, respectively.
        self.replay_buffer = MultiReplayBuffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            obs_dim=obs_dim,
            ac_dim=[self.ac_space[key].shape[0]
                    for key in sorted(self.ac_space.keys())],
            all_obs_dim=all_ob_space.shape[0],
        )

        # Create the agents and optimization scheme in TensorFlow.
        if self.shared:
            self._setup_coma(all_ob_space)
        else:
            self._setup_maddpg(all_ob_space)

    def _setup_coma(self, all_ob_space):
        """Setup TD3 variant of Counterfactual Multi-Agent Policy Gradients.

        See: https://arxiv.org/pdf/1705.08926.pdf
        """
        with tf.compat.v1.variable_scope("cvf"):
            # Create the relevant input placeholders
            with tf.compat.v1.variable_scope("input"):
                self.obs0_ph = []
                self.ac_ph = []
                for i, agent_id in enumerate(sorted(self.ob_space.keys())):
                    self.obs0_ph.append(tf.compat.v1.placeholder(
                        (None,) + self.ob_space[i].shape,
                        dtype=tf.float32))
                    self.ac_ph.append(tf.compat.v1.placeholder(
                        (None,) + self.ac_space[i].shape,
                        dtype=tf.float32))
                self.all_obs0_ph = tf.compat.v1.placeholder(
                    (None,) + all_ob_space.shape,
                    dtype=tf.float32)
                self.all_obs1_ph = tf.compat.v1.placeholder(
                    (None,) + all_ob_space.shape,
                    dtype=tf.float32)

            with tf.compat.v1.variable_scope("model"):
                # Create the agent policies.
                self.agents = [
                    self._make_actor(obs0, self.ac_space[i])
                    for i, obs0 in enumerate(self.obs0_ph)
                ]

                # Create the centralized Q-functions.
                self.central_q1 = self._make_centralized_critic(
                    self.all_obs0_ph,
                    self.ac_ph,
                    scope="qf1"
                )
                self.central_q2 = self._make_centralized_critic(
                    self.all_obs0_ph,
                    self.ac_ph,
                    scope="qf2"
                )

                # Create the centralized Q-function with inputs from the
                # actor policies of the agents. FIXME
                central_q_with_actors = self._make_centralized_critic(
                    self.all_obs0_ph,
                    self.agents,
                    scope="qf1",
                    reuse=True
                )

            with tf.compat.v1.variable_scope("target"):
                # Create the centralized target Q-function.
                pass  # TODO

                # Create the centralized Q-function target update procedure.
                pass  # TODO

            with tf.compat.v1.variable_scope("train"):
                # Create the centralized loss function.
                pass  # TODO

                # Create the optimizer object.
                pass  # TODO

                # Create the tensorflow operation for computing and applying
                # the gradients to the actor policy.
                pass  # TODO

    def _setup_maddpg(self, all_ob_space):
        """Setup TD3-variant of MADDPG.

        See: https://arxiv.org/pdf/1706.02275.pdf
        """
        pass  # TODO

    def _make_actor(self, obs, ac_space, reuse=False, scope="pi"):
        """Create an actor tensor.

        Parameters
        ----------
        obs : tf.compat.v1.placeholder
            the input observation placeholder
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
                pi_h = tf.layers.dense(
                    pi_h,
                    layer_size,
                    name='fc' + str(i),
                    kernel_initializer=slim.variance_scaling_initializer(
                        factor=1.0 / 3.0, mode='FAN_IN', uniform=True))
                if self.layer_norm:
                    pi_h = tf.contrib.layers.layer_norm(
                        pi_h, center=True, scale=True)
                pi_h = self.act_fun(pi_h)

            # create the output layer
            policy = tf.nn.tanh(tf.layers.dense(
                pi_h,
                ac_space.shape[0],
                name='output',
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3)))

            # scaling terms to the output from the policy
            ac_means = (ac_space.high + ac_space.low) / 2.
            ac_magnitudes = (ac_space.high - ac_space.low) / 2.

            policy = ac_means + ac_magnitudes * policy

        return policy

    def _make_centralized_critic(self, obs, action, reuse=False, scope="qf"):
        """Create a critic tensor.

        Parameters
        ----------
        obs : tf.compat.v1.placeholder
            the input observation placeholder
        action : list of tf.compat.v1.placeholder
            the input action placeholder for each agent
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
            qf_h = tf.concat([obs, action], axis=-1)  # FIXME

            # create the hidden layers
            for i, layer_size in enumerate(self.layers):
                qf_h = tf.layers.dense(
                    qf_h,
                    layer_size,
                    name='fc' + str(i),
                    kernel_initializer=slim.variance_scaling_initializer(
                        factor=1.0 / 3.0, mode='FAN_IN', uniform=True))
                if self.layer_norm:
                    qf_h = tf.contrib.layers.layer_norm(
                        qf_h, center=True, scale=True)
                qf_h = self.act_fun(qf_h)

            # create the output layer
            qvalue_fn = tf.layers.dense(
                qf_h,
                1,  # TODO: n_agents len(action)?
                name='qf_output',
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3))

        return qvalue_fn

    def initialize(self):
        """Initialize the policy.

        This is used at the beginning of training by the algorithm, after the
        model parameters have been initialized.
        """
        for key in self.agents.keys():
            self.agents[key].initialize()

    def update(self, update_actor=True, **kwargs):
        """Perform a gradient update step.

        Parameters
        ----------
        update_actor : bool
            specifies whether to update the actor policy. The critic policy is
            still updated if this value is set to False.

        Returns
        -------
        float
            critic loss
        float
            actor loss
        """
        if self.centralized_vfs:
            pass
        else:
            # Perform independent learners training procedure.
            for key in self.agents.keys():
                self.agents[key].update(update_actor, **kwargs)

    def get_action(self, obs, apply_noise, random_actions, **kwargs):
        """Call the actor methods to compute policy actions.

        Parameters
        ----------
        obs : dict of array_like
            the observations, with each element corresponding to a unique agent
            (as defined by the key)
        apply_noise : bool
            whether to add Gaussian noise to the output of the actor. Defaults
            to False
        random_actions : bool
            if set to True, actions are sampled randomly from the action space
            instead of being computed by the policy. This is used for
            exploration purposes.

        Returns
        -------
        dict array_like
            computed action by the policy for each observation. The output key
            match the input keys
        """
        actions = {}

        for key in obs.keys():
            # Use the same policy for all operations if shared, and the
            # corresponding policy otherwise.
            agent = self.agents["agent"] if self.shared else self.agents[key]

            # Compute the action of the provided observation.
            actions[key] = agent.get_action(
                obs[key], apply_noise, random_actions, **kwargs)

        return actions

    def value(self, obs, action=None, **kwargs):
        """Call the critic methods to compute the value.

        Parameters
        ----------
        obs : array_like
            the observation
        action : array_like, optional
            the actions performed in the given observation

        Returns
        -------
        array_like
            computed value by the critic
        """
        values = {}

        for key in obs.keys():
            # Use the same policy for all operations if shared, and the
            # corresponding policy otherwise.
            agent = self.agents["agent"] if self.shared else self.agents[key]

            # Compute the value of the provided observation.
            values[key] = agent.value(obs[key], action, **kwargs)

        return values

    def store_transition(self, obs0, action, reward, obs1, done, **kwargs):
        """Store a transition in the replay buffer.

        Parameters
        ----------
        obs0 : dict of array_like
            the last observation
        action : dict of array_like
            the dict of action
        reward : dict of float
            the reward
        obs1 : dict of array_like
            the current observation
        done : dict of float
            is the episode done
        """
        for key in obs0.keys():
            # Use the same policy for all operations if shared, and the
            # corresponding policy otherwise.
            agent = self.agents["agent"] if self.shared else self.agents[key]

            # Store the individual samples.
            agent.store_transition(
                obs0=obs0[key],
                action=action[key],
                reward=reward[key],
                obs1=obs1[key],
                done=done[key],
                **kwargs
            )

    def get_td_map(self):
        """Return dict map for the summary (to be run in the algorithm)."""
        combines_td_maps = {}

        for key in self.agents.keys():
            # get the td_map of the current agent
            combines_td_maps.update(self.agents[key].get_td_map())

        return combines_td_maps


class MultiGoalConditionedPolicy(ActorCriticPolicy):
    """Multi-agent variant of the goal-conditioned policy.

    Instead of a single agent worker, the worker within this policy is replaced
    by a set of policies (or single shared policy) that controls the
    decentralized workers.

    Attributes
    ----------
    TODO
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
                 noise,
                 target_policy_noise,
                 target_noise_clip,
                 layer_norm,
                 layers,
                 act_fun,
                 use_huber,
                 shared,
                 centralized_worker_vfs,
                 meta_period,
                 relative_goals,
                 off_policy_corrections,
                 use_fingerprints,
                 fingerprint_range,
                 centralized_manager_vfs,
                 connected_gradients,
                 cg_weights,
                 env_name=""):
        """Instantiate the multi-agent goal-conditioned hierarchical policy.

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
        noise : float
            scaling term to the range of the action space, that is subsequently
            used as the standard deviation of Gaussian noise added to the
            action if `apply_noise` is set to True in `get_action`.
        target_policy_noise : float
            standard deviation term to the noise from the output of the target
            actor policy. See TD3 paper for more.
        target_noise_clip : float
            clipping term for the noise injected in the target actor policy
        layer_norm : bool
            enable layer normalisation
        layers : list of int or None
            the size of the neural network for the policy
        act_fun : tf.nn.*
            the activation function to use in the neural network
        use_huber : bool
            specifies whether to use the huber distance function as the loss
            for the critic. If set to False, the mean-squared error metric is
            used instead
        shared : bool
            whether to use a shared policy for all agents
        centralized_worker_vfs : bool
            Whether to use a centralized value function for all agents. This is
            done in one of two ways:

            * If shared is set to True, the following technique is followed:
              https://arxiv.org/pdf/1705.08926.pdf
            * If shared is set to False, the following technique is followed:
              https://arxiv.org/pdf/1706.02275.pdf
        meta_period : int
            manger action period
        relative_goals : bool
            specifies whether the goal issued by the Manager is meant to be a
            relative or absolute goal, i.e. specific state or change in state
        off_policy_corrections : bool
            whether to use off-policy corrections during the update procedure.
            See: https://arxiv.org/abs/1805.08296
        use_fingerprints : bool
            specifies whether to add a time-dependent fingerprint to the
            observations
        fingerprint_range : (list of float, list of float)
            the low and high values for each fingerprint element, if they are
            being used
        centralized_manager_vfs : bool
            specifies whether to use centralized value functions for the
            Manager critic function
        connected_gradients : bool
            whether to connect the graph between the manager and worker
        cg_weights : float
            weights for the gradients of the loss of the worker with respect to
            the parameters of the manager. Only used if `connected_gradients`
            is set to True.
        """
        super(MultiGoalConditionedPolicy, self).__init__(
            sess, ob_space, ac_space, co_space)

        self.shared = shared
        self.centralized_worker_vfs = centralized_worker_vfs
        self.meta_period = meta_period
        self.relative_goals = relative_goals
        self.off_policy_corrections = off_policy_corrections
        self.use_fingerprints = use_fingerprints
        self.fingerprint_range = fingerprint_range
        self.fingerprint_dim = (len(self.fingerprint_range[0]),)
        self.centralized_manager_vfs = centralized_manager_vfs
        self.connected_gradients = connected_gradients
        self.cg_weights = cg_weights
        self.policy_parameters = dict(
            buffer_size=buffer_size,
            batch_size=batch_size,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            verbose=verbose,
            tau=tau,
            gamma=gamma,
            noise=noise,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            layer_norm=layer_norm,
            layers=layers,
            act_fun=act_fun,
            use_huber=use_huber,
            use_fingerprints=use_fingerprints,
            reuse=False,
        )

        # Create the replay buffer.

        # =================================================================== #
        # Part 1. Setup the Manager                                           #
        # =================================================================== #

        # Get the Manager's action space.
        manager_ac_space = get_manager_ac_space(
            ob_space, relative_goals, env_name,
            use_fingerprints, self.fingerprint_dim)

        # Create the Manager policy.
        with tf.compat.v1.variable_scope("Manager"):
            self.manager = FeedForwardPolicy(
                sess=self.sess,
                ob_space=ob_space,
                ac_space=manager_ac_space,
                co_space=co_space,
                **self.policy_parameters
            )

        # =================================================================== #
        # Part 2. Setup the Worker.                                           #
        # =================================================================== #

        # Get the Worker's observation, context, and action space.
        worker_ob_space, worker_ac_space, worker_co_space = \
            self._get_worker_spaces(ob_space, ac_space, co_space,
                                    manager_ac_space, shared, env_name)

        # Create the Worker policy.
        with tf.compat.v1.variable_scope("Worker"):
            self.worker = MultiFeedForwardPolicy(
                sess=self.sess,
                ob_space=worker_ob_space,
                ac_space=worker_ac_space,
                co_space=worker_co_space,
                shared=self.shared,
                centralized_vfs=self.centralized_worker_vfs,
                **self.policy_parameters
            )

    @staticmethod
    def _get_worker_spaces(ob_space,
                           ac_space,
                           co_space,
                           manager_ac_space,
                           shared,
                           env_name):
        """Get the Worker's observation, context, and action space.

        Parameters
        ----------
        ob_space : gym.spaces.*
            the observation space of the environment
        ac_space : gym.spaces.*
            the action space of the environment
        co_space : gym.spaces.*
            the context space of the environment
        manager_ac_space : gym.spaces.*
            the action space of the Manager policy
        shared : bool
            whether to use a shared policy for all agents
        env_name : str
            the name of the environment

        Returns
        -------
        gym.spaces.Box or dict of gym.spaces.Box
            observation space(s) of the worker agents
        gym.spaces.Box or dict of gym.spaces.Box
            action space(s) of the worker agents
        gym.spaces.Box or dict of gym.spaces.Box
            context space(s) of the worker agents
        """
        if env_name in FLOW_ENV_NAMES:
            if "figureeight" in env_name or "merge" in env_name:
                # The high and low values are the same across all actions,
                # since they are all similar agents
                manager_ac_high = manager_ac_space.high[0]
                manager_ac_low = manager_ac_space.low[0]
                worker_ac_high = ac_space.high[0]
                worker_ac_low = ac_space.low[0]

                # Create the action, observation, and context spaces.
                worker_ob_space = Box(low=0, high=1, shape=(5,))
                worker_ac_space = Box(low=worker_ac_low, high=worker_ac_high,
                                      shape=(1,))
                worker_co_space = Box(low=manager_ac_low, high=manager_ac_high,
                                      shape=(1,))

                # Create a dictionary of these spaces for all agents if the
                # policy is not shared.
                if not shared:
                    num_agents = ac_space.shape[0]
                    worker_ob_space = {"agent{}".format(i): worker_ob_space
                                       for i in range(num_agents)}
                    worker_ac_space = {"agent{}".format(i): worker_ac_space
                                       for i in range(num_agents)}
                    worker_co_space = {"agent{}".format(i): worker_co_space
                                       for i in range(num_agents)}
            else:
                # FIXME: other cases
                worker_ob_space = ob_space
                worker_ac_space = ac_space
                worker_co_space = co_space
        else:
            worker_ob_space = ob_space
            worker_ac_space = ac_space
            worker_co_space = co_space

        return worker_ob_space, worker_ac_space, worker_co_space
