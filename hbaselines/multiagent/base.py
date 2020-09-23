"""Multi-agent base actor critic policy."""
import tensorflow as tf

from hbaselines.base_policies import ActorCriticPolicy


class MultiActorCriticPolicy(ActorCriticPolicy):
    """Multi-agent base actor critic policy.

    This policy supports training off-policy variants of three popular
    multi-agent algorithms:

    * Independent learners: Independent (or Naive) learners provide a separate
      policy with independent parameters to each agent in an environment.
      Within this setting, agents are provided separate observations and reward
      signals, and store their samples and perform updates separately. A review
      of independent learners in reinforcement learning can be found here:
      https://hal.archives-ouvertes.fr/hal-00720669/document

      To train a policy using independent learners, do not modify any
      policy-specific attributes:

      >>> from hbaselines.algorithms import RLAlgorithm
      >>>
      >>> alg = RLAlgorithm(
      >>>     policy=MultiActorCriticPolicy,
      >>>     env="...",  # replace with an appropriate environment
      >>>     policy_kwargs={}
      >>> )

    * Shared policies: Unlike the independent learners formulation, shared
      policies utilize a single policy with shared parameters for all agents
      within the network. Moreover, the samples experienced by all agents are
      stored within one unified replay buffer. See the following link for an
      early review of the benefit of shared policies:
      https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.55.8066&rep=rep1&type=pdf

      To train a policy using the shared policy feature, set the `shared`
      attribute to True:

      >>> from hbaselines.algorithms import RLAlgorithm
      >>>
      >>> alg = RLAlgorithm(
      >>>     policy=MultiActorCriticPolicy,
      >>>     env="...",  # replace with an appropriate environment
      >>>     policy_kwargs={
      >>>         "shared": True,
      >>>     }
      >>> )

    * MADDPG: We implement algorithmic-variants of MAPPG for all supported
      off-policy RL algorithms. See: https://arxiv.org/pdf/1706.02275.pdf

      To train a policy using their MADDPG variants as opposed to independent
      learners, algorithm, set the `maddpg` attribute to True:

      >>> from hbaselines.algorithms import RLAlgorithm
      >>>
      >>> alg = RLAlgorithm(
      >>>     policy=MultiActorCriticPolicy,
      >>>     env="...",  # replace with an appropriate environment
      >>>     policy_kwargs={
      >>>         "maddpg": True,
      >>>         "shared": False,  # or True
      >>>     }
      >>> )

      This works for both shared and non-shared policies. For shared policies,
      we use a single centralized value function instead of a value function
      for each agent.

    Attributes
    ----------
    shared : bool
        whether to use a shared policy for all agents
    maddpg : bool
        whether to use an algorithm-specific variant of the MADDPG algorithm
    all_ob_space : gym.spaces.*
        the observation space of the full state space. Used by MADDPG variants
        of the policy.
    n_agents : int
        the number of agents in the networks. This is needed if using MADDPG
        with a shared policy to compute the length of the full action space.
        Otherwise, it is not used.
    base_policy : type [ hbaselines.base_policies.ActorCriticPolicy ]
        the base (single agent) policy model used by all agents within the
        network
    additional_params : dict
        additional algorithm-specific policy parameters. Used internally by the
        class when instantiating other (child) policies.
    agents : dict <str, hbaselines.base_policies.ActorCriticPolicy>
        Actor policy for each agent in the network. If MADDPG variants of the
        policy are being used, this attribute is not used.
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
                 use_huber,
                 model_params,
                 shared,
                 maddpg,
                 base_policy,
                 all_ob_space=None,
                 n_agents=1,
                 additional_params=None,
                 scope=None):
        """Instantiate the base multi-agent actor critic policy.

        Parameters
        ----------
        sess : tf.compat.v1.Session
            the current TensorFlow session
        ob_space : gym.spaces.* or dict <str, gym.spaces.*>
            the observation space of individual agents in the environment. If
            not a dictionary, the observation space is shared across all agents
        ac_space : gym.spaces.* or dict <str, gym.spaces.*>
            the action space of individual agents in the environment. If
            not a dictionary, the action space is shared across all agents.
        co_space : gym.spaces.* or dict <str, gym.spaces.*>
            the context space of individual agents in the environment. If
            not a dictionary, the context space is shared across all agents.
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
        use_huber : bool
            specifies whether to use the huber distance function as the loss
            for the critic. If set to False, the mean-squared error metric is
            used instead
        model_params : dict
            dictionary of model-specific parameters. See parent class.
        shared : bool
            whether to use a shared policy for all agents
        maddpg : bool
            whether to use an algorithm-specific variant of the MADDPG
            algorithm
        base_policy : type [ hbaselines.base_policies.ActorCriticPolicy ]
            the base (single agent) policy model used by all agents within the
            network
        all_ob_space : gym.spaces.*
            the observation space of the full state space. Used by MADDPG
            variants of the policy.
        n_agents : int
            the number of agents in the networks. This is needed if using
            MADDPG with a shared policy to compute the length of the full
            action space. Otherwise, it is not used.
        additional_params : dict
            additional algorithm-specific policy parameters. Used internally by
            the class when instantiating other (child) policies.
        """
        # In case no context space was passed and not using shared policies,
        # create a dictionary of no contexts.
        if co_space is None and not shared:
            co_space = {key: None for key in ob_space.keys()}

        super(MultiActorCriticPolicy, self).__init__(
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
            use_huber=use_huber,
            model_params=model_params,
        )

        self.shared = shared
        self.maddpg = maddpg
        self.all_ob_space = all_ob_space
        self.n_agents = n_agents
        self.base_policy = base_policy
        self.additional_params = additional_params or {}

        # Setup the agents and the necessary objects and operations needed to
        # support the training procedure.
        if maddpg:
            self._setup_maddpg(scope)
        else:
            self._setup_basic(scope)

    def initialize(self):
        """Initialize the policy.

        This is used at the beginning of training by the algorithm, after the
        model parameters have been initialized.
        """
        if self.maddpg:
            self._initialize_maddpg()
        else:
            self._initialize_basic()

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
        if self.maddpg:
            return self._update_maddpg(update_actor, **kwargs)
        else:
            return self._update_basic(update_actor, **kwargs)

    def get_action(self, obs, context, apply_noise, random_actions, env_num=0):
        """Call the actor methods to compute policy actions.

        Parameters
        ----------
        obs : dict of array_like
            the observations, with each element corresponding to a unique agent
            (as defined by the key)
        context : array_like or None
            the contextual term for each agent. Set to None if no context is
            provided by the environment.
        apply_noise : bool
            whether to add Gaussian noise to the output of the actor. Defaults
            to False
        random_actions : bool
            if set to True, actions are sampled randomly from the action space
            instead of being computed by the policy. This is used for
            exploration purposes.
        env_num : int
            the environment number. Used to handle situations when multiple
            parallel environments are being used.

        Returns
        -------
        dict of array_like
            computed action by the policy for each observation. The output key
            match the input keys
        """
        if self.maddpg:
            return self._get_action_maddpg(
                obs=obs,
                context=context,
                apply_noise=apply_noise,
                random_actions=random_actions,
                env_num=env_num,
            )
        else:
            return self._get_action_basic(
                obs=obs,
                context=context,
                apply_noise=apply_noise,
                random_actions=random_actions,
                env_num=env_num,
            )

    def store_transition(self,
                         obs0,
                         context0,
                         action,
                         reward,
                         obs1,
                         context1,
                         done,
                         is_final_step,
                         all_obs0=None,
                         all_obs1=None,
                         env_num=0,
                         evaluate=False):
        """Store a transition in the replay buffer.

        Parameters
        ----------
        obs0 : dict of array_like
            the last observation for each agent
        context0 : array_like or None
            the last contextual term for each agent. Set to None if no context
            is provided by the environment.
        action : dict of array_like
            the dict of action for each agent
        reward : float ro dict of float
            the reward for each agent. A single reward if the policy is shared.
        obs1 : dict of array_like
            the current observation for each agent
        context1 : array_like or None
            the current contextual term for each agent. Set to None if no
            context is provided by the environment.
        done : float or dict of float
            is the episode done for each agent
        is_final_step : bool
            whether the time horizon was met in the step corresponding to the
            current sample. This is used by the TD3 algorithm to augment the
            done mask.
        all_obs0 : array_like
            the last full-state observation
        all_obs1 : array_like
            the current full-state observation
        env_num : int
            the environment number. Used to handle situations when multiple
            parallel environments are being used.
        evaluate : bool
            whether the sample is being provided by the evaluation environment.
            If so, the data is not stored in the replay buffer.
        """
        if self.maddpg:
            self._store_transition_maddpg(
                obs0=obs0,
                context0=context0,
                action=action,
                reward=reward,
                obs1=obs1,
                context1=context1,
                done=done,
                is_final_step=is_final_step,
                all_obs0=all_obs0,
                all_obs1=all_obs1,
                env_num=env_num,
                evaluate=evaluate,
            )
        else:
            self._store_transition_basic(
                obs0=obs0,
                context0=context0,
                action=action,
                reward=reward,
                obs1=obs1,
                context1=context1,
                done=done,
                is_final_step=is_final_step,
                env_num=env_num,
                evaluate=evaluate,
            )

    def get_td_map(self):
        """Return dict map for the summary (to be run in the algorithm)."""
        if self.maddpg:
            return self._get_td_map_maddpg()
        else:
            return self._get_td_map_basic()

    # ======================================================================= #
    #               Basic version of required abstract methods.               #
    # ======================================================================= #

    def _setup_basic(self, scope):
        """Create basic independent learners / shared policy components.

        In this case, the policy consists of separate (or shared) policies for
        the individual agents that are subsequently trained in a decentralized
        manner.

        Then agents in this case are created using the base_policy class. No
        separate replay buffers, centralized value functions, or optimization
        operations are created.
        """
        policy_parameters = dict(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            verbose=self.verbose,
            tau=self.tau,
            gamma=self.gamma,
            use_huber=self.use_huber,
            model_params=self.model_params,
            **self.additional_params
        )

        # Create the actor and critic networks for each agent.
        self.agents = {}
        if self.shared:
            # One policy shared by all agents.
            self.agents["policy"] = self.base_policy(
                sess=self.sess,
                ob_space=self.ob_space,
                ac_space=self.ac_space,
                co_space=self.co_space,
                scope=scope,
                **policy_parameters
            )
        else:
            for key in self.ob_space.keys():
                # Add the outer scope if provided.
                scope_i = key if scope is None else "{}/{}".format(scope, key)

                # Each agent requires a new feed-forward policy.
                with tf.compat.v1.variable_scope(key):
                    self.agents[key] = self.base_policy(
                        sess=self.sess,
                        ob_space=self.ob_space[key],
                        ac_space=self.ac_space[key],
                        co_space=self.co_space[key],
                        scope=scope_i,
                        **policy_parameters
                    )

    def _initialize_basic(self):
        """See initialize."""
        for key in self.agents.keys():
            self.agents[key].initialize()

    def _update_basic(self, update_actor=True, **kwargs):
        """See update."""
        actor_loss = {}
        critic_loss = {}
        for key in self.agents.keys():
            c, a = self.agents[key].update(update_actor=update_actor, **kwargs)
            critic_loss[key] = c
            actor_loss[key] = a

        return critic_loss, actor_loss

    def _get_action_basic(self,
                          obs,
                          context,
                          apply_noise,
                          random_actions,
                          env_num):
        """See get_action."""
        actions = {}

        for i, key in enumerate(self._sorted_list(obs.keys())):
            # Use the same policy for all operations if shared, and the
            # corresponding policy otherwise.
            agent = self.agents["policy"] if self.shared else self.agents[key]
            env_num_i = self.n_agents * env_num + i if self.shared else env_num

            # Get the contextual term. This accounts for cases when the context
            # is set to None.
            context_i = context if context is None else context[key]

            # Compute the action of the provided observation.
            actions[key] = agent.get_action(
                obs=obs[key],
                context=context_i,
                apply_noise=apply_noise,
                random_actions=random_actions,
                env_num=env_num_i,
            )

        return actions

    def _store_transition_basic(self,
                                obs0,
                                context0,
                                action,
                                reward,
                                obs1,
                                context1,
                                done,
                                is_final_step,
                                env_num,
                                evaluate):
        """See store_transition."""
        for i, key in enumerate(self._sorted_list(obs0.keys())):
            # If the agent has exited the environment, ignore it.
            if key not in obs1.keys():
                continue

            # Use the same policy for all operations if shared, and the
            # corresponding policy otherwise.
            agent = self.agents["policy"] if self.shared else self.agents[key]
            env_num_i = self.n_agents * env_num + i if self.shared else env_num

            # Get the contextual term. This accounts for cases when the context
            # is set to None.
            context0_i = context0 if context0 is None else context0[key]
            context1_i = context0 if context1 is None else context1[key]

            # Store the individual samples.
            agent.store_transition(
                obs0=obs0[key],
                context0=context0_i,
                action=action[key],
                reward=reward[key],
                obs1=obs1[key],
                context1=context1_i,
                done=False,
                is_final_step=is_final_step,
                evaluate=evaluate,
                env_num=env_num_i,
            )

    def _get_td_map_basic(self):
        """See get_td_map."""
        combines_td_maps = {}

        for key in self.agents.keys():
            # get the td_map of the current agent
            combines_td_maps.update(self.agents[key].get_td_map())

        return combines_td_maps

    @staticmethod
    def _sorted_list(keys):
        """Return a sorted list of dict keys."""
        return sorted(list(keys))

    # ======================================================================= #
    #               MADDPG version of required abstract methods.              #
    #                  Filled in by the specific algorithms.                  #
    # ======================================================================= #

    def _setup_maddpg(self, scope):
        """Create algorithmic-variant of MADDPG components.

        See: https://arxiv.org/pdf/1706.02275.pdf
        """
        raise NotImplementedError

    def _initialize_maddpg(self):
        """See initialize."""
        raise NotImplementedError

    def _update_maddpg(self, update_actor=True, **kwargs):
        """See update."""
        raise NotImplementedError

    def _get_action_maddpg(self,
                           obs,
                           context,
                           apply_noise,
                           random_actions,
                           env_num):
        """See get_action."""
        raise NotImplementedError

    def _store_transition_maddpg(self,
                                 obs0,
                                 context0,
                                 action,
                                 reward,
                                 obs1,
                                 context1,
                                 done,
                                 is_final_step,
                                 all_obs0,
                                 all_obs1,
                                 env_num,
                                 evaluate):
        """See store_transition."""
        raise NotImplementedError

    def _get_td_map_maddpg(self):
        """See get_td_map."""
        raise NotImplementedError
