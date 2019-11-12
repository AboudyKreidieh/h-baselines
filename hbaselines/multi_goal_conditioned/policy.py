"""Script containing multiagent variants of the policies."""
import tensorflow as tf

from hbaselines.goal_conditioned.policy import ActorCriticPolicy
from hbaselines.goal_conditioned.policy import FeedForwardPolicy


class MultiFeedForwardPolicy(ActorCriticPolicy):
    """Base Actor Critic Policy.

    Attributes
    ----------
    sess : tf.compat.v1.Session
        the current TensorFlow session
    ob_space : gym.space.*
        the observation space of the environment
    ac_space : gym.space.*
        the action space of the environment
    co_space : gym.space.*
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
    policy_parameters : dict
        dictionary of policy-specific parameters. Contains the following terms:

        * buffer_size (int): the max number of transitions to store
        * batch_size (int): SGD batch size
        * actor_lr (float): actor learning rate
        * critic_lr (float): critic learning rate
        * verbose (int): the verbosity level: 0 none, 1 training information, 2
          tensorflow debug
        * tau (float): target update rate
        * gamma (float): discount factor
        * noise (float): scaling term to the range of the action space, that is
          subsequently used as the standard deviation of Gaussian noise added
          to the action if `apply_noise` is set to True in `get_action`
        * target_policy_noise (float): standard deviation term to the noise
          from the output of the target actor policy. See TD3 paper for more.
        * target_noise_clip (float): clipping term for the noise injected in
          the target actor policy
        * layer_norm (bool): enable layer normalisation
        * layers (list of int): the size of the Neural network for the policy
        * act_fun (tf.nn.*): the activation function to use in the neural
          network
        * use_huber (bool): specifies whether to use the huber distance
          function as the loss for the critic. If set to False, the
          mean-squared error metric is used instead
    agents : FeedForwardPolicy or dict of FeedForwardPolicy
        Policy for each agent in the network. If the policies are shared, this
        attribute is simply one fully connected networks. Otherwise, it is a
        dictionary of networks whose keys are the names of the agent.
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
                 use_fingerprints=False,
                 zero_fingerprint=False):
        """Instantiate the multiagent feed-forward neural network policy.

        Parameters
        ----------
        sess : tf.compat.v1.Session
            the current TensorFlow session
        ob_space : gym.space.*
            the observation space of the environment
        ac_space : gym.space.*
            the action space of the environment
        co_space : gym.space.*
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
        """
        super(MultiFeedForwardPolicy, self).__init__(
            sess, ob_space, ac_space, co_space)

        self.shared = shared
        self.centralized_vfs = centralized_vfs
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
            zero_fingerprint=zero_fingerprint,
            reuse=False,
        )

        # Create the actor and critic networks for each agent.
        self.agents = {}
        if shared:
            # One policy shared by all agents.
            with tf.compat.v1.variable_scope("agent"):
                self.agents["agent"] = FeedForwardPolicy(
                    sess=self.sess,
                    ob_space=self.ob_space,
                    ac_space=self.ac_space,
                    co_space=self.co_space,
                    scope="agent",
                    **self.policy_parameters
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
                        **self.policy_parameters
                    )

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
        obs0 : array_like
            the last observation
        action : array_like
            the action
        reward : float
            the reward
        obs1 : array_like
            the current observation
        done : float
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

    def get_stats(self):
        """Return the model statistics.

        This data wil be stored in the training csv file.

        Returns
        -------
        dict
            model statistic
        """
        combines_stats = {}

        for key in self.agents.keys():
            # get the stats of the current agent
            combines_stats.update(self.agents[key].get_stats())

        return combines_stats

    def get_td_map(self):
        """Return dict map for the summary (to be run in the algorithm)."""
        combines_td_maps = {}

        for key in self.agents.keys():
            # get the td_map of the current agent
            combines_td_maps.update(self.agents[key].get_td_map())

        return combines_td_maps
