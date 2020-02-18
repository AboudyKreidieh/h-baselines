"""TD3-compatible multi-agent feedforward policy."""
import tensorflow as tf
import numpy as np
from functools import reduce

from hbaselines.fcnet.td3 import FeedForwardPolicy
from hbaselines.multi_fcnet.base import MultiFeedForwardPolicy as BasePolicy
from hbaselines.multi_fcnet.replay_buffer import MultiReplayBuffer
from hbaselines.multi_fcnet.replay_buffer import SharedReplayBuffer
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.utils.tf_util import reduce_std


class MultiFeedForwardPolicy(BasePolicy):
    """TD3-compatible multi-agent feedforward neural.

    The attributes described in this docstring are only used if the `maddpg`
    parameter is set to True. The attributes are dictionaries of their
    described form for each agent if `shared` is set to False.

    See the docstring of the parent class for a further description of this
    class.

    Attributes
    ----------
    noise : float
        scaling term to the range of the action space, that is subsequently
        used as the standard deviation of Gaussian noise added to the action if
        `apply_noise` is set to True in `get_action`
    target_policy_noise : float
        standard deviation term to the noise from the output of the target
        actor policy. See TD3 paper for more.
    target_noise_clip : float
        clipping term for the noise injected in the target actor policy
    replay_buffer : MultiReplayBuffer or SharedReplayBuffer
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
    actor_target : tf.Variable
        the output from a noisy version of the target actor network
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
        if shared:
            # action magnitudes
            ac_mag = 0.5 * (ac_space.high - ac_space.low)

            self.noise = noise * ac_mag
            self.target_policy_noise = np.array([ac_mag * target_policy_noise])
            self.target_noise_clip = np.array([ac_mag * target_noise_clip])
        else:
            self.noise = {}
            self.target_policy_noise = {}
            self.target_noise_clip = {}
            for key in ac_space.keys():
                # action magnitudes
                ac_mag = 0.5 * (ac_space[key].high - ac_space[key].low)

                self.noise[key] = noise * ac_mag
                self.target_policy_noise[key] = \
                    np.array([ac_mag * target_policy_noise])
                self.target_noise_clip[key] = \
                    np.array([ac_mag * target_noise_clip])

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
        self.actor_target = None
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
        if self.shared:
            self._setup_maddpg_shared(scope)
        else:
            self._setup_maddpg_independent(scope)

    def _setup_maddpg_shared(self, scope):
        """Perform shared form of MADDPG setup."""
        # Create an input placeholder for the full state observations.
        self.all_obs_ph = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None,) + self.all_ob_space.shape,
            name='all_obs')
        self.all_obs1_ph = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None,) + self.all_ob_space.shape,
            name='all_obs1')

        # Create an input placeholder for the full actions.
        self.all_action_ph = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None,) + (self.ac_space.shape[0] * self.n_agents,),
            name='all_actions')

        # Compute the shape of the input observation space, which may include
        # the contextual term.
        ob_dim = self._get_ob_dim(self.ob_space, self.co_space)

        # Create the shared replay buffer.
        self.replay_buffer = SharedReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            obs_dim=ob_dim[0],
            ac_dim=self.ac_space.shape[0],
            n_agents=self.n_agents,
            all_obs_dim=self.all_ob_space.shape[0]
        )

        # Initialize some attributes.
        self.action_ph = []
        self.obs_ph = []
        self.obs1_ph = []
        actors = []
        actor_targets = []

        for i in range(self.n_agents):
            # Create input variables.
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
                shape=(None,) + self.ac_space.shape,
                name='actions')
            obs_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ob_dim,
                name='obs0')
            obs1_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ob_dim,
                name='obs1')

            # Create actor and critic networks for the shared policy.
            actor_tf, critic_tf, noisy_actor_target = self._setup_agent(
                obs_ph=obs_ph,
                obs1_ph=obs1_ph,
                ac_space=self.ac_space,
                all_obs_ph=self.all_obs_ph,
                all_action_ph=self.all_action_ph,
                target_policy_noise=self.target_policy_noise,
                target_noise_clip=self.target_noise_clip,
                reuse=i != 0,
            )

            # Store the new objects in their respective attributes.
            self.obs_ph.append(obs_ph)
            self.obs1_ph.append(obs1_ph)
            self.action_ph.append(action_ph)
            if i == 0:
                self.terminals1 = terminals1
                self.rew_ph = rew_ph
                self.actor_tf = actor_tf
                self.critic_tf = critic_tf
                self.actor_target = noisy_actor_target
            actors.append(actor_tf)
            actor_targets.append(noisy_actor_target)

        # Combine all actors for when creating a centralized differentiable
        # critic.
        combined_actors = tf.concat(actors, axis=1)

        # Combine all actor targets to create a centralized target actor.
        noisy_actor_target = tf.concat(actor_targets, axis=1)

        # Create the policy update and logging operations of the agent.
        (self.critic_loss,
         self.critic_optimizer,
         self.target_init_updates,
         self.target_soft_updates,
         self.actor_loss,
         self.actor_optimizer) = self._setup_agent_ops(
            scope=scope,
            actor_tf=self.actor_tf,
            critic_tf=self.critic_tf,
            noisy_actor_target=noisy_actor_target,
            all_obs_ph=self.all_obs_ph,
            all_obs1_ph=self.all_obs1_ph,
            rew_ph=self.rew_ph,
            terminals1=self.terminals1,
            combined_actors=combined_actors
        )

    def _setup_maddpg_independent(self, scope):
        """Perform independent form of MADDPG setup."""
        self.all_obs_ph = {}
        self.all_obs1_ph = {}
        self.all_action_ph = {}
        self.replay_buffer = {}
        self.terminals1 = {}
        self.rew_ph = {}
        self.action_ph = {}
        self.obs_ph = {}
        self.obs1_ph = {}
        self.actor_tf = {}
        self.critic_tf = {}
        self.actor_target = {}
        actors = []
        actor_targets = []

        # The size of the full action space.
        all_ac_dim = sum(
            self.ac_space[key].shape[0] for key in self.ac_space.keys())

        # We move through the keys in a sorted fashion so that we may collect
        # the observations and actions for the full state in a sorted manner.
        for key in sorted(self.ob_space.keys()):
            # Compute the shape of the input observation space, which may
            # include the contextual term.
            ob_dim = self._get_ob_dim(
                self.ob_space[key],
                None if self.co_space is None else self.co_space[key])

            # Create a replay buffer object.
            self.replay_buffer[key] = MultiReplayBuffer(
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                obs_dim=ob_dim[0],
                ac_dim=self.ac_space[key].shape[0],
                all_obs_dim=self.all_ob_space.shape[0],
                all_ac_dim=all_ac_dim,
            )

            with tf.compat.v1.variable_scope(key, reuse=False):
                # Create an input placeholder for the full state observations.
                self.all_obs_ph[key] = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=(None,) + self.all_ob_space.shape,
                    name='all_obs')
                self.all_obs1_ph[key] = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=(None,) + self.all_ob_space.shape,
                    name='all_obs1')

                # Create an input placeholder for the full actions.
                self.all_action_ph[key] = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=(None, all_ac_dim),
                    name='all_actions')

                # Create input variables.
                self.terminals1[key] = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=(None, 1),
                    name='terminals1')
                self.rew_ph[key] = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=(None, 1),
                    name='rewards')
                self.action_ph[key] = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=(None,) + self.ac_space[key].shape,
                    name='actions')
                self.obs_ph[key] = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=(None,) + ob_dim,
                    name='obs0')
                self.obs1_ph[key] = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=(None,) + ob_dim,
                    name='obs1')

                # Create actor and critic networks for the shared policy.
                actor_tf, critic_tf, noisy_actor_target = self._setup_agent(
                    obs_ph=self.obs_ph[key],
                    obs1_ph=self.obs1_ph[key],
                    ac_space=self.ac_space[key],
                    all_obs_ph=self.all_obs_ph[key],
                    all_action_ph=self.all_action_ph[key],
                    target_policy_noise=self.target_policy_noise[key],
                    target_noise_clip=self.target_noise_clip[key],
                    reuse=False,
                )

            # Store the new objects in their respective attributes.
            self.actor_tf[key] = actor_tf
            self.critic_tf[key] = critic_tf
            self.actor_target[key] = noisy_actor_target
            actors.append(actor_tf)
            actor_targets.append(noisy_actor_target)

        # Combine all actors for when creating a centralized differentiable
        # critic.
        combined_actors = tf.concat(actors, axis=1)

        # Combine all actor targets to create a centralized target actor.
        noisy_actor_target = tf.concat(actor_targets, axis=1)

        # Now that we have all actor targets, we can start constructing
        # centralized critic targets and all update procedures.
        self.critic_loss = {}
        self.critic_optimizer = {}
        self.target_init_updates = {}
        self.target_soft_updates = {}
        self.actor_loss = {}
        self.actor_optimizer = {}

        # Loop through all agents.
        for key in self.ob_space.keys():
            # Append the key to the outer scope term.
            scope_i = key if scope is None else "{}/{}".format(scope, key)

            # Create the policy update and logging operations of the agent.
            with tf.compat.v1.variable_scope(key, reuse=False):
                (self.critic_loss[key],
                 self.critic_optimizer[key],
                 self.target_init_updates[key],
                 self.target_soft_updates[key],
                 self.actor_loss[key],
                 self.actor_optimizer[key]) = self._setup_agent_ops(
                    scope=scope_i,
                    actor_tf=self.actor_tf[key],
                    critic_tf=self.critic_tf[key],
                    noisy_actor_target=noisy_actor_target,
                    all_obs_ph=self.all_obs_ph[key],
                    all_obs1_ph=self.all_obs1_ph[key],
                    rew_ph=self.rew_ph[key],
                    terminals1=self.terminals1[key],
                    combined_actors=combined_actors
                )

    def _setup_agent(self,
                     obs_ph,
                     obs1_ph,
                     ac_space,
                     all_obs_ph,
                     all_action_ph,
                     target_policy_noise,
                     target_noise_clip,
                     reuse):
        """Create the actor and critic variables for an individual agent.

        Parameters
        ----------
        obs_ph : tf.compat.v1.placeholder
            the input placeholder for the observation of the agent
        ac_space : gym.spaces.*
            the action space of the individual agent
        all_obs_ph : tf.compat.v1.placeholder
            placeholder for the last step full state observations
        all_action_ph : tf.compat.v1.placeholder
            placeholder for the actions of all agents
        target_policy_noise : array_like
            standard deviation term to the noise from the output of the target
            actor policy of the individual agent. See TD3 paper for more.
        target_noise_clip : array_like
            clipping term for the noise injected in the target actor policy of
            the individual agent
        reuse : bool
            whether to reuse policy objects

        Returns
        -------
        tf.Variable
            the output from the actor network
        list of tf.Variable
            the output from the critic networks. Two networks are used to
            stabilize training.
        tf.Variable
            the output from a noise target actor network
        """
        with tf.compat.v1.variable_scope("model", reuse=reuse):
            actor_tf = self.make_actor(obs_ph, ac_space)
            critic_tf = [
                self.make_critic(all_obs_ph, all_action_ph,
                                 scope="centralized_qf_{}".format(i))
                for i in range(2)
            ]

        with tf.compat.v1.variable_scope("target", reuse=reuse):
            # create the target actor policy
            actor_target = self.make_actor(obs1_ph, ac_space)

            # smooth target policy by adding clipped noise to target actions
            target_noise = tf.random.normal(
                tf.shape(actor_target), stddev=target_policy_noise)
            target_noise = tf.clip_by_value(
                target_noise, -target_noise_clip, target_noise_clip)

            # clip the noisy action to remain in the bounds
            noisy_actor_target = tf.clip_by_value(
                actor_target + target_noise,
                ac_space.low,
                ac_space.high
            )

        return actor_tf, critic_tf, noisy_actor_target

    def _setup_agent_ops(self,
                         scope,
                         actor_tf,
                         critic_tf,
                         noisy_actor_target,
                         all_obs_ph,
                         all_obs1_ph,
                         rew_ph,
                         terminals1,
                         combined_actors):
        """Create the optimizer and logging operations for a single agent.

        Parameters
        ----------
        scope : str
            the outer scope term
        actor_tf : tf.Variable
            the output from the actor of a given agent
        critic_tf : tf.Variable
            the output from the critic of a given agent
        noisy_actor_target : tf.Variable
            the output from the shared noisy actor target
        all_obs_ph : tf.compat.v1.placeholder
            TODO
        all_obs1_ph : tf.compat.v1.placeholder
            TODO
        rew_ph : tf.compat.v1.placeholder
            placeholder for the rewards of the agent
        terminals1 : tf.compat.v1.placeholder
            placeholder for the done mask of the agent
        combined_actors : tf.Variable
            the output from all actors, as a function of the agent's policy
            parameters

        Returns
        -------
        tf.Variable
            the loss of the critic
        tf.Operation
            the operation that updates the trainable parameters of the critic
        tf.Operation
            an operation that sets the values of the trainable parameters of
            the target actor/critic to match those actual actor/critic
        tf.Operation
            soft target update function
        tf.Variable
            the loss of the actor
        tf.Operation
            the operation that updates the trainable parameters of the actor
        """
        # Setup the target critic and critic update procedure.
        critic_loss, critic_optimizer = self._setup_critic_update(
            critic=critic_tf,
            all_obs1_ph=all_obs1_ph,
            actor_target=noisy_actor_target,
            rew_ph=rew_ph,
            done1=terminals1,
            scope=scope
        )

        # Create the target update operations.
        init, soft = self._setup_target_updates(
            'model', 'target', scope, self.tau, self.verbose)

        # Setup the actor update procedure.
        actor_loss, actor_optimizer = self._setup_actor_update(
            all_obs_ph=all_obs_ph,
            combined_actors=combined_actors,
            scope=scope
        )

        # Setup the running means and standard deviations of the model
        # inputs and outputs.
        self._setup_stats(
            rew_ph=rew_ph,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            critic_tf=critic_tf,
            actor_tf=actor_tf
        )

        return critic_loss, critic_optimizer, init, soft, actor_loss, \
            actor_optimizer

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
            an outer scope term

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

    def _setup_critic_update(self,
                             critic,
                             all_obs1_ph,
                             actor_target,
                             rew_ph,
                             done1,
                             scope):
        """Create the critic loss and optimization process.

        Parameters
        ----------
        critic : tf.Variable
            the output from the centralized critic of the agent
        all_obs1_ph : tf.compat.v1.placeholder
            TODO
        actor_target : tf.Variable
            the output from the combined target actors of all agents
        rew_ph : tf.compat.v1.placeholder
            placeholder for the rewards of the agent
        done1 : tf.compat.v1.placeholder
            placeholder for the done mask of the agent
        scope : str
            an outer scope term

        Returns
        -------
        tf.Operation
            the operation that returns the loss of the critic
        tf.Operation
            the operation that updates the trainable parameters of the critic
        """
        if self.verbose >= 2:
            print('setting up critic optimizer')

        # Create the centralized target critic policy.
        with tf.compat.v1.variable_scope("target", reuse=False):
            critic_target = [
                self.make_critic(all_obs1_ph, actor_target,
                                 scope="centralized_qf_{}".format(i))
                for i in range(2)
            ]

        # compute the target critic term
        with tf.compat.v1.variable_scope("loss", reuse=False):
            q_obs1 = tf.minimum(critic_target[0], critic_target[1])
            target_q = tf.stop_gradient(
                rew_ph + (1. - done1) * self.gamma * q_obs1)

            tf.compat.v1.summary.scalar('critic_target',
                                        tf.reduce_mean(target_q))

        # choose the loss function
        if self.use_huber:
            loss_fn = tf.compat.v1.losses.huber_loss
        else:
            loss_fn = tf.compat.v1.losses.mean_squared_error

        critic_loss = [loss_fn(q, target_q) for q in critic]

        critic_optimizer = []

        for i, loss in enumerate(critic_loss):
            scope_name = 'model/centralized_qf_{}'.format(i)
            if scope is not None:
                scope_name = scope + '/' + scope_name

            if self.verbose >= 2:
                critic_shapes = [var.get_shape().as_list()
                                 for var in get_trainable_vars(scope_name)]
                critic_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                        for shape in critic_shapes])
                print('  critic shapes: {}'.format(critic_shapes))
                print('  critic params: {}'.format(critic_nb_params))

            # create an optimizer object
            optimizer = tf.compat.v1.train.AdamOptimizer(self.critic_lr)

            # create the optimizer object
            critic_optimizer.append(optimizer.minimize(
                loss=loss,
                var_list=get_trainable_vars(scope_name)))

        return critic_loss, critic_optimizer

    def _setup_actor_update(self, all_obs_ph, combined_actors, scope):
        """Create the actor loss and optimization process.

        Parameters
        ----------
        all_obs_ph : tf.compat.v1.placeholder
            TODO
        combined_actors : tf.Variable
            the output from all actors, as a function of the agent's policy
            parameters
        scope : str
            an outer scope term

        Returns
        -------
        tf.Operation
            the operation that returns the loss of the actor
        tf.Operation
            the operation that updates the trainable parameters of the actor
        """
        if self.verbose >= 2:
            print('setting up actor optimizer')

        scope_name = 'model/pi/'
        if scope is not None:
            scope_name = scope + '/' + scope_name

        if self.verbose >= 2:
            actor_shapes = [var.get_shape().as_list()
                            for var in get_trainable_vars(scope_name)]
            actor_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                   for shape in actor_shapes])
            print('  actor shapes: {}'.format(actor_shapes))
            print('  actor params: {}'.format(actor_nb_params))

        # Create a differentiable form of the critic.
        with tf.compat.v1.variable_scope("model", reuse=False):
            critic_with_actor_tf = [
                self.make_critic(
                    all_obs_ph, combined_actors,
                    scope="centralized_qf_{}".format(i), reuse=True)
                for i in range(2)
            ]

        # compute the actor loss
        actor_loss = -tf.reduce_mean(critic_with_actor_tf[0])

        # create an optimizer object
        optimizer = tf.compat.v1.train.AdamOptimizer(self.actor_lr)

        actor_optimizer = optimizer.minimize(
            loss=actor_loss,
            var_list=get_trainable_vars(scope_name))

        return actor_loss, actor_optimizer

    @staticmethod
    def _setup_stats(rew_ph, actor_loss, critic_loss, actor_tf, critic_tf):
        """Prepare tensorboard logging for attributes of the agent.

        Parameters
        ----------
        rew_ph : tf.compat.v1.placeholder
            a placeholder for the rewards of an agent
        actor_loss : tf.Operation
            the operation that returns the loss of the actor
        critic_loss : list of tf.Operation
            the operation that returns the loss of the critic
        actor_tf : tf.Variable
            the output from the actor of the agent
        critic_tf : tf.Variable
            the output from the critics of the agent
        """
        # rewards
        tf.compat.v1.summary.scalar('rewards', tf.reduce_mean(rew_ph))

        # actor and critic losses
        tf.compat.v1.summary.scalar('actor_loss', actor_loss)
        tf.compat.v1.summary.scalar('Q1_loss', critic_loss[0])
        tf.compat.v1.summary.scalar('Q2_loss', critic_loss[1])

        # critic dynamics
        tf.compat.v1.summary.scalar(
            'reference_Q1_mean', tf.reduce_mean(critic_tf[0]))
        tf.compat.v1.summary.scalar(
            'reference_Q1_std', reduce_std(critic_tf[0]))

        tf.compat.v1.summary.scalar(
            'reference_Q2_mean', tf.reduce_mean(critic_tf[1]))
        tf.compat.v1.summary.scalar(
            'reference_Q2_std', reduce_std(critic_tf[1]))

        # actor dynamics
        tf.compat.v1.summary.scalar(
            'reference_action_mean', tf.reduce_mean(actor_tf))
        tf.compat.v1.summary.scalar(
            'reference_action_std', reduce_std(actor_tf))

    def _initialize_maddpg(self):
        """See initialize.

        This method initializes the target parameters to match the model
        parameters.
        """
        if self.shared:
            self.sess.run(self.target_init_updates)
        else:
            self.sess.run([self.target_init_updates[key]
                           for key in self.target_init_updates.keys()])

    def _update_maddpg(self, update_actor=True, **kwargs):
        """See update."""
        # =================================================================== #
        #                       Shared update procedure                       #
        # =================================================================== #

        if self.shared:
            # Not enough samples in the replay buffer.
            if not self.replay_buffer.can_sample():
                return {"policy": [0, 0]}, {"policy": 0}

            # Get a batch.
            obs0, actions, rewards, obs1, done1, all_obs0, all_obs1 = \
                self.replay_buffer.sample()

            # Reshape to match previous behavior and placeholder shape.
            rewards = rewards.reshape(-1, 1)
            done1 = done1.reshape(-1, 1)

            # Update operations for the critic networks.
            step_ops = [self.critic_loss,
                        self.critic_optimizer[0],
                        self.critic_optimizer[1]]

            if update_actor:
                # Actor updates and target soft update operation.
                step_ops += [self.actor_loss,
                             self.actor_optimizer,
                             self.target_soft_updates]

            # Prepare the feed_dict information.
            feed_dict = {
                self.all_obs_ph: all_obs0,
                self.all_obs1_ph: all_obs1,
                self.all_action_ph: np.concatenate(actions, axis=1),
                self.rew_ph: rewards,
                self.terminals1: done1
            }

            # Add the agent-level data to the feed dict.
            feed_dict.update({
                self.obs_ph[i]: obs0[i] for i in range(self.n_agents)})
            feed_dict.update({
                self.action_ph[i]: actions[i] for i in range(self.n_agents)})
            feed_dict.update({
                self.obs1_ph[i]: obs1[i] for i in range(self.n_agents)})

            # Perform the update operations and collect the critic loss.
            critic_loss, *_vals = self.sess.run(step_ops, feed_dict=feed_dict)
            critic_loss = {"policy": critic_loss}

            # Extract the actor loss.
            actor_loss = _vals[2] if update_actor else 0
            actor_loss = {"policy": actor_loss}

        # =================================================================== #
        #                    Independent update procedure                     #
        # =================================================================== #

        else:
            actor_loss = {}
            critic_loss = {}

            # Loop through all agent.
            for key in self.replay_buffer.keys():
                # Not enough samples in the replay buffer.
                if not self.replay_buffer[key].can_sample():
                    actor_loss[key] = 0
                    critic_loss[key] = [0, 0]
                    continue

                # Get a batch.
                obs0, actions, rewards, obs1, done1, all_obs0, all_actions, \
                    all_obs1 = self.replay_buffer[key].sample()

                # Reshape to match previous behavior and placeholder shape.
                rewards = rewards.reshape(-1, 1)
                done1 = done1.reshape(-1, 1)

                # Update operations for the critic networks.
                step_ops = [self.critic_loss[key],
                            self.critic_optimizer[key][0],
                            self.critic_optimizer[key][1]]

                if update_actor:
                    # Actor updates and target soft update operation.
                    step_ops += [self.actor_loss[key],
                                 self.actor_optimizer[key],
                                 self.target_soft_updates[key]]

                # Prepare the feed_dict information.
                feed_dict = {
                    self.obs_ph[key]: obs0,
                    self.obs1_ph[key]: obs1,
                    self.action_ph[key]: actions,
                    self.all_obs_ph[key]: all_obs0,
                    self.all_obs1_ph[key]: all_obs1,
                    self.all_action_ph[key]: all_actions,
                    self.rew_ph[key]: rewards,
                    self.terminals1[key]: done1
                }

                # Perform the update operations and collect the critic loss.
                critic_loss[key], *_vals = self.sess.run(
                    step_ops, feed_dict=feed_dict)

                # Extract the actor loss.
                actor_loss[key] = _vals[2] if update_actor else 0

        return critic_loss, actor_loss

    def _get_action_maddpg(self, obs, context, apply_noise, random_actions):
        """See get_action."""
        actions = {}

        if random_actions:
            for key in obs.keys():
                # Get the action space of the specific agent.
                ac_space = self.ac_space if self.shared else self.ac_space[key]

                # Sample a random action.
                actions[key] = np.array([ac_space.sample()])

        else:
            for key in obs.keys():
                # Get the action space of the specific agent.
                ac_space = self.ac_space if self.shared else self.ac_space[key]

                # Compute the deterministic action.
                if self.shared:
                    action = self.sess.run(
                        self.actor_tf,
                        feed_dict={self.obs_ph[0]: obs[key]})
                else:
                    action = self.sess.run(
                        self.actor_tf[key],
                        feed_dict={self.obs_ph[key]: obs[key]})

                # compute noisy action
                if apply_noise:
                    noise = self.noise if self.shared else self.noise[key]
                    action += np.random.normal(0, noise, action.shape)

                # clip by bounds
                actions[key] = np.clip(action, ac_space.low, ac_space.high)

        return actions

    def _value_maddpg(self, obs, context, action):
        """See value."""
        # Combine all actions under one variable. This is done by order of
        # agent IDs in alphabetical order.
        # FIXME: this could cause problems in the merge.
        all_actions = np.concatenate(
            [action[key] for key in sorted(list(action.keys()))], axis=1)

        if self.shared:
            # Compute the shared value.
            value_all = self.sess.run(
                self.critic_tf,
                feed_dict={
                    self.all_obs_ph: obs,
                    self.all_action_ph: all_actions
                }
            )

            # Distribute across all agents.
            value = {key: value_all for key in action.keys()}
        else:
            # Loop through all agent.
            value = {}
            for key in self.replay_buffer.keys():
                value[key] = self.sess.run(
                    self.critic_tf[key],
                    feed_dict={
                        self.all_obs_ph[key]: obs,
                        self.all_action_ph[key]: all_actions
                    }
                )

        return value

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
                                 evaluate):
        """See store_transition."""
        if self.shared:
            # Collect the observations and actions in order as listed by their
            # agent IDs. FIXME: this could cause problems in the merge.
            list_obs0, list_obs1, list_action = [], [], []
            for key in sorted(list(obs0.keys())):
                list_obs0.append(self._get_obs(
                    obs0[key], None if context0 is None else context0[key]))
                list_obs1.append(self._get_obs(
                    obs1[key], None if context1 is None else context0[key]))
                list_action.append(action[key])

            # Store the new sample.
            self.replay_buffer.add(
                obs_t=list_obs0,
                action=list_action,
                reward=reward,
                obs_tp1=list_obs1,
                done=float(done and not is_final_step),
                all_obs_t=all_obs0,
                all_obs_tp1=all_obs1
            )
        else:
            # Collect the actions in order as listed by their agent IDs.
            # FIXME: this could cause problems in the merge.
            combines_actions = np.array(
                [action[key] for key in sorted(list(action.keys()))])

            # Store the new samples in their replay buffer.
            for key in obs0.keys():
                self.replay_buffer[key].add(
                    obs_t=obs0[key],
                    action=action[key],
                    reward=reward[key],
                    obs_tp1=obs1[key],
                    done=float(done and not is_final_step),
                    all_obs_t=all_obs0,
                    all_action_t=combines_actions,
                    all_obs_tp1=all_obs1
                )

    def _get_td_map_maddpg(self):
        """See get_td_map."""
        if self.shared:
            # Not enough samples in the replay buffer.
            if not self.replay_buffer.can_sample():
                return {}

            # Get a batch.
            obs0, actions, rewards, obs1, done1, all_obs0, all_obs1 = \
                self.replay_buffer.sample()

            # Reshape to match previous behavior and placeholder shape.
            rewards = rewards.reshape(-1, 1)
            done1 = done1.reshape(-1, 1)

            # Combine all actions under one variable. This is done by order of
            # agent IDs in alphabetical order.
            # FIXME: this could cause problems in the merge.
            all_actions = np.concatenate(actions, axis=1)

            td_map = {
                self.all_obs_ph: all_obs0,
                self.all_action_ph: all_actions,
                self.all_obs1_ph: all_obs1,
                self.rew_ph: rewards,
                self.terminals1: done1
            }

            # Add the agent-level placeholders and variables.
            td_map.update({
                self.obs_ph[i]: obs0[i] for i in range(self.n_agents)})
            td_map.update({
                self.action_ph[i]: actions[i] for i in range(self.n_agents)})
            td_map.update({
                self.obs1_ph[i]: obs1[i] for i in range(self.n_agents)})
        else:
            td_map = {}

            # Loop through all agent.
            for key in sorted(list(self.replay_buffer.keys())):
                # Not enough samples in the replay buffer.
                if not self.replay_buffer[key].can_sample():
                    return {}

                # Get a batch.
                obs0, actions, rewards, obs1, done1, all_obs0, all_actions, \
                    all_obs1 = self.replay_buffer[key].sample()

                # Reshape to match previous behavior and placeholder shape.
                rewards = rewards.reshape(-1, 1)
                done1 = done1.reshape(-1, 1)

                # Add the agent-level placeholders and variables.
                td_map.update({
                    self.rew_ph[key]: rewards,
                    self.terminals1[key]: done1,
                    self.obs_ph[key]: obs0,
                    self.action_ph[key]: actions,
                    self.obs1_ph[key]: obs1,
                    self.all_obs_ph[key]: all_obs0,
                    self.all_action_ph[key]: all_actions,
                    self.all_obs1_ph[key]: all_obs1,
                })

        return td_map
