"""SAC-compatible multi-agent feedforward policy."""
import tensorflow as tf
import numpy as np

from hbaselines.multi_fcnet.base import MultiFeedForwardPolicy as BasePolicy
from hbaselines.fcnet.sac import FeedForwardPolicy
from hbaselines.multi_fcnet.replay_buffer import MultiReplayBuffer
from hbaselines.multi_fcnet.replay_buffer import SharedReplayBuffer
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.utils.tf_util import reduce_std
from hbaselines.utils.tf_util import gaussian_likelihood
from hbaselines.utils.tf_util import apply_squashing_func
from hbaselines.utils.tf_util import print_params_shape

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
    critic_loss : tf.Operation
        the operation that returns the loss of the critic
    critic_optimizer : tf.Operation
        the operation that updates the trainable parameters of the critic
    target_init_updates : tf.Operation
        an operation that sets the values of the trainable parameters of the
        target actor/critic to match those actual actor/critic
    target_soft_updates : tf.Operation
        soft target update function
    alpha_loss : tf.Operation
        the operation that returns the loss of the entropy term
    alpha_optimizer : tf.Operation
        the operation that updates the trainable parameters of the entropy term
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

        if shared:
            self._ac_mean = 0.5 * (ac_space.high + ac_space.low)
            self._ac_mag = 0.5 * (ac_space.high - ac_space.low)
        else:
            self._ac_mean = {
                key: 0.5 * (ac_space[key].high + ac_space[key].low)
                for key in ac_space.keys()
            }
            self._ac_mag = {
                key: 0.5 * (ac_space[key].high - ac_space[key].low)
                for key in ac_space.keys()
            }

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
        self.critic_loss = None
        self.critic_optimizer = None
        self.target_init_updates = None
        self.target_soft_updates = None
        self.alpha_loss = None
        self.alpha_optimizer = None
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
                target_entropy=target_entropy,
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

        # Compute the shape of the input observation space, which may
        # include the contextual term.
        ob_dim = self._get_ob_dim(self.ob_space, self.co_space)

        for i in range(self.n_agents):
            # Create input variables for the current agent. The policies
            # parameters will still be shared by all agents.
            with tf.compat.v1.variable_scope("agent_{}".format(i)):
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
            deterministic_action, policy_out, logp_pi, logp_action, qf1, qf2, \
                value_fn, log_alpha, alpha, value_target = \
                self._setup_agent(
                    obs_ph=obs_ph,
                    action_ph=action_ph,
                    ac_space=self.ac_space,
                    all_obs_ph=self.all_obs_ph,
                    all_obs1_ph=self.all_obs1_ph,
                    all_action_ph=self.all_action_ph,
                    reuse=i != 0,
                )

            # Store the new objects in their respective attributes.
            self.action_ph.append(action_ph)  # TODO: maybe not?
            self.obs_ph.append(obs_ph)
            self.obs1_ph.append(obs1_ph)  # TODO: maybe not?
            if i == 0:
                self.terminals1 = terminals1
                self.rew_ph = rew_ph
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
            actors.append(policy_out)

        # Combine all actors for when creating a centralized differentiable
        # critic.
        combined_actors = tf.concat(actors, axis=1)

        # Create the policy update and logging operations of the agent.
        (self.critic_loss,
         self.critic_optimizer,
         self.target_init_updates,
         self.target_soft_updates,
         self.alpha_loss,
         self.alpha_optimizer,
         self.actor_loss,
         self.actor_optimizer) = self._setup_agent_ops(
            scope=scope,
            all_obs_ph=self.all_obs_ph,
            combined_actors=combined_actors,
            rew_ph=self.rew_ph,
            terminals1=self.terminals1,
            value_target=self.value_target,
            qf1=self.qf1,
            qf2=self.qf2,
            alpha=self.alpha,
            logp_pi=self.logp_pi,
            value_fn=self.value_fn,
            log_alpha=self.log_alpha,
            target_entropy=self.target_entropy,
            policy_out=self.policy_out
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
        actors = []

        # The size of the full action space.
        all_ac_dim = sum(
            self.ac_space[key].shape[0] for key in self.ac_space.keys())

        # We move through the keys in a sorted fashion so that we may collect
        # the observations and actions for the full state in a sorted manner.
        for key in sorted(self.ob_space.keys()):
            # Compute the shape of the input observation space, which may
            # include the contextual term.
            ob_dim = self._get_ob_dim(
                self.ob_space[key], self.co_space[key])

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

                # Create actor and critic networks for the the individual
                # policies.
                deterministic_action, policy_out, logp_pi, logp_action, qf1, \
                    qf2, value_fn, log_alpha, alpha, value_target = \
                    self._setup_agent(
                        obs_ph=self.obs_ph[key],
                        action_ph=self.action_ph[key],
                        ac_space=self.ac_space[key],
                        all_obs_ph=self.all_obs_ph[key],
                        all_obs1_ph=self.all_obs1_ph[key],
                        all_action_ph=self.all_action_ph[key],
                        reuse=False,
                    )

            # Store the new objects in their respective attributes.
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
            actors.append(policy_out)

        # Combine all actors for when creating a centralized differentiable
        # critic.
        combined_actors = tf.concat(actors, axis=1)

        # Now that we have all actors, we can start constructing centralized
        # critic targets and all update procedures.
        self.critic_loss = {}
        self.critic_optimizer = {}
        self.target_init_updates = {}
        self.target_soft_updates = {}
        self.alpha_loss = {}
        self.alpha_optimizer = {}
        self.actor_loss = {}
        self.actor_optimizer = {}

        # Loop through all agents.
        for key in sorted(self.ob_space.keys()):
            # Append the key to the outer scope term.
            scope_i = key if scope is None else "{}/{}".format(scope, key)

            # Create the policy update and logging operations of the agent.
            with tf.compat.v1.variable_scope(key, reuse=False):
                (self.critic_loss[key],
                 self.critic_optimizer[key],
                 self.target_init_updates[key],
                 self.target_soft_updates[key],
                 self.alpha_loss[key],
                 self.alpha_optimizer[key],
                 self.actor_loss[key],
                 self.actor_optimizer[key]) = self._setup_agent_ops(
                    scope=scope_i,
                    all_obs_ph=self.all_obs_ph[key],
                    combined_actors=combined_actors,
                    rew_ph=self.rew_ph[key],
                    terminals1=self.terminals1[key],
                    value_target=self.value_target[key],
                    qf1=self.qf1[key],
                    qf2=self.qf2[key],
                    alpha=self.alpha[key],
                    logp_pi=self.logp_pi[key],
                    value_fn=self.value_fn[key],
                    log_alpha=self.log_alpha[key],
                    target_entropy=self.target_entropy[key],
                    policy_out=self.policy_out[key]
                )

    def _setup_agent(self,
                     obs_ph,
                     action_ph,
                     ac_space,
                     all_obs_ph,
                     all_obs1_ph,
                     all_action_ph,
                     reuse):
        """Create the actor and critic variables for an individual agent.

        Parameters
        ----------
        obs_ph : tf.compat.v1.placeholder
            placeholder for the observations for each agent
        action_ph : tf.compat.v1.placeholder
            placeholder for the actions for each agent
        ac_space : gym.spaces.*
            the action space of the individual agent
        all_obs_ph : tf.compat.v1.placeholder
            placeholder for the last step full state observations
        all_obs1_ph : tf.compat.v1.placeholder
            placeholder for the current step full state observations
        all_action_ph : tf.compat.v1.placeholder
            placeholder for the actions of all agents
        reuse : bool
            whether to reuse the policy. This is set to True when creating
            shared policies.

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
        # Create networks and core TF parts that are shared across setup parts.
        with tf.compat.v1.variable_scope("model", reuse=reuse):
            deterministic_action, policy_out, logp_pi, logp_action = \
                self.make_actor(obs_ph, ac_space, action_ph)
            qf1, qf2, value_fn = self.make_critic(
                all_obs_ph, all_action_ph,
                scope="centralized_value_fns", create_qf=True, create_vf=True)

            # The entropy coefficient or entropy can be learned automatically,
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            log_alpha = tf.compat.v1.get_variable(
                'log_alpha',
                dtype=tf.float32,
                initializer=0.0)
            alpha = tf.exp(log_alpha)

        with tf.compat.v1.variable_scope("target", reuse=reuse):
            # Create the value network
            _, _, value_target = self.make_critic(
                all_obs1_ph,
                scope="centralized_value_fns", create_qf=False, create_vf=True)

        return deterministic_action, policy_out, logp_pi, logp_action, qf1, \
            qf2, value_fn, log_alpha, alpha, value_target

    def _setup_agent_ops(self,
                         scope,
                         all_obs_ph,
                         combined_actors,
                         rew_ph,
                         terminals1,
                         value_target,
                         qf1,
                         qf2,
                         alpha,
                         logp_pi,
                         value_fn,
                         log_alpha,
                         target_entropy,
                         policy_out):
        """Create the optimizer and logging operations for a single agent.

        Parameters
        ----------
        scope : str
            an outer scope term, with the appended agent ID in the case of
            independent policies
        all_obs_ph : tf.compat.v1.placeholder
            placeholder for the last step full state observations
        combined_actors : tf.Variable
            the output from the joint actors
        rew_ph : tf.compat.v1.placeholder
            a placeholder for the rewards of a given agent
        terminals1 : tf.compat.v1.placeholder
            placeholder for the next step terminals of a given agent
        value_target : tf.Variable
            the output from the shared target value function
        qf1 : tf.Variable
            the output from the first Q-function of a given agent
        qf2 : tf.Variable
            the output from the second Q-function of a given agent
        alpha : tf.Variable
            the entropy coefficient of a given agent
        logp_pi : tf.Variable
            the log-probability of a given observation given the output action
            from the policy
        value_fn : tf.Variable
            the output from the value function of a given agent
        log_alpha : tf.Variable
            the log of the entropy coefficient of a given agent
        target_entropy : tf.Variable or float
            target entropy used when learning the entropy coefficient of a
            given agent
        policy_out : tf.Variable
            the output from the stochastic actor of a given agent

        Returns
        -------
        critic_loss : tf.Variable
            the loss of the critic
        critic_optimizer : tf.Operation
            the operation that updates the trainable parameters of the critic
        target_init_updates : tf.Operation
            an operation that sets the values of the trainable parameters of
            the target actor/critic to match those actual actor/critic
        target_soft_updates : tf.Operation
            soft target update function
        alpha_loss : tf.Variable
            the loss of the entropy term
        alpha_optimizer : tf.Operation
            the operation that updates the trainable parameters of the entropy
            term
        actor_loss : tf.Variable
            the loss of the actor
        actor_optimizer : tf.Operation
            the operation that updates the trainable parameters of the actor
        """
        # Create the differentiable critics for the agent.
        with tf.compat.v1.variable_scope("model", reuse=False):
            qf1_pi, qf2_pi, _ = self.make_critic(
                all_obs_ph, combined_actors,
                scope="centralized_value_fns",
                create_qf=True, create_vf=False, reuse=True
            )

        # Setup the target critic and critic update procedure.
        critic_loss, critic_optimizer = self._setup_critic_update(
            qf1_pi=qf1_pi,
            qf2_pi=qf2_pi,
            rew_ph=rew_ph,
            terminals1=terminals1,
            value_target=value_target,
            qf1=qf1,
            qf2=qf2,
            alpha=alpha,
            logp_pi=logp_pi,
            value_fn=value_fn,
            scope=scope
        )

        # Create the target update operations.
        init, soft = self._setup_target_updates(
            model_scope='model/centralized_value_fns/vf',
            target_scope='target/centralized_value_fns/vf',
            scope=scope,
            tau=self.tau,
            verbose=self.verbose
        )

        # Setup the actor update procedure.
        alpha_l, alpha_o, actor_l, actor_o = self._setup_actor_update(
            qf1_pi=qf1_pi,
            qf2_pi=qf2_pi,
            log_alpha=log_alpha,
            alpha=alpha,
            logp_pi=logp_pi,
            target_entropy=target_entropy,
            scope=scope
        )

        # Setup the running means and standard deviations of the model
        # inputs and outputs.
        self._setup_stats(
            rew_ph=rew_ph,
            alpha_loss=alpha_l,
            actor_loss=actor_l,
            critic_loss=critic_loss,
            qf1=qf1,
            qf2=qf2,
            value_fn=value_fn,
            qf1_pi=qf1_pi,
            qf2_pi=qf2_pi,
            policy_out=policy_out,
            logp_pi=logp_pi
        )

        return critic_loss, critic_optimizer, init, soft, alpha_l, alpha_o, \
            actor_l, actor_o

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
        logp_pi = gaussian_likelihood(policy, policy_mean, log_std)
        logp_ac = gaussian_likelihood(action, policy_mean, log_std)

        # Apply squashing and account for it in the probability
        _, _, logp_ac = apply_squashing_func(
            policy_mean, action, logp_ac)
        deterministic_policy, policy, logp_pi = apply_squashing_func(
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

    def _setup_critic_update(self,
                             qf1_pi,
                             qf2_pi,
                             rew_ph,
                             terminals1,
                             value_target,
                             qf1,
                             qf2,
                             alpha,
                             logp_pi,
                             value_fn,
                             scope):
        """Create the critic loss and optimization process.

        Parameters
        ----------
        qf1_pi : tf.Variable
            the output from the first Q-function with the action provided
            directly by the actor policy
        qf2_pi : tf.Variable
            the output from the second Q-function with the action provided
            directly by the actor policy
        rew_ph : tf.compat.v1.placeholder
            a placeholder for the rewards of an agent
        terminals1 : tf.compat.v1.placeholder
            placeholder for the next step terminals for each agent
        value_target : tf.Variable
            the output from the shared target value function
        qf1 : tf.Variable
            the output from the first Q-function
        qf2 : tf.Variable
            the output from the second Q-function
        alpha : tf.Variable
            the entropy coefficient
        logp_pi : tf.Variable
            the log-probability of a given observation given the output action
            from the policy
        value_fn : tf.Variable
            the output from the value function
        scope : str
            an outer scope term

        Returns
        -------
        tf.Operation
            the operation that returns the loss of the critic
        tf.Operation
            the operation that updates the trainable parameters of the critic
        """
        scope_name = 'model/centralized_value_fns'
        if scope is not None:
            scope_name = scope + '/' + scope_name

        if self.verbose >= 2:
            print('setting up critic optimizer')
            for name in ['qf1', 'qf2', 'vf']:
                scope_i = '{}/{}'.format(scope_name, name)
                print_params_shape(scope_i, name)

        # Take the min of the two Q-Values (Double-Q Learning)
        min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

        # Target for Q value regression
        q_backup = tf.stop_gradient(
            rew_ph + (1 - terminals1) * self.gamma * value_target)

        # choose the loss function
        if self.use_huber:
            loss_fn = tf.compat.v1.losses.huber_loss
        else:
            loss_fn = tf.compat.v1.losses.mean_squared_error

        # Compute Q-Function loss
        qf1_loss = loss_fn(q_backup, qf1)
        qf2_loss = loss_fn(q_backup, qf2)

        # Target for value fn regression
        # We update the vf towards the min of two Q-functions in order to
        # reduce overestimation bias from function approximation error.
        v_backup = tf.stop_gradient(min_qf_pi - alpha * logp_pi)
        value_loss = loss_fn(value_fn, v_backup)

        critic_loss = (qf1_loss, qf2_loss, value_loss)

        # Critic train op
        optimizer = tf.compat.v1.train.AdamOptimizer(self.critic_lr)
        critic_optimizer = optimizer.minimize(
            qf1_loss + qf2_loss + value_loss,
            var_list=get_trainable_vars(scope_name))

        return critic_loss, critic_optimizer

    def _setup_actor_update(self,
                            qf1_pi,
                            qf2_pi,
                            log_alpha,
                            alpha,
                            logp_pi,
                            target_entropy,
                            scope):
        """Create the actor and alpha loss and optimization process.

        Parameters
        ----------
        qf1_pi : tf.Variable
            the output from the first Q-function with the action provided
            directly by the actor policy
        qf2_pi : tf.Variable
            the output from the second Q-function with the action provided
            directly by the actor policy
        log_alpha : tf.Variable
            the log of the entropy coefficient
        alpha : tf.Variable
            the entropy coefficient
        logp_pi : tf.Variable
            the log-probability of a given observation given the output action
            from the policy
        target_entropy : float or tf.Variable
            target entropy used when learning the entropy coefficient
        scope : str
            an outer scope term

        Returns
        -------
        tf.Operation
            the operation that returns the loss of the alpha term
        tf.Operation
            the operation that updates the alpha term
        tf.Operation
            the operation that returns the loss of the actor
        tf.Operation
            the operation that updates the trainable parameters of the actor
        """
        scope_name = 'model/pi/'
        if scope is not None:
            scope_name = scope + '/' + scope_name

        if self.verbose >= 2:
            print('setting up actor and alpha optimizers')
            print_params_shape(scope_name, "actor")

        # Take the min of the two Q-Values (Double-Q Learning)
        min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

        # Compute the entropy temperature loss.
        alpha_loss = -tf.reduce_mean(
            log_alpha * tf.stop_gradient(logp_pi + target_entropy))

        optimizer = tf.compat.v1.train.AdamOptimizer(self.actor_lr)

        alpha_optimizer = optimizer.minimize(
            alpha_loss,
            var_list=log_alpha)

        # Compute the policy loss
        actor_loss = tf.reduce_mean(alpha * logp_pi - min_qf_pi)

        # Policy train op (has to be separate from value train op, because
        # min_qf_pi appears in policy_loss)
        optimizer = tf.compat.v1.train.AdamOptimizer(self.actor_lr)

        actor_optimizer = optimizer.minimize(
            actor_loss,
            var_list=get_trainable_vars(scope_name))

        return alpha_loss, alpha_optimizer, actor_loss, actor_optimizer

    @staticmethod
    def _setup_stats(rew_ph,
                     alpha_loss,
                     actor_loss,
                     critic_loss,
                     qf1,
                     qf2,
                     value_fn,
                     qf1_pi,
                     qf2_pi,
                     policy_out,
                     logp_pi):
        """Prepare tensorboard logging for attributes of the agent.

        Parameters
        ----------
        rew_ph : tf.compat.v1.placeholder
            a placeholder for the rewards of an agent
        alpha_loss : tf.Operation
            the operation that returns the loss of the actor
        actor_loss : tf.Operation
            the operation that returns the loss of the actor
        critic_loss : list of tf.Operation
            the operation that returns the loss of the critic
        qf1 : tf.Variable
            the output from the first Q-function
        qf2 : tf.Variable
            the output from the second Q-function
        value_fn : tf.Variable
            the output from the value function
        qf1_pi : tf.Variable
            the output from the first Q-function with the action provided
            directly by the actor policy
        qf2_pi : tf.Variable
            the output from the second Q-function with the action provided
            directly by the actor policy
        policy_out : tf.Variable
            the output from the stochastic actor
        logp_pi : tf.Variable
            the log-probability of a given observation given the output action
            from the policy
        """
        # rewards
        tf.compat.v1.summary.scalar('rewards', tf.reduce_mean(rew_ph))

        # actor and critic losses
        tf.compat.v1.summary.scalar('alpha_loss', alpha_loss)
        tf.compat.v1.summary.scalar('actor_loss', actor_loss)
        tf.compat.v1.summary.scalar('Q1_loss', critic_loss[0])
        tf.compat.v1.summary.scalar('Q2_loss', critic_loss[1])
        tf.compat.v1.summary.scalar('value_loss', critic_loss[1])

        # critic dynamics
        tf.compat.v1.summary.scalar(
            'reference_Q1_mean', tf.reduce_mean(qf1))
        tf.compat.v1.summary.scalar(
            'reference_Q1_std', reduce_std(qf1))

        tf.compat.v1.summary.scalar(
            'reference_Q2_mean', tf.reduce_mean(qf2))
        tf.compat.v1.summary.scalar(
            'reference_Q2_std', reduce_std(qf2))

        tf.compat.v1.summary.scalar(
            'reference_value_mean', tf.reduce_mean(value_fn))
        tf.compat.v1.summary.scalar(
            'reference_value_mean', reduce_std(value_fn))

        tf.compat.v1.summary.scalar(
            'reference_actor_Q1_mean', tf.reduce_mean(qf1_pi))
        tf.compat.v1.summary.scalar(
            'reference_actor_Q1_std', reduce_std(qf1_pi))

        tf.compat.v1.summary.scalar(
            'reference_actor_Q2_mean', tf.reduce_mean(qf2_pi))
        tf.compat.v1.summary.scalar(
            'reference_actor_Q2_std', reduce_std(qf2_pi))

        # actor dynamics
        tf.compat.v1.summary.scalar(
            'reference_action_mean', tf.reduce_mean(policy_out))
        tf.compat.v1.summary.scalar(
            'reference_action_std', reduce_std(policy_out))

        # log probability dynamics
        tf.compat.v1.summary.scalar(
            'reference_log_probability_mean', tf.reduce_mean(logp_pi))
        tf.compat.v1.summary.scalar(
            'reference_log_probability_std', reduce_std(logp_pi))

    def _initialize_maddpg(self):
        """See initialize."""
        if self.shared:
            self.sess.run(self.target_init_updates)
        else:
            self.sess.run([self.target_init_updates[key]
                           for key in self.target_init_updates.keys()])

    def _update_maddpg(self, update_actor=True, **kwargs):
        """See update."""
        del update_actor  # unused by this method

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

            # Normalize the actions (bounded between [-1, 1]).
            actions = (actions - self._ac_mean) / self._ac_mag

            # Collect all update and loss call operations.
            step_ops = [
                self.critic_loss[0],
                self.critic_loss[1],
                self.critic_loss[2],
                self.actor_loss,
                self.alpha_loss,
                self.critic_optimizer,
                self.actor_optimizer,
                self.alpha_optimizer,
                self.target_soft_updates,
            ]

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

            # Perform the update operations and collect the actor and critic
            # loss.
            q1_loss, q2_loss, vf_loss, actor_loss, *_ = self.sess.run(
                step_ops, feed_dict)
            critic_loss = {"policy": [q1_loss, q2_loss]}
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

                # Normalize the actions (bounded between [-1, 1]).
                actions = (actions - self._ac_mean[key]) / self._ac_mag[key]

                # Collect all update and loss call operations.
                step_ops = [
                    self.critic_loss[key][0],
                    self.critic_loss[key][1],
                    self.critic_loss[key][2],
                    self.actor_loss[key],
                    self.alpha_loss[key],
                    self.critic_optimizer[key],
                    self.actor_optimizer[key],
                    self.alpha_optimizer[key],
                    self.target_soft_updates[key],
                ]

                # Prepare the feed_dict information.
                feed_dict = {
                    self.rew_ph[key]: rewards,
                    self.terminals1[key]: done1,
                    self.obs_ph[key]: obs0,
                    self.action_ph[key]: actions,
                    self.obs1_ph[key]: obs1,
                    self.all_obs_ph[key]: all_obs0,
                    self.all_action_ph[key]: all_actions,
                    self.all_obs1_ph[key]: all_obs1,
                }

                # Perform the update operations and collect the actor and
                # critic loss.
                q1_loss, q2_loss, vf_loss, actor_loss[key], *_ = \
                    self.sess.run(step_ops, feed_dict)
                critic_loss[key] = [q1_loss, q2_loss]

        return critic_loss, actor_loss  # FIXME: add vf_loss

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
                # Add the contextual observation, if applicable.
                obs[key] = self._get_obs(
                    obs[key],
                    None if context is None else context[key], axis=1)

                if apply_noise:
                    if self.shared:
                        normalized_action = self.sess.run(
                            self.policy_out,
                            feed_dict={self.obs_ph[0]: obs[key]})
                    else:
                        normalized_action = self.sess.run(
                            self.policy_out[key],
                            feed_dict={self.obs_ph[key]: obs[key]})
                else:
                    if self.shared:
                        normalized_action = self.sess.run(
                            self.deterministic_action,
                            feed_dict={self.obs_ph[0]: obs[key]})
                    else:
                        normalized_action = self.sess.run(
                            self.deterministic_action[key],
                            feed_dict={self.obs_ph[key]: obs[key]})

                # Get the scaling terms for the actions.
                ac_mag = self._ac_mag if self.shared else self._ac_mag[key]
                ac_mean = self._ac_mean if self.shared else self._ac_mean[key]

                # Scale by the action space.
                actions[key] = ac_mag * normalized_action + ac_mean

        return actions

    def _value_maddpg(self, obs, context, action):
        """See value."""
        for key in sorted(list(action.keys())):
            # Get the scaling terms for the actions.
            ac_mag = self._ac_mag if self.shared else self._ac_mag[key]
            ac_mean = self._ac_mean if self.shared else self._ac_mean[key]

            # Normalize the action (bounded between [-1, 1]).
            action[key] = (action[key] - ac_mean) / ac_mag

        # Combine all actions under one variable. This is done by order of
        # agent IDs in alphabetical order.
        # FIXME: this could cause problems in the merge.
        all_actions = np.concatenate(
            [action[key] for key in sorted(list(action.keys()))], axis=1)

        if self.shared:
            # Compute the shared value.
            value_all = self.sess.run(
                [self.qf1, self.qf2],  # , self.value_fn],  FIXME
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
            for key in action.keys():
                value[key] = self.sess.run(
                    [self.qf1[key], self.qf2[key]],  # , self.value_fn],  FIXME
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
                done=float(done),
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
                    done=float(done),
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
            # Loop through all agent.
            td_map = {}
            for key in self.replay_buffer.keys():
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
