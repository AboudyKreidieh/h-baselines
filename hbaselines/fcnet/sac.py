"""SAC-compatible feedforward policy."""
import tensorflow as tf
import numpy as np

from hbaselines.base_policies import ActorCriticPolicy
from hbaselines.fcnet.replay_buffer import ReplayBuffer
from hbaselines.utils.tf_util import create_fcnet
from hbaselines.utils.tf_util import create_conv
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.utils.tf_util import reduce_std
from hbaselines.utils.tf_util import gaussian_likelihood
from hbaselines.utils.tf_util import apply_squashing_func
from hbaselines.utils.tf_util import print_params_shape


# Cap the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class FeedForwardPolicy(ActorCriticPolicy):
    """Feed-forward neural network actor-critic policy.

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
    buffer_size : int
        the max number of transitions to store
    batch_size : int
        SGD batch size
    actor_lr : float
        actor learning rate
    critic_lr : float
        critic learning rate
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    tau : float
        target update rate
    gamma : float
        discount factor
    use_huber : bool
        specifies whether to use the huber distance function as the loss for
        the critic. If set to False, the mean-squared error metric is used
        instead
    model_params : dict
        dictionary of model-specific parameters. See parent class.
    target_entropy : float
        target entropy used when learning the entropy coefficient
    replay_buffer : hbaselines.fcnet.replay_buffer.ReplayBuffer
        the replay buffer
    terminals1 : tf.compat.v1.placeholder
        placeholder for the next step terminals
    rew_ph : tf.compat.v1.placeholder
        placeholder for the rewards
    action_ph : tf.compat.v1.placeholder
        placeholder for the actions
    obs_ph : tf.compat.v1.placeholder
        placeholder for the observations
    obs1_ph : tf.compat.v1.placeholder
        placeholder for the next step observations
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
    qf1_pi : tf.Variable
        the output from the first Q-function with the action provided directly
        by the actor policy
    qf2_pi : tf.Variable
        the output from the second Q-function with the action provided directly
        by the actor policy
    log_alpha : tf.Variable
        the log of the entropy coefficient
    alpha : tf.Variable
        the entropy coefficient
    value_target : tf.Variable
        the output from the target value function. Takes as input the next-step
        observations
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
    critic_loss : tf.Operation
        the operation that returns the loss of the critic
    critic_optimizer : tf.Operation
        the operation that updates the trainable parameters of the critic
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
                 target_entropy,
                 scope=None):
        """Instantiate the feed-forward neural network policy.

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
        use_huber : bool
            specifies whether to use the huber distance function as the loss
            for the critic. If set to False, the mean-squared error metric is
            used instead
        model_params : dict
            dictionary of model-specific parameters. See parent class.
        target_entropy : float
            target entropy used when learning the entropy coefficient. If set
            to None, a heuristic value is used.
        scope : str
            an upper-level scope term. Used by policies that call this one.
        """
        super(FeedForwardPolicy, self).__init__(
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

        if target_entropy is None:
            self.target_entropy = -np.prod(self.ac_space.shape)
        else:
            self.target_entropy = target_entropy

        self._ac_means = 0.5 * (ac_space.high + ac_space.low)
        self._ac_magnitudes = 0.5 * (ac_space.high - ac_space.low)

        # Compute the shape of the input observation space, which may include
        # the contextual term.
        ob_dim = self._get_ob_dim(ob_space, co_space)

        # =================================================================== #
        # Step 1: Create a replay buffer object.                              #
        # =================================================================== #

        self.replay_buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            obs_dim=ob_dim[0],
            ac_dim=self.ac_space.shape[0],
        )

        # =================================================================== #
        # Step 2: Create input variables.                                     #
        # =================================================================== #

        with tf.compat.v1.variable_scope("input", reuse=False):
            self.terminals1 = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='terminals1')
            self.rew_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='rewards')
            self.action_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ac_space.shape,
                name='actions')
            self.obs_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ob_dim,
                name='obs0')
            self.obs1_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ob_dim,
                name='obs1')

        # =================================================================== #
        # Step 3: Create actor and critic variables.                          #
        # =================================================================== #

        # Create networks and core TF parts that are shared across setup parts.
        with tf.compat.v1.variable_scope("model", reuse=False):
            self.deterministic_action, self.policy_out, self.logp_pi, \
                self.logp_action = self.make_actor(self.obs_ph, self.action_ph)
            self.qf1, self.qf2, self.value_fn = self.make_critic(
                self.obs_ph, self.action_ph,
                create_qf=True, create_vf=True)
            self.qf1_pi, self.qf2_pi, _ = self.make_critic(
                self.obs_ph, self.policy_out,
                create_qf=True, create_vf=False, reuse=True)

            # The entropy coefficient or entropy can be learned automatically,
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            self.log_alpha = tf.compat.v1.get_variable(
                'log_alpha',
                dtype=tf.float32,
                initializer=0.0)
            self.alpha = tf.exp(self.log_alpha)

        with tf.compat.v1.variable_scope("target", reuse=False):
            # Create the value network
            _, _, value_target = self.make_critic(
                self.obs1_ph, create_qf=False, create_vf=True)
            self.value_target = value_target

        # Create the target update operations.
        init, soft = self._setup_target_updates(
            'model/value_fns/vf', 'target/value_fns/vf', scope, tau, verbose)
        self.target_init_updates = init
        self.target_soft_updates = soft

        # =================================================================== #
        # Step 4: Setup the optimizers for the actor and critic.              #
        # =================================================================== #

        with tf.compat.v1.variable_scope("Optimizer", reuse=False):
            self._setup_actor_optimizer(scope)
            self._setup_critic_optimizer(scope)

        # =================================================================== #
        # Step 5: Setup the operations for computing model statistics.        #
        # =================================================================== #

        # Setup the running means and standard deviations of the model inputs
        # and outputs.
        self.stats_ops, self.stats_names = self._setup_stats(scope or "Model")

    def make_actor(self, obs, action, reuse=False, scope="pi"):
        """Create the actor variables.

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
            the output from the deterministic actor
        tf.Variable
            the output from the stochastic actor
        tf.Variable
            the log-probability of a given observation given the output action
            from the policy
        tf.Variable
            the log-probability of a given observation given a fixed action
        """
        # Initial image pre-processing (for convolutional policies).
        if self.model_params["model_type"] == "conv":
            pi_h = create_conv(
                obs=obs,
                image_height=self.model_params["image_height"],
                image_width=self.model_params["image_width"],
                image_channels=self.model_params["image_channels"],
                ignore_flat_channels=self.model_params["ignore_flat_channels"],
                ignore_image=self.model_params["ignore_image"],
                filters=self.model_params["filters"],
                kernel_sizes=self.model_params["kernel_sizes"],
                strides=self.model_params["strides"],
                act_fun=self.model_params["act_fun"],
                layer_norm=self.model_params["layer_norm"],
                scope=scope,
                reuse=reuse,
            )
        else:
            pi_h = obs

        # Create the model.
        policy_mean, log_std = create_fcnet(
            obs=pi_h,
            layers=self.model_params["layers"],
            num_output=self.ac_space.shape[0],
            stochastic=True,
            act_fun=self.model_params["act_fun"],
            layer_norm=self.model_params["layer_norm"],
            scope=scope,
            reuse=reuse,
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
        conv_params = dict(
            image_height=self.model_params["image_height"],
            image_width=self.model_params["image_width"],
            image_channels=self.model_params["image_channels"],
            ignore_flat_channels=self.model_params["ignore_flat_channels"],
            ignore_image=self.model_params["ignore_image"],
            filters=self.model_params["filters"],
            kernel_sizes=self.model_params["kernel_sizes"],
            strides=self.model_params["strides"],
            act_fun=self.model_params["act_fun"],
            layer_norm=self.model_params["layer_norm"],
            reuse=reuse,
        )

        fcnet_params = dict(
            layers=self.model_params["layers"],
            num_output=1,
            stochastic=False,
            act_fun=self.model_params["act_fun"],
            layer_norm=self.model_params["layer_norm"],
            reuse=reuse,
        )

        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Value function
            if create_vf:
                if self.model_params["model_type"] == "conv":
                    vf_h = create_conv(obs=obs, scope="vf", **conv_params)
                else:
                    vf_h = obs

                value_fn = create_fcnet(
                    obs=vf_h, scope="vf", output_pre="vf_", **fcnet_params)
            else:
                value_fn = None

            # Double Q values to reduce overestimation
            if create_qf:
                # Concatenate the observations and actions.
                qf_h = tf.concat([obs, action], axis=-1)

                if self.model_params["model_type"] == "conv":
                    qf1_h = create_conv(obs=qf_h, scope="qf1", **conv_params)
                    qf2_h = create_conv(obs=qf_h, scope="qf2", **conv_params)
                else:
                    qf1_h = qf_h
                    qf2_h = qf_h

                qf1 = create_fcnet(
                    obs=qf1_h, scope="qf1", output_pre="qf_", **fcnet_params)
                qf2 = create_fcnet(
                    obs=qf2_h, scope="qf2", output_pre="qf_", **fcnet_params)
            else:
                qf1, qf2 = None, None

        return qf1, qf2, value_fn

    def update(self, **kwargs):
        """Perform a gradient update step.

        Returns
        -------
        [float, float]
            Q1 loss, Q2 loss
        float
            actor loss
        """
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample():
            return [0, 0], 0

        # Get a batch
        obs0, actions, rewards, obs1, done1 = self.replay_buffer.sample()

        return self.update_from_batch(obs0, actions, rewards, obs1, done1)

    def update_from_batch(self, obs0, actions, rewards, obs1, terminals1,
                          update_actor=True):
        """Perform gradient update step given a batch of data.

        Parameters
        ----------
        obs0 : array_like
            batch of observations
        actions : array_like
            batch of actions executed given obs_batch
        rewards : array_like
            rewards received as results of executing act_batch
        obs1 : array_like
            next set of observations seen after executing act_batch
        terminals1 : numpy bool
            done_mask[i] = 1 if executing act_batch[i] resulted in the end of
            an episode and 0 otherwise.
        update_actor : bool
            whether to update the actor policy. Unused by this method.

        Returns
        -------
        [float, float]
            Q1 loss, Q2 loss
        float
            actor loss
        """
        del update_actor  # unused by this method

        # Normalize the actions (bounded between [-1, 1]).
        actions = (actions - self._ac_means) / self._ac_magnitudes

        # Reshape to match previous behavior and placeholder shape.
        rewards = rewards.reshape(-1, 1)
        terminals1 = terminals1.reshape(-1, 1)

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
            self.obs_ph: obs0,
            self.action_ph: actions,
            self.rew_ph: rewards,
            self.obs1_ph: obs1,
            self.terminals1: terminals1
        }

        # Perform the update operations and collect the actor and critic loss.
        q1_loss, q2_loss, vf_loss, actor_loss, *_ = self.sess.run(
            step_ops, feed_dict)

        return [q1_loss, q2_loss], actor_loss

    def get_action(self, obs, context, apply_noise, random_actions, env_num=0):
        """See parent class."""
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        if random_actions:
            return np.array([self.ac_space.sample()])
        elif apply_noise:
            normalized_action = self.sess.run(
                self.policy_out, feed_dict={self.obs_ph: obs})
            return self._ac_magnitudes * normalized_action + self._ac_means
        else:
            normalized_action = self.sess.run(
                self.deterministic_action, feed_dict={self.obs_ph: obs})
            return self._ac_magnitudes * normalized_action + self._ac_means

    def _setup_critic_optimizer(self, scope):
        """Create minimization operation for critic Q-function.

        Create a `tf.optimizer.minimize` operation for updating critic
        Q-function with gradient descent.

        See Equations (5, 6) in [1], for further information of the Q-function
        update rule.
        """
        scope_name = 'model/value_fns'
        if scope is not None:
            scope_name = scope + '/' + scope_name

        if self.verbose >= 2:
            print('setting up critic optimizer')
            for name in ['qf1', 'qf2', 'vf']:
                scope_i = '{}/{}'.format(scope_name, name)
                print_params_shape(scope_i, name)

        # Take the min of the two Q-Values (Double-Q Learning)
        min_qf_pi = tf.minimum(self.qf1_pi, self.qf2_pi)

        # Target for Q value regression
        q_backup = tf.stop_gradient(
            self.rew_ph +
            (1 - self.terminals1) * self.gamma * self.value_target)

        # choose the loss function
        if self.use_huber:
            loss_fn = tf.compat.v1.losses.huber_loss
        else:
            loss_fn = tf.compat.v1.losses.mean_squared_error

        # Compute Q-Function loss
        qf1_loss = loss_fn(q_backup, self.qf1)
        qf2_loss = loss_fn(q_backup, self.qf2)

        # Target for value fn regression
        # We update the vf towards the min of two Q-functions in order to
        # reduce overestimation bias from function approximation error.
        v_backup = tf.stop_gradient(min_qf_pi - self.alpha * self.logp_pi)
        value_loss = loss_fn(self.value_fn, v_backup)

        self.critic_loss = (qf1_loss, qf2_loss, value_loss)

        # Combine the loss functions for the optimizer.
        critic_loss = qf1_loss + qf2_loss + value_loss

        # Critic train op
        critic_optimizer = tf.compat.v1.train.AdamOptimizer(self.critic_lr)
        self.critic_optimizer = critic_optimizer.minimize(
            critic_loss,
            var_list=get_trainable_vars(scope_name))

    def _setup_actor_optimizer(self, scope):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating policy and
        entropy with gradient descent.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """
        scope_name = 'model/pi/'
        if scope is not None:
            scope_name = scope + '/' + scope_name

        if self.verbose >= 2:
            print('setting up actor and alpha optimizers')
            print_params_shape(scope_name, "actor")

        # Take the min of the two Q-Values (Double-Q Learning)
        min_qf_pi = tf.minimum(self.qf1_pi, self.qf2_pi)

        # Compute the entropy temperature loss.
        self.alpha_loss = -tf.reduce_mean(
            self.log_alpha
            * tf.stop_gradient(self.logp_pi + self.target_entropy))

        alpha_optimizer = tf.compat.v1.train.AdamOptimizer(self.actor_lr)

        self.alpha_optimizer = alpha_optimizer.minimize(
            self.alpha_loss,
            var_list=self.log_alpha)

        # Compute the policy loss
        self.actor_loss = tf.reduce_mean(self.alpha * self.logp_pi - min_qf_pi)

        # Policy train op (has to be separate from value train op, because
        # min_qf_pi appears in policy_loss)
        actor_optimizer = tf.compat.v1.train.AdamOptimizer(self.actor_lr)

        self.actor_optimizer = actor_optimizer.minimize(
            self.actor_loss,
            var_list=get_trainable_vars(scope_name))

    def _setup_stats(self, base):
        """Create the running means and std of the model inputs and outputs.

        This method also adds the same running means and stds as scalars to
        tensorboard for additional storage.
        """
        ops = []
        names = []

        ops += [tf.reduce_mean(self.qf1)]
        names += ['{}/reference_Q1_mean'.format(base)]
        ops += [reduce_std(self.qf1)]
        names += ['{}/reference_Q1_std'.format(base)]

        ops += [tf.reduce_mean(self.qf2)]
        names += ['{}/reference_Q2_mean'.format(base)]
        ops += [reduce_std(self.qf2)]
        names += ['{}/reference_Q2_std'.format(base)]

        ops += [tf.reduce_mean(self.qf1_pi)]
        names += ['{}/reference_actor_Q1_mean'.format(base)]
        ops += [reduce_std(self.qf1_pi)]
        names += ['{}/reference_actor_Q1_std'.format(base)]

        ops += [tf.reduce_mean(self.qf2_pi)]
        names += ['{}/reference_actor_Q2_mean'.format(base)]
        ops += [reduce_std(self.qf2_pi)]
        names += ['{}/reference_actor_Q2_std'.format(base)]

        ops += [tf.reduce_mean(
            self._ac_magnitudes * self.policy_out + self._ac_means)]
        names += ['{}/reference_action_mean'.format(base)]
        ops += [reduce_std(
            self._ac_magnitudes * self.policy_out + self._ac_means)]
        names += ['{}/reference_action_std'.format(base)]

        ops += [tf.reduce_mean(self.logp_pi)]
        names += ['{}/reference_log_probability_mean'.format(base)]
        ops += [reduce_std(self.logp_pi)]
        names += ['{}/reference_log_probability_std'.format(base)]

        ops += [tf.reduce_mean(self.rew_ph)]
        names += ['{}/rewards'.format(base)]

        ops += [self.alpha_loss]
        names += ['{}/alpha_loss'.format(base)]

        ops += [self.actor_loss]
        names += ['{}/actor_loss'.format(base)]

        ops += [self.critic_loss[0]]
        names += ['{}/Q1_loss'.format(base)]

        ops += [self.critic_loss[1]]
        names += ['{}/Q2_loss'.format(base)]
        tf.compat.v1.summary.scalar('Q2_loss', self.critic_loss[1])

        ops += [self.critic_loss[2]]
        names += ['{}/value_loss'.format(base)]

        # Add all names and ops to the tensorboard summary.
        for op, name in zip(ops, names):
            tf.compat.v1.summary.scalar(name, op)

        return ops, names

    def initialize(self):
        """See parent class."""
        self.sess.run(self.target_init_updates)

    def store_transition(self, obs0, context0, action, reward, obs1, context1,
                         done, is_final_step, env_num=0, evaluate=False):
        """See parent class."""
        if not evaluate:
            # Add the contextual observation, if applicable.
            obs0 = self._get_obs(obs0, context0, axis=0)
            obs1 = self._get_obs(obs1, context1, axis=0)

            self.replay_buffer.add(obs0, action, reward, obs1, float(done))

    def get_td_map(self):
        """See parent class."""
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample():
            return {}

        # Get a batch.
        obs0, actions, rewards, obs1, done1 = self.replay_buffer.sample()

        return self.get_td_map_from_batch(obs0, actions, rewards, obs1, done1)

    def get_td_map_from_batch(self, obs0, actions, rewards, obs1, terminals1):
        """Convert a batch to a td_map."""
        # Reshape to match previous behavior and placeholder shape.
        rewards = rewards.reshape(-1, 1)
        terminals1 = terminals1.reshape(-1, 1)

        td_map = {
            self.obs_ph: obs0,
            self.action_ph: actions,
            self.rew_ph: rewards,
            self.obs1_ph: obs1,
            self.terminals1: terminals1
        }

        return td_map
