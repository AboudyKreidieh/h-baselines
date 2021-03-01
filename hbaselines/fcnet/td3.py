"""TD3-compatible feedforward policy."""
import tensorflow as tf
import numpy as np

from hbaselines.base_policies import Policy
from hbaselines.fcnet.replay_buffer import ReplayBuffer
from hbaselines.utils.tf_util import create_fcnet
from hbaselines.utils.tf_util import create_conv
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.utils.tf_util import reduce_std
from hbaselines.utils.tf_util import print_params_shape
from hbaselines.utils.tf_util import setup_target_updates


class FeedForwardPolicy(Policy):
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
    l2_penalty : float
        L2 regularization penalty. This is applied to the policy network.
    model_params : dict
        dictionary of model-specific parameters. See parent class.
    noise : float
        scaling term to the range of the action space, that is subsequently
        used as the standard deviation of Gaussian noise added to the action if
        `apply_noise` is set to True in `get_action`
    target_policy_noise : float
        standard deviation term to the noise from the output of the target
        actor policy. See TD3 paper for more.
    target_noise_clip : float
        clipping term for the noise injected in the target actor policy
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
    phase_ph : tf.compat.v1.placeholder
        a placeholder that defines whether training is occurring for the batch
        normalization layer. Set to True in training and False in testing.
    rate_ph : tf.compat.v1.placeholder
        the probability that each element is dropped if dropout is implemented
    actor_tf : tf.Variable
        the output from the actor network
    critic_tf : list of tf.Variable
        the output from the critic networks. Two networks are used to stabilize
        training.
    critic_with_actor_tf : list of tf.Variable
        the output from the critic networks with the action provided directly
        by the actor policy
    target_init_updates : tf.Operation
        an operation that sets the values of the trainable parameters of the
        target actor/critic to match those actual actor/critic
    target_soft_updates : tf.Operation
        soft target update function
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
                 l2_penalty,
                 model_params,
                 noise,
                 target_policy_noise,
                 target_noise_clip,
                 scope=None,
                 num_envs=1):
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
        l2_penalty : float
            L2 regularization penalty. This is applied to the policy network.
        model_params : dict
            dictionary of model-specific parameters. See parent class.
        noise : float
            scaling term to the range of the action space, that is subsequently
            used as the standard deviation of Gaussian noise added to the
            action if `apply_noise` is set to True in `get_action`
        target_policy_noise : float
            standard deviation term to the noise from the output of the target
            actor policy. See TD3 paper for more.
        target_noise_clip : float
            clipping term for the noise injected in the target actor policy
        scope : str
            an upper-level scope term. Used by policies that call this one.
        """
        super(FeedForwardPolicy, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            co_space=co_space,
            verbose=verbose,
            l2_penalty=l2_penalty,
            model_params=model_params,
            num_envs=num_envs,
        )

        # action magnitudes
        ac_mag = 0.5 * (ac_space.high - ac_space.low)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = gamma
        self.use_huber = use_huber
        self.noise = noise * ac_mag
        self.target_policy_noise = np.array([ac_mag * target_policy_noise])
        self.target_noise_clip = np.array([ac_mag * target_noise_clip])

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
            self.phase_ph = tf.compat.v1.placeholder(
                tf.bool,
                name='phase')
            self.rate_ph = tf.compat.v1.placeholder(
                tf.float32,
                name='rate')

        # =================================================================== #
        # Step 3: Create actor and critic variables.                          #
        # =================================================================== #

        # Create networks and core TF parts that are shared across setup parts.
        with tf.compat.v1.variable_scope("model", reuse=False):
            self.actor_tf = self.make_actor(self.obs_ph)
            self.critic_tf = [
                self.make_critic(self.obs_ph, self.action_ph,
                                 scope="qf_{}".format(i))
                for i in range(2)
            ]
            self.critic_with_actor_tf = [
                self.make_critic(self.obs_ph, self.actor_tf, reuse=True,
                                 scope="qf_{}".format(i))
                for i in range(2)
            ]

        with tf.compat.v1.variable_scope("target", reuse=False):
            # create the target actor policy
            actor_target = self.make_actor(self.obs1_ph)

            # smooth target policy by adding clipped noise to target actions
            target_noise = tf.random.normal(
                tf.shape(actor_target), stddev=self.target_policy_noise)
            target_noise = tf.clip_by_value(
                target_noise, -self.target_noise_clip, self.target_noise_clip)

            # clip the noisy action to remain in the bounds
            noisy_actor_target = tf.clip_by_value(
                actor_target + target_noise,
                self.ac_space.low,
                self.ac_space.high
            )

            # create the target critic policies
            critic_target = [
                self.make_critic(self.obs1_ph, noisy_actor_target,
                                 scope="qf_{}".format(i))
                for i in range(2)
            ]

        # Create the target update operations.
        init, soft = setup_target_updates(
            'model', 'target', scope, tau, verbose)
        self.target_init_updates = init
        self.target_soft_updates = soft

        # =================================================================== #
        # Step 4: Setup the optimizers for the actor and critic.              #
        # =================================================================== #

        with tf.compat.v1.variable_scope("Optimizer", reuse=False):
            self._setup_actor_optimizer(scope)
            self._setup_critic_optimizer(critic_target, scope)

        # =================================================================== #
        # Step 5: Setup the operations for computing model statistics.        #
        # =================================================================== #

        # Setup the running means and standard deviations of the model inputs
        # and outputs.
        self.stats_ops, self.stats_names = self._setup_stats(scope or "Model")

    def _setup_actor_optimizer(self, scope):
        """Create the actor loss, gradient, and optimizer."""
        scope_name = 'model/pi/'
        if scope is not None:
            scope_name = scope + '/' + scope_name

        if self.verbose >= 2:
            print('setting up actor optimizer')
            print_params_shape(scope_name, "actor")

        # Compute the actor loss.
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf[0])

        # Add a regularization penalty.
        self.actor_loss += self._l2_loss(self.l2_penalty, scope_name)

        # Create an optimizer object.
        optimizer = tf.compat.v1.train.AdamOptimizer(self.actor_lr)

        self.actor_optimizer = optimizer.minimize(
            self.actor_loss,
            var_list=get_trainable_vars(scope_name)
        )

    def _setup_critic_optimizer(self, critic_target, scope):
        """Create the critic loss, gradient, and optimizer."""
        if self.verbose >= 2:
            print('setting up critic optimizer')

        # compute the target critic term
        with tf.compat.v1.variable_scope("loss", reuse=False):
            q_obs1 = tf.minimum(critic_target[0], critic_target[1])
            target_q = tf.stop_gradient(
                self.rew_ph + (1. - self.terminals1) * self.gamma * q_obs1)

            tf.compat.v1.summary.scalar('critic_target',
                                        tf.reduce_mean(target_q))

        # choose the loss function
        if self.use_huber:
            loss_fn = tf.compat.v1.losses.huber_loss
        else:
            loss_fn = tf.compat.v1.losses.mean_squared_error

        self.critic_loss = [loss_fn(q, target_q) for q in self.critic_tf]

        self.critic_optimizer = []

        for i, critic_loss in enumerate(self.critic_loss):
            scope_name = 'model/qf_{}/'.format(i)
            if scope is not None:
                scope_name = scope + '/' + scope_name

            if self.verbose >= 2:
                print_params_shape(scope_name, "critic {}".format(i))

            # create an optimizer object
            optimizer = tf.compat.v1.train.AdamOptimizer(self.critic_lr)

            # create the optimizer object
            self.critic_optimizer.append(optimizer.minimize(
                loss=critic_loss,
                var_list=get_trainable_vars(scope_name)))

    def make_actor(self, obs, reuse=False, scope="pi"):
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
                batch_norm=self.model_params["batch_norm"],
                phase=self.phase_ph,
                dropout=self.model_params["dropout"],
                rate=self.rate_ph,
                scope=scope,
                reuse=reuse,
            )
        else:
            pi_h = obs

        # Create the model.
        policy = create_fcnet(
            obs=pi_h,
            layers=self.model_params["layers"],
            num_output=self.ac_space.shape[0],
            stochastic=False,
            act_fun=self.model_params["act_fun"],
            layer_norm=self.model_params["layer_norm"],
            batch_norm=self.model_params["batch_norm"],
            phase=self.phase_ph,
            dropout=self.model_params["dropout"],
            rate=self.rate_ph,
            scope=scope,
            reuse=reuse,
        )

        # Scaling terms to the output from the policy.
        ac_means = (self.ac_space.high + self.ac_space.low) / 2.
        ac_magnitudes = (self.ac_space.high - self.ac_space.low) / 2.

        # Apply squashing and scale by action space.
        return ac_means + ac_magnitudes * tf.nn.tanh(policy)

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
        # Concatenate the observations and actions.
        qf_h = tf.concat([obs, action], axis=-1)

        # Initial image pre-processing (for convolutional policies).
        if self.model_params["model_type"] == "conv":
            qf_h = create_conv(
                obs=qf_h,
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
                batch_norm=self.model_params["batch_norm"],
                phase=self.phase_ph,
                dropout=self.model_params["dropout"],
                rate=self.rate_ph,
                scope=scope,
                reuse=reuse,
            )

        return create_fcnet(
            obs=qf_h,
            layers=self.model_params["layers"],
            num_output=1,
            stochastic=False,
            act_fun=self.model_params["act_fun"],
            layer_norm=self.model_params["layer_norm"],
            batch_norm=self.model_params["batch_norm"],
            phase=self.phase_ph,
            dropout=self.model_params["dropout"],
            rate=self.rate_ph,
            scope=scope,
            reuse=reuse,
            output_pre="qf_",
        )

    def update(self, update_actor=True, **kwargs):
        """Perform a gradient update step.

        **Note**; The target update soft updates occur at the same frequency as
        the actor update frequencies.

        Parameters
        ----------
        update_actor : bool
            specifies whether to update the actor policy. The critic policy is
            still updated if this value is set to False.
        """
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample():
            return

        # Get a batch
        obs0, actions, rewards, obs1, terminals1 = self.replay_buffer.sample()

        return self.update_from_batch(
            obs0, actions, rewards, obs1, terminals1, update_actor)

    def update_from_batch(self,
                          obs0,
                          actions,
                          rewards,
                          obs1,
                          terminals1,
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
        update_actor : bool, optional
            specified whether to perform gradient update procedures to the
            actor policy. Default set to True. Note that the update procedure
            for the critic is always performed when calling this method.
        """
        # Reshape to match previous behavior and placeholder shape.
        rewards = rewards.reshape(-1, 1)
        terminals1 = terminals1.reshape(-1, 1)

        # Update operations for the critic networks.
        step_ops = [self.critic_optimizer[0],
                    self.critic_optimizer[1]]

        if update_actor:
            # Actor updates and target soft update operation.
            step_ops += [self.actor_optimizer,
                         self.target_soft_updates]

        # Perform the update operations.
        self.sess.run(step_ops, feed_dict={
            self.obs_ph: obs0,
            self.action_ph: actions,
            self.rew_ph: rewards,
            self.obs1_ph: obs1,
            self.terminals1: terminals1,
            self.phase_ph: 1,
            self.rate_ph: 0.5,
        })

    def get_action(self, obs, context, apply_noise, random_actions, env_num=0):
        """See parent class."""
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        if random_actions:
            action = np.array([self.ac_space.sample()])
        else:
            action = self.sess.run(self.actor_tf, {
                self.obs_ph: obs,
                self.phase_ph: 0,
                self.rate_ph: 0.0,
            })

            if apply_noise:
                # compute noisy action
                if apply_noise:
                    action += np.random.normal(0, self.noise, action.shape)

                # clip by bounds
                action = np.clip(action, self.ac_space.low, self.ac_space.high)

        return action

    def store_transition(self, obs0, context0, action, reward, obs1, context1,
                         done, is_final_step, env_num=0, evaluate=False):
        """See parent class."""
        if not evaluate:
            # Add the contextual observation, if applicable.
            obs0 = self._get_obs(obs0, context0, axis=0)
            obs1 = self._get_obs(obs1, context1, axis=0)

            # Modify the done mask in accordance with the TD3 algorithm. Done
            # masks that correspond to the final step are set to False.
            done = done and not is_final_step

            self.replay_buffer.add(obs0, action, reward, obs1, float(done))

    def initialize(self):
        """See parent class.

        This method initializes the target parameters to match the model
        parameters.
        """
        self.sess.run(self.target_init_updates)

    def _setup_stats(self, base):
        """Create the running means and std of the model inputs and outputs.

        This method also adds the same running means and stds as scalars to
        tensorboard for additional storage.
        """
        ops = []
        names = []

        ops += [tf.reduce_mean(self.critic_tf[0])]
        names += ['{}/reference_Q1_mean'.format(base)]
        ops += [reduce_std(self.critic_tf[0])]
        names += ['{}/reference_Q1_std'.format(base)]

        ops += [tf.reduce_mean(self.critic_tf[1])]
        names += ['{}/reference_Q2_mean'.format(base)]
        ops += [reduce_std(self.critic_tf[1])]
        names += ['{}/reference_Q2_std'.format(base)]

        ops += [tf.reduce_mean(self.critic_with_actor_tf[0])]
        names += ['{}/reference_actor_Q1_mean'.format(base)]
        ops += [reduce_std(self.critic_with_actor_tf[0])]
        names += ['{}/reference_actor_Q1_std'.format(base)]

        ops += [tf.reduce_mean(self.critic_with_actor_tf[1])]
        names += ['{}/reference_actor_Q2_mean'.format(base)]
        ops += [reduce_std(self.critic_with_actor_tf[1])]
        names += ['{}/reference_actor_Q2_std'.format(base)]

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['{}/reference_action_mean'.format(base)]
        ops += [reduce_std(self.actor_tf)]
        names += ['{}/reference_action_std'.format(base)]

        ops += [tf.reduce_mean(self.rew_ph)]
        names += ['{}/rewards'.format(base)]

        ops += [self.actor_loss]
        names += ['{}/actor_loss'.format(base)]

        ops += [self.critic_loss[0]]
        names += ['{}/Q1_loss'.format(base)]

        ops += [self.critic_loss[1]]
        names += ['{}/Q2_loss'.format(base)]

        # Add all names and ops to the tensorboard summary.
        for op, name in zip(ops, names):
            tf.compat.v1.summary.scalar(name, op)

        return ops, names

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
            self.terminals1: terminals1,
            self.phase_ph: 0,
            self.rate_ph: 0.0,
        }

        return td_map
