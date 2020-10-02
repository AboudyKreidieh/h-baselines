"""PPO-compatible feedforward policy."""
import numpy as np
import tensorflow as tf

from hbaselines.base_policies.policy import Policy
from hbaselines.utils.tf_util import create_fcnet
from hbaselines.utils.tf_util import create_conv
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.utils.tf_util import explained_variance
from hbaselines.utils.tf_util import print_params_shape


class FeedForwardPolicy(Policy):
    """Feed-forward neural network policy.

    Attributes
    ----------
    learning_rate : float
        the learning rate
    n_minibatches : int
        number of training minibatches per update
    n_opt_epochs : int
        number of training epochs per update procedure
    gamma : float
        the discount factor
    lam : float
        factor for trade-off of bias vs variance for Generalized Advantage
        Estimator
    ent_coef : float
        entropy coefficient for the loss calculation
    vf_coef : float
        value function coefficient for the loss calculation
    max_grad_norm : float
        the maximum value for the gradient clipping
    cliprange : float or callable
        clipping parameter, it can be a function
    cliprange_vf : float or callable
        clipping parameter for the value function, it can be a function. This
        is a parameter specific to the OpenAI implementation. If None is passed
        (default), then `cliprange` (that is used for the policy) will be used.
        IMPORTANT: this clipping depends on the reward scaling. To deactivate
        value function clipping (and recover the original PPO implementation),
        you have to pass a negative value (e.g. -1).
    num_envs : int
        number of environments used to run simulations in parallel.
    mb_rewards : array_like
        a minibatch of environment rewards
    mb_obs : array_like
        a minibatch of observations
    mb_contexts : array_like
        a minibatch of contextual terms
    mb_actions : array_like
        a minibatch of actions
    mb_values : array_like
        a minibatch of estimated values by the policy
    mb_neglogpacs : array_like
        a minibatch of the negative log-likelihood of performed actions
    mb_dones : array_like
        a minibatch of done masks
    mb_all_obs : array_like
        a minibatch of full-state observations
    mb_returns : array_like
        a minibatch of expected discounted returns
    last_obs : array_like
        the most recent observation from each environment. Used to compute the
        GAE terms.
    mb_advs : array_like
        a minibatch of estimated advantages
    rew_ph : tf.compat.v1.placeholder
        placeholder for the rewards / discounted returns
    action_ph : tf.compat.v1.placeholder
        placeholder for the actions
    obs_ph : tf.compat.v1.placeholder
        placeholder for the observations
    advs_ph : tf.compat.v1.placeholder
        placeholder for the advantages
    old_neglog_pac_ph : tf.compat.v1.placeholder
        placeholder for the negative-log probability of the actions that were
        performed
    old_vpred_ph : tf.compat.v1.placeholder
        placeholder for the current predictions of the values of given states
    action : tf.Variable
        the output from the policy/actor
    pi_mean : tf.Variable
        the output from the policy's mean term
    pi_logstd : tf.Variable
        the output from the policy's log-std term
    pi_std : tf.Variable
        the expnonential of the pi_logstd term
    neglogp : tf.Variable
        a differentiable form of the negative log-probability of actions by the
        current policy
    value_fn : tf.Variable
        the output from the value function
    value_flat : tf.Variable
        a one-dimensional (vector) version of value_fn
    entropy : tf.Variable
        computes the entropy of actions performed by the policy
    vf_loss : tf.Variable
        the output from the computed value function loss of a batch of data
    pg_loss : tf.Variable
        the output from the computed policy gradient loss of a batch of data
    approxkl : tf.Variable
        computes the KL-divergence between two models
    loss : tf.Variable
        the output from the computed loss of a batch of data
    optimizer : tf.Operation
        the operation that updates the trainable parameters of the actor
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 co_space,
                 verbose,
                 learning_rate,
                 n_minibatches,
                 n_opt_epochs,
                 gamma,
                 lam,
                 ent_coef,
                 vf_coef,
                 max_grad_norm,
                 cliprange,
                 cliprange_vf,
                 l2_penalty,
                 model_params,
                 scope=None,
                 num_envs=1):
        """Instantiate the policy object.

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
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        learning_rate : float
            the learning rate
        n_minibatches : int
            number of training minibatches per update
        n_opt_epochs : int
            number of training epochs per update procedure
        gamma : float
            the discount factor
        lam : float
            factor for trade-off of bias vs variance for Generalized Advantage
            Estimator
        ent_coef : float
            entropy coefficient for the loss calculation
        vf_coef : float
            value function coefficient for the loss calculation
        max_grad_norm : float
            the maximum value for the gradient clipping
        cliprange : float or callable
            clipping parameter, it can be a function
        cliprange_vf : float or callable
            clipping parameter for the value function, it can be a function.
            This is a parameter specific to the OpenAI implementation. If None
            is passed (default), then `cliprange` (that is used for the policy)
            will be used. IMPORTANT: this clipping depends on the reward
            scaling. To deactivate value function clipping (and recover the
            original PPO implementation), you have to pass a negative value
            (e.g. -1).
        l2_penalty : float
            L2 regularization penalty. This is applied to the policy network.
        """
        super(FeedForwardPolicy, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            co_space=co_space,
            verbose=verbose,
            l2_penalty=l2_penalty,
            model_params=model_params,
        )

        self.learning_rate = learning_rate
        self.n_minibatches = n_minibatches
        self.n_opt_epochs = n_opt_epochs
        self.gamma = gamma
        self.lam = lam
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.cliprange = cliprange
        self.cliprange_vf = cliprange_vf
        self.num_envs = num_envs

        # Create variables to store on-policy data.
        self.mb_rewards = [[] for _ in range(num_envs)]
        self.mb_obs = [[] for _ in range(num_envs)]
        self.mb_contexts = [[] for _ in range(num_envs)]
        self.mb_actions = [[] for _ in range(num_envs)]
        self.mb_values = [[] for _ in range(num_envs)]
        self.mb_neglogpacs = [[] for _ in range(num_envs)]
        self.mb_dones = [[] for _ in range(num_envs)]
        self.mb_all_obs = [[] for _ in range(num_envs)]
        self.mb_returns = [[] for _ in range(num_envs)]
        self.last_obs = [None for _ in range(num_envs)]
        self.mb_advs = None

        # Compute the shape of the input observation space, which may include
        # the contextual term.
        ob_dim = self._get_ob_dim(ob_space, co_space)

        # =================================================================== #
        # Step 1: Create input variables.                                     #
        # =================================================================== #

        with tf.compat.v1.variable_scope("input", reuse=False):
            self.rew_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,),
                name='rewards')
            self.action_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ac_space.shape,
                name='actions')
            self.obs_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ob_dim,
                name='obs0')
            self.advs_ph = tf.placeholder(
                tf.float32,
                shape=(None,),
                name="advs_ph")
            self.old_neglog_pac_ph = tf.placeholder(
                tf.float32,
                shape=(None,),
                name="old_neglog_pac_ph")
            self.old_vpred_ph = tf.placeholder(
                tf.float32,
                shape=(None,),
                name="old_vpred_ph")

        # =================================================================== #
        # Step 2: Create actor and critic variables.                          #
        # =================================================================== #

        # Create networks and core TF parts that are shared across setup parts.
        with tf.variable_scope("model", reuse=False):
            # Create the policy.
            self.action, self.pi_mean, self.pi_logstd = self.make_actor(
                self.obs_ph, scope="pi")
            self.pi_std = tf.exp(self.pi_logstd)

            # Create a method the log-probability of current actions.
            self.neglogp = self._neglogp(self.action)

            # Create the value function.
            self.value_fn = self.make_critic(self.obs_ph, scope="vf")
            self.value_flat = self.value_fn[:, 0]

        # =================================================================== #
        # Step 4: Setup the optimizers for the actor and critic.              #
        # =================================================================== #

        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.approxkl = None
        self.loss = None
        self.optimizer = None

        with tf.compat.v1.variable_scope("Optimizer", reuse=False):
            self._setup_optimizers(scope)

        # =================================================================== #
        # Step 5: Setup the operations for computing model statistics.        #
        # =================================================================== #

        self._setup_stats(scope or "Model")

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
                scope=scope,
                reuse=reuse,
            )
        else:
            pi_h = obs

        # Create the output mean.
        policy_mean = create_fcnet(
            obs=pi_h,
            layers=self.model_params["layers"],
            num_output=self.ac_space.shape[0],
            stochastic=False,
            act_fun=self.model_params["act_fun"],
            layer_norm=self.model_params["layer_norm"],
            scope=scope,
            reuse=reuse,
        )

        # Create the output log_std.
        log_std = tf.get_variable(
            name='logstd',
            shape=[1, self.ac_space.shape[0]],
            initializer=tf.zeros_initializer()
        )

        # Create a method to sample from the distribution.
        std = tf.exp(log_std)
        action = policy_mean + std * tf.random_normal(
            shape=tf.shape(policy_mean),
            dtype=tf.float32
        )

        return action, policy_mean, log_std

    def make_critic(self, obs, reuse=False, scope="qf"):
        """Create a critic tensor.

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
            the output from the critic
        """
        # Initial image pre-processing (for convolutional policies).
        if self.model_params["model_type"] == "conv":
            vf_h = create_conv(
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
            vf_h = obs

        return create_fcnet(
            obs=vf_h,
            layers=self.model_params["layers"],
            num_output=1,
            stochastic=False,
            act_fun=self.model_params["act_fun"],
            layer_norm=self.model_params["layer_norm"],
            scope=scope,
            reuse=reuse,
        )

    def _neglogp(self, x):
        """Compute the negative log-probability of an input action (x)."""
        return 0.5 * tf.reduce_sum(
            tf.square((x - self.pi_mean) / self.pi_std), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) \
            * tf.cast(tf.shape(x)[-1], tf.float32) \
            + tf.reduce_sum(self.pi_logstd, axis=-1)

    def _setup_optimizers(self, scope):
        """Create the actor and critic optimizers."""
        scope_name = 'model/'
        if scope is not None:
            scope_name = scope + '/' + scope_name

        if self.verbose >= 2:
            print('setting up actor optimizer')
            print_params_shape("{}pi/".format(scope_name), "actor")
            print('setting up critic optimizer')
            print_params_shape("{}vf/".format(scope_name), "critic")

        neglogpac = self._neglogp(self.action_ph)
        self.entropy = tf.reduce_sum(
            tf.reshape(self.pi_logstd, [-1])
            + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

        # Value function clipping: not present in the original PPO
        if self.cliprange_vf is None:
            # Default behavior (legacy from OpenAI baselines):
            # use the same clipping as for the policy
            self.cliprange_vf = self.cliprange

        if self.cliprange_vf < 0:
            # Original PPO implementation: no value function clipping.
            vpred_clipped = self.value_flat
        else:
            # Clip the different between old and new value
            # NOTE: this depends on the reward scaling
            vpred_clipped = self.old_vpred_ph + tf.clip_by_value(
                self.value_flat - self.old_vpred_ph,
                -self.cliprange_vf, self.cliprange_vf)

        vf_losses1 = tf.square(self.value_flat - self.rew_ph)
        vf_losses2 = tf.square(vpred_clipped - self.rew_ph)
        self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
        pg_losses = -self.advs_ph * ratio
        pg_losses2 = -self.advs_ph * tf.clip_by_value(
            ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        self.approxkl = .5 * tf.reduce_mean(
            tf.square(neglogpac - self.old_neglog_pac_ph))
        self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(
            tf.abs(ratio - 1.0), self.cliprange), tf.float32))
        self.loss = self.pg_loss - self.entropy * self.ent_coef \
            + self.vf_loss * self.vf_coef

        # Add a regularization penalty.
        self.loss += self._l2_loss(self.l2_penalty, scope_name)

        # Compute the gradients of the loss.
        var_list = get_trainable_vars(scope_name)
        grads = tf.gradients(self.loss, var_list)

        # Perform gradient clipping if requested.
        if self.max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(
                grads, self.max_grad_norm)
        grads = list(zip(grads, var_list))

        # Create the operation that applies the gradients.
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            epsilon=1e-5
        ).apply_gradients(grads)

    def _setup_stats(self, base):
        """Create the running means and std of the model inputs and outputs.

        This method also adds the same running means and stds as scalars to
        tensorboard for additional storage.
        """
        ops = {
            'reference_action_mean': tf.reduce_mean(self.pi_mean),
            'reference_action_std': tf.reduce_mean(self.pi_logstd),
            'rewards': tf.reduce_mean(self.rew_ph),
            'advantage': tf.reduce_mean(self.advs_ph),
            'old_neglog_action_probability': tf.reduce_mean(
                self.old_neglog_pac_ph),
            'old_value_pred': tf.reduce_mean(self.old_vpred_ph),
            'entropy_loss': self.entropy,
            'policy_gradient_loss': self.pg_loss,
            'value_function_loss': self.vf_loss,
            'approximate_kullback-leibler': self.approxkl,
            'clip_factor': self.clipfrac,
            'loss': self.loss,
            'explained_variance': explained_variance(
                self.old_vpred_ph, self.rew_ph)
        }

        # Add all names and ops to the tensorboard summary.
        for key in ops.keys():
            name = "{}/{}".format(base, key)
            op = ops[key]
            tf.compat.v1.summary.scalar(name, op)

    def initialize(self):
        """See parent class."""
        pass

    def get_action(self, obs, context, apply_noise, random_actions, env_num=0):
        """See parent class."""
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        action, values, neglogpacs = self.sess.run(
            [self.action if apply_noise else self.pi_mean,
             self.value_flat, self.neglogp],
            {self.obs_ph: obs}
        )

        # Store information on the values and negative-log likelihood.
        self.mb_values[env_num].append(values)
        self.mb_neglogpacs[env_num].append(neglogpacs)

        return action

    def value(self, obs, context):
        """See parent class."""
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    def store_transition(self, obs0, context0, action, reward, obs1, context1,
                         done, is_final_step, env_num=0, evaluate=False):
        """Store a transition in the replay buffer.

        Parameters
        ----------
        obs0 : array_like
            the last observation
        context0 : array_like or None
            the last contextual term. Set to None if no context is provided by
            the environment.
        action : array_like
            the action
        reward : float
            the reward
        obs1 : array_like
            the current observation
        context1 : array_like or None
            the current contextual term. Set to None if no context is provided
            by the environment.
        done : float
            is the episode done
        is_final_step : bool
            whether the time horizon was met in the step corresponding to the
            current sample. This is used by the TD3 algorithm to augment the
            done mask.
        env_num : int
            the environment number. Used to handle situations when multiple
            parallel environments are being used.
        evaluate : bool
            whether the sample is being provided by the evaluation environment.
            If so, the data is not stored in the replay buffer.
        """
        # Update the minibatch of samples.
        self.mb_rewards[env_num].append(reward)
        self.mb_obs[env_num].append([obs0])
        self.mb_contexts[env_num].append(context0)
        self.mb_actions[env_num].append([action])
        self.mb_dones[env_num].append(done)

        # Update the last observation (to compute the last value for the GAE
        # expected returns).
        self.last_obs[env_num] = self._get_obs([obs1], context1)

    def update(self, **kwargs):
        """See parent class."""
        n_steps = 0

        for env_num in range(self.num_envs):
            # Convert the data to numpy arrays.
            self.mb_obs[env_num] = \
                np.concatenate(self.mb_obs[env_num], axis=0)
            self.mb_rewards[env_num] = \
                np.asarray(self.mb_rewards[env_num])
            self.mb_actions[env_num] = \
                np.concatenate(self.mb_actions[env_num], axis=0)
            self.mb_values[env_num] = \
                np.concatenate(self.mb_values[env_num], axis=0)
            self.mb_neglogpacs[env_num] = \
                np.concatenate(self.mb_neglogpacs[env_num], axis=0)
            self.mb_dones[env_num] = \
                np.asarray(self.mb_dones[env_num])
            n_steps += self.mb_obs[env_num].shape[0]

            # Compute the bootstrapped/discounted returns.
            self.mb_returns[env_num] = self._gae_returns(
                mb_rewards=self.mb_rewards[env_num],
                mb_values=self.mb_values[env_num],
                mb_dones=self.mb_dones[env_num],
                obs=self.last_obs[env_num],
            )

        # Concatenate the stored data.
        if self.num_envs > 1:
            self.mb_obs = np.concatenate(self.mb_obs, axis=0)
            self.mb_contexts = np.concatenate(self.mb_contexts, axis=0)
            self.mb_actions = np.concatenate(self.mb_actions, axis=0)
            self.mb_values = np.concatenate(self.mb_values, axis=0)
            self.mb_neglogpacs = np.concatenate(self.mb_neglogpacs, axis=0)
            self.mb_all_obs = np.concatenate(self.mb_all_obs, axis=0)
            self.mb_returns = np.concatenate(self.mb_returns, axis=0)
        else:
            self.mb_obs = self.mb_obs[0]
            self.mb_contexts = self.mb_contexts[0]
            self.mb_actions = self.mb_actions[0]
            self.mb_values = self.mb_values[0]
            self.mb_neglogpacs = self.mb_neglogpacs[0]
            self.mb_all_obs = self.mb_all_obs[0]
            self.mb_returns = self.mb_returns[0]

        # Compute the advantages.
        advs = self.mb_returns - self.mb_values
        self.mb_advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Run the optimization procedure.
        batch_size = n_steps // self.n_minibatches

        inds = np.arange(n_steps)
        for _ in range(self.n_opt_epochs):
            np.random.shuffle(inds)
            for start in range(0, n_steps, batch_size):
                end = start + batch_size
                mbinds = inds[start:end]
                self.update_from_batch(
                    obs=self.mb_obs[mbinds],
                    context=None if self.mb_contexts[0] is None
                    else self.mb_contexts[mbinds],
                    returns=self.mb_returns[mbinds],
                    actions=self.mb_actions[mbinds],
                    values=self.mb_values[mbinds],
                    advs=self.mb_advs[mbinds],
                    neglogpacs=self.mb_neglogpacs[mbinds],
                )

    def update_from_batch(self,
                          obs,
                          context,
                          returns,
                          actions,
                          values,
                          advs,
                          neglogpacs):
        """Perform gradient update step given a batch of data.

        Parameters
        ----------
        obs : array_like
            a minibatch of observations
        context : array_like
            a minibatch of contextual terms
        returns : array_like
            a minibatch of contextual expected discounted returns
        actions : array_like
            a minibatch of actions
        values : array_like
            a minibatch of estimated values by the policy
        advs : array_like
            a minibatch of estimated advantages
        neglogpacs : array_like
            a minibatch of the negative log-likelihood of performed actions
        """
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        return self.sess.run(self.optimizer, {
            self.obs_ph: obs,
            self.action_ph: actions,
            self.advs_ph: advs,
            self.rew_ph: returns,
            self.old_neglog_pac_ph: neglogpacs,
            self.old_vpred_ph: values,
        })

    def get_td_map(self):
        """See parent class."""
        # Add the contextual observation, if applicable.
        context = None if self.mb_contexts[0] is None else self.mb_contexts
        obs = self._get_obs(self.mb_obs, context, axis=1)

        td_map = {
            self.obs_ph: obs.copy(),
            self.action_ph: self.mb_actions,
            self.advs_ph: self.mb_advs,
            self.rew_ph: self.mb_returns,
            self.old_neglog_pac_ph: self.mb_neglogpacs,
            self.old_vpred_ph: self.mb_values,
        }

        # Clear memory
        self.mb_rewards = [[] for _ in range(self.num_envs)]
        self.mb_obs = [[] for _ in range(self.num_envs)]
        self.mb_contexts = [[] for _ in range(self.num_envs)]
        self.mb_actions = [[] for _ in range(self.num_envs)]
        self.mb_values = [[] for _ in range(self.num_envs)]
        self.mb_neglogpacs = [[] for _ in range(self.num_envs)]
        self.mb_dones = [[] for _ in range(self.num_envs)]
        self.mb_all_obs = [[] for _ in range(self.num_envs)]
        self.mb_returns = [[] for _ in range(self.num_envs)]
        self.last_obs = [None for _ in range(self.num_envs)]
        self.mb_advs = None

        return td_map

    def _gae_returns(self, mb_rewards, mb_values, mb_dones, obs):
        """Compute the bootstrapped/discounted returns.

        Parameters
        ----------
        mb_rewards : array_like
            a minibatch of rewards from a given environment
        mb_values : array_like
            a minibatch of values computed by the policy from a given
            environment
        mb_dones : array_like
            a minibatch of done masks from a given environment
        obs : array_like
            the current observation within the environment

        Returns
        -------
        array_like
            GAE-style expected discounted returns.
        """
        n_steps = mb_rewards.shape[0]

        # Compute the last estimated value.
        last_values = self.sess.run(self.value_flat, {self.obs_ph: obs})

        # Discount/bootstrap off value fn.
        mb_advs = np.zeros_like(mb_rewards)
        mb_vactual = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                nextnonterminal = 1.0 - mb_dones[-1]
                nextvalues = last_values
                mb_vactual[t] = mb_rewards[t]
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
                mb_vactual[t] = mb_rewards[t] \
                    + self.gamma * nextnonterminal * nextvalues
            delta = mb_rewards[t] \
                + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta \
                + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return mb_returns
