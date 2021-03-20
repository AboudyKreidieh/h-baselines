"""PPO-compatible multi-agent feedforward neural network policy."""
import tensorflow as tf
import numpy as np

from hbaselines.multiagent.base import MultiAgentPolicy as BasePolicy
from hbaselines.fcnet.ppo import FeedForwardPolicy
from hbaselines.utils.tf_util import create_conv
from hbaselines.utils.tf_util import create_fcnet
from hbaselines.utils.tf_util import neglogp
from hbaselines.utils.tf_util import process_minibatch
from hbaselines.utils.tf_util import print_params_shape
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.utils.tf_util import explained_variance


class MultiFeedForwardPolicy(BasePolicy):
    """PPO-compatible multi-agent feedforward neural network policy."""

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 co_space,
                 verbose,
                 l2_penalty,
                 model_params,
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
                 shared,
                 maddpg,
                 n_agents,
                 all_ob_space=None,
                 num_envs=1,
                 scope=None):
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
        model_params : dict
            dictionary of model-specific parameters. See parent class.
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
        """
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

        super(MultiFeedForwardPolicy, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            co_space=co_space,
            verbose=verbose,
            l2_penalty=l2_penalty,
            model_params=model_params,
            shared=shared,
            maddpg=maddpg,
            all_ob_space=all_ob_space,
            n_agents=n_agents,
            base_policy=FeedForwardPolicy,
            num_envs=num_envs,
            scope=scope,
            additional_params=dict(
                learning_rate=learning_rate,
                n_minibatches=n_minibatches,
                n_opt_epochs=n_opt_epochs,
                gamma=gamma,
                lam=lam,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                cliprange=cliprange,
                cliprange_vf=cliprange_vf,
            ),
        )

    def _setup_maddpg(self, scope):
        """See setup."""
        num_envs = self.num_envs * self.n_agents
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
        self.last_all_obs = [None for _ in range(num_envs)]
        self.mb_advs = None

        # Compute the shape of the input observation space, which may include
        # the contextual term.
        ob_dim = self._get_ob_dim(self.ob_space, self.co_space)

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
                shape=(None,) + self.ac_space.shape,
                name='actions')
            self.obs_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ob_dim,
                name='obs0')
            self.all_obs_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + tuple(
                    map(sum, zip(ob_dim, self.all_ob_space.shape))),
                name='all_obs0')
            self.advs_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,),
                name="advs_ph")
            self.old_neglog_pac_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,),
                name="old_neglog_pac_ph")
            self.old_vpred_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,),
                name="old_vpred_ph")
            self.phase_ph = tf.compat.v1.placeholder(
                tf.bool,
                name='phase')
            self.rate_ph = tf.compat.v1.placeholder(
                tf.float32,
                name='rate')

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
            self.neglogp = neglogp(
                x=self.action,
                pi_mean=self.pi_mean,
                pi_std=self.pi_std,
                pi_logstd=self.pi_logstd,
            )

            # Create the value function.
            self.value_fn = self.make_critic(self.all_obs_ph, scope="vf")
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

    def _initialize_maddpg(self):
        """See initialize."""
        pass

    def _update_maddpg(self, update_actor=True, **kwargs):
        """See update."""
        # Compute the last estimated value.
        last_values = [
            self.sess.run(
                self.value_flat,
                feed_dict={
                    self.all_obs_ph: np.concatenate(
                        (self.last_obs[env_num], self.last_all_obs[env_num]),
                        axis=1,
                    ),
                    self.phase_ph: 0,
                    self.rate_ph: 0.0,
                })
            for env_num in range(self.num_envs * self.n_agents)
        ]

        (self.mb_obs,
         self.mb_contexts,
         self.mb_actions,
         self.mb_values,
         self.mb_neglogpacs,
         self.mb_all_obs,
         self.mb_rewards,
         self.mb_returns,
         self.mb_dones,
         self.mb_advs, n_steps) = process_minibatch(
            mb_obs=self.mb_obs,
            mb_contexts=self.mb_contexts,
            mb_actions=self.mb_actions,
            mb_values=self.mb_values,
            mb_neglogpacs=self.mb_neglogpacs,
            mb_all_obs=self.mb_all_obs,
            mb_rewards=self.mb_rewards,
            mb_returns=self.mb_returns,
            mb_dones=self.mb_dones,
            last_values=last_values,
            gamma=self.gamma,
            lam=self.lam,
            num_envs=self.num_envs * self.n_agents,
        )

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
                    all_obs=self.mb_all_obs[mbinds],
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
                          all_obs,
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
        all_obs : array_like
            a minibatch of full-state observations
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

        # Combine the observations and full-state observations.
        all_obs = np.concatenate((obs, all_obs), axis=1)

        return self.sess.run(self.optimizer, {
            self.obs_ph: obs,
            self.all_obs_ph: all_obs,
            self.action_ph: actions,
            self.advs_ph: advs,
            self.rew_ph: returns,
            self.old_neglog_pac_ph: neglogpacs,
            self.old_vpred_ph: values,
            self.phase_ph: 1,
            self.rate_ph: 0.5,
        })

    def _get_action_maddpg(self,
                           obs,
                           context,
                           apply_noise,
                           random_actions,
                           env_num):
        """See get_action."""
        actions = {}

        # Update the index of agent observations. This helps support action
        # computations for agents with memory (e.g. goal-conditioned policies)
        # and variable agents (e.g. the highway and I-210  networks).
        if self.shared:
            self._update_agent_index(obs, env_num)

        for key in obs.keys():
            # Get the contextual term. This accounts for cases when the context
            # is set to None.
            context_i = context if context is None else context[key]

            # Add the contextual observation, if applicable.
            obs_i = self._get_obs(obs[key], context_i, axis=1)

            # Compute the action.
            actions[key] = self.sess.run(
                self.action if apply_noise else self.pi_mean,
                feed_dict={
                    self.obs_ph: obs_i,
                    self.phase_ph: 0,
                    self.rate_ph: 0.0,
                }
            )

        return actions

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
        for key in obs0.keys():
            # Use the same policy for all operations if shared, and the
            # corresponding policy otherwise.
            env_num_i = \
                self.n_agents * env_num + self._agent_index[env_num][key] \
                if self.shared else env_num

            # Get the contextual term. This accounts for cases when the context
            # is set to None.
            context0_i = context0 if context0 is None else context0[key]
            context1_i = context1 if context1 is None else context1[key]

            # Add the contextual observation, if applicable.
            obs_i = self._get_obs(obs0[key], context0_i, axis=1)

            # Combine the observations and full-state observations.
            all_obs_i = np.concatenate((obs_i, all_obs0), axis=1)

            # Store information on the values and negative-log likelihood.
            values, neglogpacs = self.sess.run(
                [self.value_flat, self.neglogp],
                feed_dict={
                    self.obs_ph: obs_i,
                    self.all_obs_ph: all_obs_i,
                    self.phase_ph: 0,
                    self.rate_ph: 0.0,
                }
            )
            self.mb_values[env_num_i].append(values)
            self.mb_neglogpacs[env_num_i].append(neglogpacs)

            # Update the minibatch of samples.
            self.mb_rewards[env_num_i].append(reward[key])
            self.mb_obs[env_num_i].append(obs0[key].reshape(1, -1))
            self.mb_all_obs[env_num_i].append(all_obs0)
            self.mb_contexts[env_num_i].append(context0_i)
            self.mb_actions[env_num_i].append(action[key].reshape(1, -1))
            self.mb_dones[env_num_i].append(done[key])

            # Update the last observation (to compute the last value for the
            # GAE expected returns).
            self.last_obs[env_num_i] = self._get_obs([obs1[key]], context1_i)
            self.last_all_obs[env_num_i] = all_obs1

    def _get_td_map_maddpg(self):
        """See get_td_map."""
        # Add the contextual observation, if applicable.
        context = None if self.mb_contexts[0] is None else self.mb_contexts
        obs = self._get_obs(self.mb_obs, context, axis=1)

        td_map = self.get_td_map_from_batch(
            obs=obs.copy(),
            all_obs=self.mb_all_obs,
            mb_actions=self.mb_actions,
            mb_advs=self.mb_advs,
            mb_returns=self.mb_returns,
            mb_neglogpacs=self.mb_neglogpacs,
            mb_values=self.mb_values,
        )

        # Clear memory
        num_envs = self.num_envs * self.n_agents
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

        return td_map

    def get_td_map_from_batch(self,
                              obs,
                              all_obs,
                              mb_actions,
                              mb_advs,
                              mb_returns,
                              mb_neglogpacs,
                              mb_values):
        """Convert a batch to a td_map."""
        # Combine the observations and full-state observations.
        all_obs = np.concatenate((obs, all_obs), axis=1)

        return {
            self.obs_ph: obs,
            self.all_obs_ph: all_obs,
            self.action_ph: mb_actions,
            self.advs_ph: mb_advs,
            self.rew_ph: mb_returns,
            self.old_neglog_pac_ph: mb_neglogpacs,
            self.old_vpred_ph: mb_values,
            self.phase_ph: 0,
            self.rate_ph: 0.0,
        }

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

        # Create the output mean.
        policy_mean = create_fcnet(
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
                batch_norm=self.model_params["batch_norm"],
                phase=self.phase_ph,
                dropout=self.model_params["dropout"],
                rate=self.rate_ph,
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
            batch_norm=self.model_params["batch_norm"],
            phase=self.phase_ph,
            dropout=self.model_params["dropout"],
            rate=self.rate_ph,
            scope=scope,
            reuse=reuse,
        )

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

        neglogpac = neglogp(
            x=self.action_ph,
            pi_mean=self.pi_mean,
            pi_std=self.pi_std,
            pi_logstd=self.pi_logstd,
        )

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
