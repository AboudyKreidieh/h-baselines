"""TRPO-compatible feedforward policy."""
import tensorflow as tf
import numpy as np

from hbaselines.base_policies import Policy
from hbaselines.utils.tf_util import create_fcnet
from hbaselines.utils.tf_util import create_conv
from hbaselines.utils.tf_util import print_params_shape
from hbaselines.utils.tf_util import process_minibatch
from hbaselines.utils.tf_util import get_globals_vars
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.utils.tf_util import flatgrad
from hbaselines.utils.tf_util import SetFromFlat
from hbaselines.utils.tf_util import GetFlat
from hbaselines.utils.tf_util import explained_variance


class FeedForwardPolicy(Policy):
    """TRPO-compatible feedforward policy..

    Attributes
    ----------
    gamma : float
        the discount factor
    lam : float
        factor for trade-off of bias vs variance for Generalized Advantage
        Estimator
    ent_coef : float
        entropy coefficient for the loss calculation
    cg_iters : int
        the number of iterations for the conjugate gradient calculation
    vf_iters : int
        the value function’s number iterations for learning
    vf_stepsize : float
        the value function stepsize
    cg_damping : float
        the compute gradient dampening factor
    max_kl : float
        the Kullback-Leibler loss threshold
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
    action_ph : tf.compat.v1.placeholder
        placeholder for the actions
    obs_ph : tf.compat.v1.placeholder
        placeholder for the observations
    ret_ph : tf.compat.v1.placeholder
        placeholder for the discounted returns
    advs_ph : tf.compat.v1.placeholder
        placeholder for the advantages
    old_vpred_ph : tf.compat.v1.placeholder
        placeholder for the predicted value
    flat_tangent : tf.compat.v1.placeholder
        placeholder for the tangents
    phase_ph : tf.compat.v1.placeholder
        a placeholder that defines whether training is occurring for the batch
        normalization layer. Set to True in training and False in testing.
    rate_ph : tf.compat.v1.placeholder
        the probability that each element is dropped if dropout is implemented
    action : tf.Variable
        the output from the policy/actor
    pi_mean : tf.Variable
        the output from the policy's mean term
    pi_logstd : tf.Variable
        the output from the policy's log-std term
    value_fn : tf.Variable
        the output from the value function
    value_flat : tf.Variable
        the output from the flattened value function
    old_action : tf.Variable
        the output from the previous instantiation of the policy/actor
    pi_mean : tf.Variable
        the output from the previous instantiation of the policy's mean term
    pi_logstd : tf.Variable
        the output from the previous instantiation of the policy's log-std term
    old_value_fn : tf.Variable
        the output from the previous instantiation of the value function
    value_flat : tf.Variable
        the output from the previous instantiation of the flattened value
        function
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 co_space,
                 verbose,
                 l2_penalty,
                 model_params,
                 gamma,
                 lam,
                 ent_coef,
                 cg_iters,
                 vf_iters,
                 vf_stepsize,
                 cg_damping,
                 max_kl,
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
        l2_penalty : float
            L2 regularization penalty. This is applied to the policy network.
        model_params : dict
            dictionary of model-specific parameters. See parent class.
        gamma : float
            the discount factor
        lam : float
            factor for trade-off of bias vs variance for Generalized Advantage
            Estimator
        ent_coef : float
            entropy coefficient for the loss calculation
        cg_iters : int
            the number of iterations for the conjugate gradient calculation
        vf_iters : int
            the value function’s number iterations for learning
        vf_stepsize : float
            the value function stepsize
        cg_damping : float
            the compute gradient dampening factor
        max_kl : float
            the Kullback-Leibler loss threshold
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

        self.gamma = gamma
        self.lam = lam
        self.ent_coef = ent_coef
        self.cg_iters = cg_iters
        self.vf_iters = vf_iters
        self.vf_stepsize = vf_stepsize
        self.cg_damping = cg_damping
        self.max_kl = max_kl

        # Create variables to store on-policy data.
        self.mb_rewards = [[] for _ in range(num_envs)]
        self.mb_obs = [[] for _ in range(num_envs)]
        self.mb_contexts = [[] for _ in range(num_envs)]
        self.mb_actions = [[] for _ in range(num_envs)]
        self.mb_values = [[] for _ in range(num_envs)]
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
            self.action_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ac_space.shape,
                name='actions')
            self.obs_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ob_dim,
                name='obs0')
            self.ret_ph = tf.placeholder(
                dtype=tf.float32,
                shape=(None,),
                name="ret_ph")
            self.advs_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,),
                name="advs_ph")
            self.old_vpred_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,),
                name="old_vpred_ph")
            self.flat_tangent = tf.placeholder(
                dtype=tf.float32,
                shape=[None],
                name="flat_tan")
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

            # Create the value function.
            self.value_fn = self.make_critic(self.obs_ph, scope="vf")
            self.value_flat = self.value_fn[:, 0]

        # Network for old policy
        with tf.variable_scope("oldpi/model", reuse=False):
            # Create the policy.
            self.old_action, self.old_pi_mean, self.old_pi_logstd = \
                self.make_actor(self.obs_ph, scope="pi")
            self.old_pi_std = tf.exp(self.old_pi_logstd)

            # Create the value function.
            self.old_value_fn = self.make_critic(self.obs_ph, scope="vf")
            self.old_value_flat = self.old_value_fn[:, 0]

        # =================================================================== #
        # Step 3: Setup the optimizers for the actor and critic.              #
        # =================================================================== #

        with tf.compat.v1.variable_scope("Optimizer", reuse=False):
            self._setup_optimizers(scope)

        # =================================================================== #
        # Step 4: Setup the operations for computing model statistics.        #
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
        old_scope_name = "oldpi/"
        if scope is not None:
            scope_name = scope + '/' + scope_name
            old_scope_name = scope + '/' + old_scope_name

        if self.verbose >= 2:
            print('setting up actor optimizer')
            print_params_shape("{}pi/".format(scope_name), "actor")
            print('setting up critic optimizer')
            print_params_shape("{}vf/".format(scope_name), "critic")

        # =================================================================== #
        # Create the policy loss and optimizers.                              #
        # =================================================================== #

        with tf.variable_scope("loss", reuse=False):
            # Compute the KL divergence.
            kloldnew = tf.reduce_sum(
                self.pi_logstd - self.old_pi_logstd + (
                    tf.square(self.old_pi_std) +
                    tf.square(self.old_pi_mean - self.pi_mean))
                / (2.0 * tf.square(self.pi_std)) - 0.5, axis=-1)
            meankl = tf.reduce_mean(kloldnew)

            # Compute the entropy bonus.
            entropy = tf.reduce_sum(
                self.pi_logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
            meanent = tf.reduce_mean(entropy)
            entbonus = self.ent_coef * meanent

            # advantage * pnew / pold
            ratio = tf.exp(
                self.logp(self.action_ph, old=False) -
                self.logp(self.action_ph, old=True))
            surrgain = tf.reduce_mean(ratio * self.advs_ph)

            optimgain = surrgain + entbonus
            self.losses = [optimgain, meankl, entbonus, surrgain, meanent]

            all_var_list = get_trainable_vars(scope_name)
            var_list = [
                v for v in all_var_list
                if "/vf" not in v.name and "/q/" not in v.name]
            vf_var_list = [
                v for v in all_var_list
                if "/pi" not in v.name and "/logstd" not in v.name]

            self.get_flat = GetFlat(var_list, sess=self.sess)
            self.set_from_flat = SetFromFlat(var_list, sess=self.sess)

            klgrads = tf.gradients(meankl, var_list)
            shapes = [var.get_shape().as_list() for var in var_list]
            start = 0
            tangents = []
            for shape in shapes:
                var_size = int(np.prod(shape))
                tangents.append(tf.reshape(
                    self.flat_tangent[start: start + var_size], shape))
                start += var_size
            gvp = tf.add_n(
                [tf.reduce_sum(grad * tangent)
                 for (grad, tangent) in zip(klgrads, tangents)])
            # Fisher vector products
            self.fvp = flatgrad(gvp, var_list)

        # =================================================================== #
        # Update the old model to match the new one.                          #
        # =================================================================== #

        self.assign_old_eq_new = tf.group(
            *[tf.assign(oldv, newv) for (oldv, newv) in
              zip(get_globals_vars(old_scope_name),
                  get_globals_vars(scope_name))])

        # =================================================================== #
        # Create the value function optimizer.                                #
        # =================================================================== #

        vferr = tf.reduce_mean(tf.square(
            self.value_flat - self.ret_ph))
        optimizer = tf.compat.v1.train.AdamOptimizer(self.vf_stepsize)
        self.vf_optimizer = optimizer.minimize(
            vferr,
            var_list=vf_var_list,
        )

        # Initialize the model parameters and optimizers.
        with self.sess.as_default():
            self.sess.run(tf.compat.v1.global_variables_initializer())

        th_init = self.get_flat()
        self.set_from_flat(th_init)

        self.grad = flatgrad(optimgain, var_list)

    def _setup_stats(self, base):
        """Create the running means and std of the model inputs and outputs.

        This method also adds the same running means and stds as scalars to
        tensorboard for additional storage.
        """
        ops = {
            "reference_action_mean": tf.reduce_mean(self.pi_mean),
            "reference_action_std": tf.reduce_mean(self.pi_logstd),
            "discounted_returns": tf.reduce_mean(self.ret_ph),
            "advantage": tf.reduce_mean(self.advs_ph),
            "old_value_pred": tf.reduce_mean(self.old_vpred_ph),
            "optimgain": self.losses[0],
            "meankl": self.losses[1],
            "entloss": self.losses[2],
            "surrgain": self.losses[3],
            "entropy": self.losses[4],
            "explained_variance": explained_variance(
                self.old_vpred_ph, self.ret_ph)
        }

        # Add all names and ops to the tensorboard summary.
        for key in ops.keys():
            name = "{}/{}".format(base, key)
            op = ops[key]
            tf.compat.v1.summary.scalar(name, op)

    def get_action(self, obs, context, apply_noise, random_actions, env_num=0):
        """See parent class."""
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        action = self.sess.run(
            self.action if apply_noise else self.pi_mean,
            feed_dict={
                self.obs_ph: obs,
                self.phase_ph: 0,
                self.rate_ph: 0.0,
            }
        )

        return action

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
        self.mb_obs[env_num].append(obs0.reshape(1, -1))
        self.mb_contexts[env_num].append(context0)
        self.mb_actions[env_num].append(action.reshape(1, -1))
        self.mb_dones[env_num].append(done)

        # Store information on the values.
        values = self.sess.run(
            self.value_flat,
            feed_dict={
                self.obs_ph: obs0.reshape(1, -1),
                self.phase_ph: 0,
                self.rate_ph: 0.0,
            }
        )
        self.mb_values[env_num].append(values)

        # Update the last observation (to compute the last value for the GAE
        # expected returns).
        self.last_obs[env_num] = self._get_obs([obs1], context1)

    def update(self, **kwargs):
        """See parent class."""
        # In case not all environment numbers were used, reduce the shape of
        # the datasets.
        indices = [
            i for i in range(self.num_envs) if self.last_obs[i] is not None]
        num_envs = len(indices)

        self.mb_rewards = self._sample(self.mb_rewards, indices)
        self.mb_obs = self._sample(self.mb_obs, indices)
        self.mb_contexts = self._sample(self.mb_contexts, indices)
        self.mb_actions = self._sample(self.mb_actions, indices)
        self.mb_values = self._sample(self.mb_values, indices)
        self.mb_dones = self._sample(self.mb_dones, indices)
        self.mb_all_obs = self._sample(self.mb_all_obs, indices)
        self.mb_returns = self._sample(self.mb_returns, indices)
        self.last_obs = self._sample(self.last_obs, indices)

        # Compute the last estimated value.
        last_values = [
            self.sess.run(
                self.value_flat,
                feed_dict={
                    self.obs_ph: self.last_obs[env_num],
                    self.phase_ph: 0,
                    self.rate_ph: 0.0,
                })
            for env_num in range(num_envs)
        ]

        (self.mb_obs,
         self.mb_contexts,
         self.mb_actions,
         self.mb_values,
         _,
         self.mb_all_obs,
         self.mb_rewards,
         self.mb_returns,
         self.mb_dones,
         self.mb_advs, n_steps) = process_minibatch(
            mb_obs=self.mb_obs,
            mb_contexts=self.mb_contexts,
            mb_actions=self.mb_actions,
            mb_values=self.mb_values,
            mb_neglogpacs=None,
            mb_all_obs=self.mb_all_obs,
            mb_rewards=self.mb_rewards,
            mb_returns=self.mb_returns,
            mb_dones=self.mb_dones,
            last_values=last_values,
            gamma=self.gamma,
            lam=self.lam,
            num_envs=num_envs,
        )

        self.update_from_batch(
            obs=self.mb_obs,
            context=None if self.mb_contexts[0] is None else self.mb_contexts,
            returns=self.mb_returns,
            actions=self.mb_actions,
            advs=self.mb_advs,
        )

    def update_from_batch(self, obs, context, returns, actions, advs):
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
        advs : array_like
            a minibatch of estimated advantages
        """
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        def fisher_vector_product(vec):
            return self.sess.run(self.fvp, feed_dict={
                self.flat_tangent: vec,
                self.obs_ph: fvpargs[0],
                self.action_ph: fvpargs[1],
                self.advs_ph: fvpargs[2],
            }) + self.cg_damping * vec

        # standardized advantage function estimate
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Subsampling: see p40-42 of John Schulman thesis
        # http://joschu.net/docs/thesis.pdf
        args = obs, actions, advs
        fvpargs = [arr[::5] for arr in args]

        self.sess.run(self.assign_old_eq_new)

        # run loss backprop with summary, and save the metadata (memory,
        # compute time, ...)
        grad, *lossbefore = self.sess.run(
            [self.grad] + self.losses,
            feed_dict={
                self.obs_ph: obs,
                self.action_ph: actions,
                self.advs_ph: advs,
                self.ret_ph: returns,
            }
        )

        lossbefore = np.array(lossbefore)
        if np.allclose(grad, 0):
            print("Got zero gradient. not updating")
        else:
            stepdir = self.conjugate_gradient(
                fisher_vector_product,
                grad,
                cg_iters=self.cg_iters,
                verbose=self.verbose >= 1,
            )
            assert np.isfinite(stepdir).all()
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            # abs(shs) to avoid taking square root of negative values
            lagrange_multiplier = np.sqrt(abs(shs) / self.max_kl)
            fullstep = stepdir / lagrange_multiplier
            expectedimprove = grad.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = self.get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                self.set_from_flat(thnew)
                mean_losses = surr, kl_loss, *_ = self.sess.run(
                    self.losses,
                    feed_dict={
                        self.obs_ph: obs,
                        self.action_ph: actions,
                        self.advs_ph: advs,
                    }
                )
                improve = surr - surrbefore
                print("Expected: %.3f Actual: %.3f" % (
                    expectedimprove, improve))
                if not np.isfinite(mean_losses).all():
                    print("Got non-finite value of losses -- bad!")
                elif kl_loss > self.max_kl * 1.5:
                    print("violated KL constraint. shrinking step.")
                elif improve < 0:
                    print("surrogate didn't improve. shrinking step.")
                else:
                    print("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                print("couldn't compute a good step")
                self.set_from_flat(thbefore)

        for _ in range(self.vf_iters):
            for (mbob, mbret) in self.iterbatches(
                    (obs, returns),
                    include_final_partial_batch=False,
                    batch_size=128,
                    shuffle=True):
                self.sess.run(self.vf_optimizer, feed_dict={
                    self.obs_ph: mbob,
                    self.action_ph: actions,
                    self.ret_ph: mbret,
                })

    def get_td_map(self):
        """See parent class."""
        # Add the contextual observation, if applicable.
        context = None if self.mb_contexts[0] is None else self.mb_contexts
        obs = self._get_obs(self.mb_obs, context, axis=1)

        td_map = self.get_td_map_from_batch(
            obs=obs.copy(),
            mb_actions=self.mb_actions,
            mb_advs=self.mb_advs,
            mb_returns=self.mb_returns,
            mb_values=self.mb_values,
        )

        # Clear memory
        self.mb_rewards = [[] for _ in range(self.num_envs)]
        self.mb_obs = [[] for _ in range(self.num_envs)]
        self.mb_contexts = [[] for _ in range(self.num_envs)]
        self.mb_actions = [[] for _ in range(self.num_envs)]
        self.mb_values = [[] for _ in range(self.num_envs)]
        self.mb_dones = [[] for _ in range(self.num_envs)]
        self.mb_all_obs = [[] for _ in range(self.num_envs)]
        self.mb_returns = [[] for _ in range(self.num_envs)]
        self.last_obs = [None for _ in range(self.num_envs)]
        self.mb_advs = None

        return td_map

    def get_td_map_from_batch(self,
                              obs,
                              mb_actions,
                              mb_advs,
                              mb_returns,
                              mb_values):
        """Convert a batch to a td_map."""
        return {
            self.obs_ph: obs,
            self.action_ph: mb_actions,
            self.advs_ph: mb_advs,
            self.ret_ph: mb_returns,
            self.old_vpred_ph: mb_values,
            self.phase_ph: 0,
            self.rate_ph: 0.0,
        }

    def initialize(self):
        """See parent class."""
        pass

    def logp(self, x, old=False):
        """Return the logp of an action from the old or current policy."""
        if old:
            return - self._old_neglogp(x)
        else:
            return - self._neglogp(x)

    def _neglogp(self, x):
        """Return the negative-logp of the current policy."""
        return 0.5 * tf.reduce_sum(
            tf.square((x - self.pi_mean) / self.pi_std), axis=-1) + 0.5 * \
            np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
            + tf.reduce_sum(self.pi_logstd, axis=-1)

    def _old_neglogp(self, x):
        """Return the negative-logp of the previous policy."""
        return 0.5 * tf.reduce_sum(
            tf.square((x - self.old_pi_mean) / self.old_pi_std), axis=-1) \
            + 0.5 * np.log(2. * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
            + tf.reduce_sum(self.old_pi_logstd, axis=-1)

    @staticmethod
    def iterbatches(arrays,
                    *,
                    num_batches=None,
                    batch_size=None,
                    shuffle=True,
                    include_final_partial_batch=True):
        """Iterate over arrays in batches.

        Must provide either num_batches or batch_size, the other must be None.

        Parameters
        ----------
        arrays : tuple
            a tuple of arrays
        num_batches : int
            the number of batches, must be None is batch_size is defined
        batch_size : int
            the size of the batch, must be None is num_batches is defined
        shuffle : bool
            enable auto shuffle
        include_final_partial_batch : bool
            add the last batch if not the same size as the batch_size

        Returns
        -------
        tuples
            a tuple of a batch of the arrays
        """
        assert (num_batches is None) != (batch_size is None), \
            'Provide num_batches or batch_size, but not both'
        arrays = tuple(map(np.asarray, arrays))
        n_samples = arrays[0].shape[0]
        assert all(a.shape[0] == n_samples for a in arrays[1:])
        inds = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(inds)
        sections = np.arange(0, n_samples, batch_size)[1:] \
            if num_batches is None else num_batches
        for batch_inds in np.array_split(inds, sections):
            if include_final_partial_batch or len(batch_inds) == batch_size:
                yield tuple(a[batch_inds] for a in arrays)

    @staticmethod
    def conjugate_gradient(f_ax,
                           b_vec,
                           cg_iters=10,
                           verbose=False,
                           residual_tol=1e-10):
        """Calculate the conjugate gradient of Ax = b.

        Based on https://epubs.siam.org/doi/book/10.1137/1.9781611971446 Demmel
        p 312.

        Parameters
        ----------
        f_ax : function
            The function describing the Matrix A dot the vector x (x being the
            input parameter of the function)
        b_vec : array_like
            vector b, where Ax = b
        cg_iters : int
            the maximum number of iterations for converging
        verbose : bool
            print extra information
        residual_tol : float
            the break point if the residual is below this value

        Returns
        -------
        array_like
            vector x, where Ax = b
        """
        # the first basis vector
        first_basis_vect = b_vec.copy()
        # the residual
        residual = b_vec.copy()
        # vector x, where Ax = b
        x_var = np.zeros_like(b_vec)
        # L2 norm of the residual
        residual_dot_residual = residual.dot(residual)

        fmt_str = "%10i %10.3g %10.3g"
        title_str = "%10s %10s %10s"
        if verbose:
            print(title_str % ("iter", "residual norm", "soln norm"))

        for i in range(cg_iters):
            if verbose:
                print(fmt_str %
                      (i, residual_dot_residual, np.linalg.norm(x_var)))
            z_var = f_ax(first_basis_vect)
            v_var = residual_dot_residual / first_basis_vect.dot(z_var)
            x_var += v_var * first_basis_vect
            residual -= v_var * z_var
            new_residual_dot_residual = residual.dot(residual)
            mu_val = new_residual_dot_residual / residual_dot_residual
            first_basis_vect = residual + mu_val * first_basis_vect

            residual_dot_residual = new_residual_dot_residual
            if residual_dot_residual < residual_tol:
                break

        if verbose:
            print(fmt_str %
                  (cg_iters, residual_dot_residual, np.linalg.norm(x_var)))
        return x_var

    @staticmethod
    def _sample(vals, indices):
        """Sample indices from a list."""
        return [vals[i] for i in range(len(vals)) if i in indices]
