"""TD3-compatible goal-conditioned hierarchical policy."""
import tensorflow as tf
import numpy as np

from hbaselines.fcnet.base import ActorCriticPolicy
from hbaselines.fcnet.td3 import FeedForwardPolicy
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.goal_conditioned.replay_buffer import HierReplayBuffer
from hbaselines.utils.reward_fns import negative_distance
from hbaselines.utils.misc import get_manager_ac_space, get_state_indices


class GoalConditionedPolicy(ActorCriticPolicy):
    r"""Goal-conditioned hierarchical reinforcement learning model.

    This policy is an implementation of the two-level hierarchy presented
    in [1], which itself is similar to the feudal networks formulation [2, 3].
    This network consists of a high-level, or Manager, pi_{\theta_H} that
    computes and outputs goals g_t ~ pi_{\theta_H}(s_t, h) every `meta_period`
    time steps, and a low-level policy pi_{\theta_L} that takes as inputs the
    current state and the assigned goals and attempts to perform an action
    a_t ~ pi_{\theta_L}(s_t,g_t) that satisfies these goals.

    The Manager is rewarded based on the original environment reward function:
    r_H = r(s,a;h).

    The Target term, h, parameterizes the reward assigned to the Manager in
    order to allow the policy to generalize to several goals within a task, a
    technique that was first proposed by [4].

    Finally, the Worker is motivated to follow the goals set by the Manager via
    an intrinsic reward based on the distance between the current observation
    and the goal observation:
    r_L (s_t, g_t, s_{t+1}) = -||s_t + g_t - s_{t+1}||_2

    Bibliography:

    [1] Nachum, Ofir, et al. "Data-efficient hierarchical reinforcement
        learning." Advances in Neural Information Processing Systems. 2018.
    [2] Dayan, Peter, and Geoffrey E. Hinton. "Feudal reinforcement learning."
        Advances in neural information processing systems. 1993.
    [3] Vezhnevets, Alexander Sasha, et al. "Feudal networks for hierarchical
        reinforcement learning." Proceedings of the 34th International
        Conference on Machine Learning-Volume 70. JMLR. org, 2017.
    [4] Schaul, Tom, et al. "Universal value function approximators."
        International Conference on Machine Learning. 2015.

    Attributes
    ----------
    manager : hbaselines.fcnet.td3.FeedForwardPolicy
        the manager policy
    meta_period : int
        manger action period
    worker_reward_scale : float
        the value the intrinsic (Worker) reward should be scaled by
    relative_goals : bool
        specifies whether the goal issued by the Manager is meant to be a
        relative or absolute goal, i.e. specific state or change in state
    off_policy_corrections : bool
        whether to use off-policy corrections during the update procedure. See:
        https://arxiv.org/abs/1805.08296.
    use_fingerprints : bool
        specifies whether to add a time-dependent fingerprint to the
        observations
    fingerprint_range : (list of float, list of float)
        the low and high values for each fingerprint element, if they are being
        used
    fingerprint_dim : tuple of int
        the shape of the fingerprint elements, if they are being used
    centralized_value_functions : bool
        specifies whether to use centralized value functions for the Manager
        critic functions
    connected_gradients : bool
        whether to connect the graph between the manager and worker
    cg_weights : float
        weights for the gradients of the loss of the worker with respect to the
        parameters of the manager. Only used if `connected_gradients` is set to
        True.
    prev_meta_obs : array_like
        previous observation by the Manager
    meta_action : array_like
        current action by the Manager
    meta_reward : float
        current meta reward, counting as the cumulative environment reward
        during the meta period
    batch_size : int
        SGD batch size
    worker : hbaselines.fcnet.td3.FeedForwardPolicy
        the worker policy
    worker_reward_fn : function
        reward function for the worker
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
                 meta_period,
                 worker_reward_scale,
                 relative_goals,
                 off_policy_corrections,
                 use_fingerprints,
                 fingerprint_range,
                 centralized_value_functions,
                 connected_gradients,
                 cg_weights,
                 env_name=""):
        """Instantiate the goal-conditioned hierarchical policy.

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
        meta_period : int
            manger action period
        worker_reward_scale : float
            the value the intrinsic (Worker) reward should be scaled by
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
        centralized_value_functions : bool
            specifies whether to use centralized value functions for the
            Manager and Worker critic functions
        connected_gradients : bool
            whether to connect the graph between the manager and worker
        cg_weights : float
            weights for the gradients of the loss of the worker with respect to
            the parameters of the manager. Only used if `connected_gradients`
            is set to True.
        """
        super(GoalConditionedPolicy, self).__init__(
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
            use_huber=use_huber
        )

        self.meta_period = meta_period
        self.worker_reward_scale = worker_reward_scale
        self.relative_goals = relative_goals
        self.off_policy_corrections = off_policy_corrections
        self.use_fingerprints = use_fingerprints
        self.fingerprint_range = fingerprint_range
        self.fingerprint_dim = (len(self.fingerprint_range[0]),)
        self.centralized_value_functions = centralized_value_functions
        self.connected_gradients = connected_gradients
        self.cg_weights = cg_weights

        # Get the Manager's action space.
        manager_ac_space = get_manager_ac_space(
            ob_space, relative_goals, env_name,
            use_fingerprints, self.fingerprint_dim)

        # Manager observation size
        meta_ob_dim = self._get_ob_dim(ob_space, co_space)

        # Create the replay buffer.
        self.replay_buffer = HierReplayBuffer(
            buffer_size=int(buffer_size/meta_period),
            batch_size=batch_size,
            meta_obs_dim=meta_ob_dim[0],
            meta_ac_dim=manager_ac_space.shape[0],
            worker_obs_dim=ob_space.shape[0] + manager_ac_space.shape[0],
            worker_ac_dim=ac_space.shape[0],
        )

        # =================================================================== #
        # Part 1. Setup the Manager                                           #
        # =================================================================== #

        # Create the Manager policy.
        with tf.compat.v1.variable_scope("Manager"):
            self.manager = FeedForwardPolicy(
                sess=sess,
                ob_space=ob_space,
                ac_space=manager_ac_space,
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
                scope="Manager",
                noise=noise,
                target_policy_noise=target_policy_noise,
                target_noise_clip=target_noise_clip,
                zero_fingerprint=False,
                fingerprint_dim=self.fingerprint_dim[0],
            )

        # a fixed goal transition function for the meta-actions in between meta
        # periods. This is used when relative_goals is set to True in order to
        # maintain a fixed absolute position of the goal.
        if relative_goals:
            def goal_transition_fn(obs0, goal, obs1):
                return obs0 + goal - obs1
        else:
            def goal_transition_fn(obs0, goal, obs1):
                return goal
        self.goal_transition_fn = goal_transition_fn

        # previous observation by the Manager
        self.prev_meta_obs = None

        # current action by the Manager
        self.meta_action = None

        # current meta reward, counting as the cumulative environment reward
        # during the meta period
        self.meta_reward = None

        # The following is redundant but necessary if the changes to the update
        # function are to be in the GoalConditionedPolicy policy and not
        # FeedForwardPolicy.
        self.batch_size = batch_size

        # Use this to store a list of observations that stretch as long as the
        # dilated horizon chosen for the Manager. These observations correspond
        # to the s(t) in the HIRO paper.
        self._observations = []

        # Use this to store the list of environmental actions that the worker
        # takes. These actions correspond to the a(t) in the HIRO paper.
        self._worker_actions = []

        # rewards provided by the policy to the worker
        self._worker_rewards = []

        # done masks at every time step for the worker
        self._dones = []

        # actions performed by the manager during a given meta period. Used by
        # the replay buffer.
        self._meta_actions = []

        # =================================================================== #
        # Part 2. Setup the Worker                                            #
        # =================================================================== #

        # Create the Worker policy.
        with tf.compat.v1.variable_scope("Worker"):
            self.worker = FeedForwardPolicy(
                sess,
                ob_space=ob_space,
                ac_space=ac_space,
                co_space=manager_ac_space,
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
                scope="Worker",
                noise=noise,
                target_policy_noise=target_policy_noise,
                target_noise_clip=target_noise_clip,
                zero_fingerprint=self.use_fingerprints,
                fingerprint_dim=self.fingerprint_dim[0],
            )

        # Collect the state indices for the worker rewards.
        state_indices = get_state_indices(
            ob_space, env_name, use_fingerprints, self.fingerprint_dim)

        # reward function for the worker
        def worker_reward_fn(states, goals, next_states):
            return negative_distance(
                states=states,
                state_indices=state_indices,
                goals=goals,
                next_states=next_states,
                relative_context=relative_goals,
                offset=0.0
            )
        self.worker_reward_fn = worker_reward_fn

        if self.connected_gradients:
            self._setup_connected_gradients()

    def initialize(self):
        """See parent class.

        This method calls the initialization methods of the manager and worker.
        """
        self.manager.initialize()
        self.worker.initialize()
        self.meta_reward = 0

    def update(self, update_actor=True, **kwargs):
        """Perform a gradient update step.

        This is done both at the level of the Manager and Worker policies.

        The kwargs argument for this method contains two additional terms:

        * update_meta (bool): specifies whether to perform a gradient update
          step for the meta-policy (i.e. Manager)
        * update_meta_actor (bool): similar to the `update_policy` term, but
          for the meta-policy. Note that, if `update_meta` is set to False,
          this term is void.

        **Note**; The target update soft updates for both the manager and the
        worker policies occur at the same frequency as their respective actor
        update frequencies.

        Parameters
        ----------
        update_actor : bool
            specifies whether to update the actor policy. The critic policy is
            still updated if this value is set to False.

        Returns
        -------
        (float, float)
            manager critic loss, worker critic loss
        (float, float)
            manager actor loss, worker actor loss
        """
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample():
            return (0, 0), (0, 0)

        # Get a batch.
        meta_obs0, meta_obs1, meta_act, meta_rew, meta_done, worker_obs0, \
            worker_obs1, worker_act, worker_rew, worker_done = \
            self.replay_buffer.sample()

        # Update the Manager policy.
        if kwargs['update_meta']:
            if self.connected_gradients:
                # Perform the connected gradients update procedure.
                m_critic_loss, m_actor_loss = self._connected_gradients_update(
                    obs0=meta_obs0,
                    actions=meta_act,
                    rewards=meta_rew,
                    obs1=meta_obs1,
                    terminals1=meta_done,
                    update_actor=kwargs['update_meta_actor'],
                    worker_obs0=worker_obs0,
                    worker_obs1=worker_obs1,
                    worker_actions=worker_act,
                )
            else:
                m_critic_loss, m_actor_loss = self.manager.update_from_batch(
                    obs0=meta_obs0,
                    actions=meta_act,
                    rewards=meta_rew,
                    obs1=meta_obs1,
                    terminals1=meta_done,
                    update_actor=kwargs['update_meta_actor'],
                )
        else:
            m_critic_loss, m_actor_loss = 0, 0

        # Update the Worker policy.
        w_critic_loss, w_actor_loss = self.worker.update_from_batch(
            obs0=worker_obs0,
            actions=worker_act,
            rewards=worker_rew,
            obs1=worker_obs1,
            terminals1=worker_done,
            update_actor=update_actor,
        )

        return (m_critic_loss, w_critic_loss), (m_actor_loss, w_actor_loss)

    def get_action(self, obs, context, apply_noise, random_actions):
        """See parent class."""
        if self._update_meta:
            # Update the meta action based on the output from the policy if the
            # time period requires is.
            self.meta_action = self.manager.get_action(
                obs, context, apply_noise, random_actions)
        else:
            # Update the meta-action in accordance with the fixed transition
            # function.
            goal_dim = self.meta_action.shape[1]
            self.meta_action = self.goal_transition_fn(
                obs0=np.asarray([self._observations[-1][:goal_dim]]),
                goal=self.meta_action,
                obs1=obs[:, :goal_dim]
            )

        # Return the worker action.
        worker_action = self.worker.get_action(
            obs, self.meta_action, apply_noise, random_actions)

        return worker_action

    def value(self, obs, context, action):
        """See parent class."""
        return 0, 0  # FIXME

    def store_transition(self, obs0, context0, action, reward, obs1, context1,
                         done, is_final_step, evaluate=False):
        """See parent class."""
        # Compute the worker reward and append it to the list of rewards.
        self._worker_rewards.append(
            self.worker_reward_scale *
            self.worker_reward_fn(obs0, self.meta_action.flatten(), obs1)
        )

        # Add the environmental observations and done masks, and the manager
        # and worker actions to their respective lists.
        self._worker_actions.append(action)
        self._meta_actions.append(self.meta_action.flatten())
        self._observations.append(self._get_obs(obs0, self.meta_action, 0))

        # Modify the done mask in accordance with the TD3 algorithm. Done
        # masks that correspond to the final step are set to False.
        self._dones.append(done and not is_final_step)

        # Increment the meta reward with the most recent reward.
        self.meta_reward += reward

        # Modify the previous meta observation whenever the action has changed.
        if len(self._observations) == 1:
            self.prev_meta_obs = self._get_obs(obs0, context0, 0)

        # Add a sample to the replay buffer.
        if len(self._observations) == self.meta_period or done:
            # Add the last observation.
            self._observations.append(self._get_obs(obs1, self.meta_action, 0))

            # Add the contextual observation to the most recent environmental
            # observation, if applicable.
            meta_obs1 = self._get_obs(obs1, context1, 0)

            # Store a sample in the Manager policy.
            if not evaluate:
                self.replay_buffer.add(
                    obs_t=self._observations,
                    goal_t=self._meta_actions[0],
                    action_t=self._worker_actions,
                    reward_t=self._worker_rewards,
                    done=self._dones,
                    meta_obs_t=(self.prev_meta_obs, meta_obs1),
                    meta_reward_t=self.meta_reward,
                )

            # Clear the worker rewards and actions, and the environmental
            # observation and reward.
            self.clear_memory()

    @property
    def _update_meta(self):
        """Return True if the meta-action should be updated by the policy.

        This is done by checking the length of the observation lists that are
        passed to the replay buffer, which are cleared whenever the meta-period
        has been met or the environment has been reset.
        """
        return len(self._observations) == 0

    def clear_memory(self):
        """Clear internal memory that is used by the replay buffer.

        By clearing memory, the Manager policy is then informed during the
        `get_action` procedure to update the meta-action.
        """
        self.meta_reward = 0
        self._observations = []
        self._worker_actions = []
        self._worker_rewards = []
        self._dones = []
        self._meta_actions = []

    def _sample_best_meta_action(self,
                                 state_reps,
                                 next_state_reprs,
                                 prev_meta_actions,
                                 low_states,
                                 low_actions,
                                 low_state_reprs,
                                 k=8):
        """Return meta-actions that approximately maximize low-level log-probs.

        Parameters
        ----------
        state_reps : array_like
            current Manager state observation
        next_state_reprs : array_like
            next Manager state observation
        prev_meta_actions : array_like
            previous meta Manager action
        low_states : array_like
            current Worker state observation
        low_actions : array_like
            current Worker environmental action
        low_state_reprs : array_like
            current Worker state observation
        k : int, optional
            number of goals returned, excluding the initial goal and the mean
            value

        Returns
        -------
        array_like
            most likely meta-actions
        """
        # Collect several samples of potentially optimal goals.
        sampled_actions = self._sample(
            state_reps, next_state_reprs, k, prev_meta_actions)

        sampled_log_probs = tf.reshape(self._log_probs(
            tf.tile(low_states, [k, 1, 1]),
            tf.tile(low_actions, [k, 1, 1]),
            tf.tile(low_state_reprs, [k, 1, 1]),
            [tf.reshape(sampled_actions, [-1, sampled_actions.shape[-1]])]),
            [k, low_states.shape[0], low_states.shape[1], -1])

        fitness = tf.reduce_sum(sampled_log_probs, [2, 3])
        best_actions = tf.argmax(fitness, 0)
        best_goals = tf.gather_nd(
            sampled_actions,
            tf.stack([
                best_actions,
                tf.range(prev_meta_actions.shape[0], dtype=tf.int64)], -1))

        return best_goals

    def _log_probs(self, manager_obs, worker_obs, actions, goals):
        """Calculate the log probability of the next goal by the Manager.

        Parameters
        ----------
        manager_obs : array_like
            (batch_size, m_obs_dim) matrix of manager observations
        worker_obs : array_like
            (batch_size, w_obs_dim, meta_period) matrix of worker observations
        actions : array_like
            (batch_size, ac_dim, meta_period-1) list of low-level actions
        goals : array_like
            (batch_size, goal_dim, num_samples) matrix of sampled goals

        Returns
        -------
        array_like
            (batch_size, num_samples) error associated with every state /
            action / goal pair

        Helps
        -----
        * _sample_best_meta_action(self):
        """
        # Action a policy would perform given a specific observation / goal.
        pred_actions = self.worker.get_action(
            worker_obs, goals,
            apply_noise=False,
            random_actions=False)

        # Normalize the error based on the range of applicable goals.
        goal_space = self.manager.ac_space
        spec_range = goal_space.high - goal_space.low
        scale = np.tile(np.square(spec_range), (manager_obs.shape[0], 1))

        # Compute error as the distance between expected and actual actions.
        normalized_error = np.mean(
            np.square(np.divide(actions - pred_actions, scale)), axis=1)

        return -normalized_error

    def _sample(self, states, next_states, num_samples, orig_goals, sc=0.5):
        """Sample different goals.

        These goals are acquired from a random Gaussian distribution centered
        at s_{t+c} - s_t.

        Parameters
        ----------
        states : array_like
            (batch_size, obs_dim) matrix of current time step observation
        next_states : array_like
            (batch_size, obs_dim) matrix of next time step observation
        num_samples : int
            number of samples
        orig_goals : array_like
            (batch_size, goal_dim) matrix of original goal specified by Manager
        sc : float
            scaling factor for the normal distribution.

        Returns
        -------
        array_like
            (batch_size, goal_dim, num_samples) matrix of sampled goals

        Helps
        -----
        * _sample_best_meta_action(self)
        """
        batch_size, goal_dim = orig_goals.shape
        goal_space = self.manager.ac_space
        spec_range = goal_space.high - goal_space.low

        # Compute the mean and std for the Gaussian distribution to sample from
        loc = np.tile((next_states - states)[:, :goal_dim].flatten(),
                      (num_samples-2, 1))
        scale = np.tile(sc * spec_range / 2, (num_samples-2, batch_size))

        # Sample the requested number of goals from the Gaussian distribution.
        samples = loc + np.random.normal(
            size=(num_samples - 2, goal_dim * batch_size)) * scale

        # Add the original goal and the average of the original and final state
        # to the sampled goals.
        samples = np.vstack(
            [samples,
             (next_states - states)[:, :goal_dim].flatten(),
             orig_goals.flatten()],
        )

        # Clip the values based on the Manager action space range.
        minimum = np.tile(goal_space.low, (num_samples, batch_size))
        maximum = np.tile(goal_space.high, (num_samples, batch_size))
        samples = np.minimum(np.maximum(samples, minimum), maximum)

        # Reshape to (batch_size, goal_dim, num_samples).
        samples = samples.T.reshape((batch_size, goal_dim, num_samples))

        return samples

    def get_td_map(self):
        """See parent class."""
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample():
            return {}

        # Get a batch.
        meta_obs0, meta_obs1, meta_act, meta_rew, meta_done, worker_obs0, \
            worker_obs1, worker_act, worker_rew, worker_done = \
            self.replay_buffer.sample()

        td_map = {}
        td_map.update(self.manager.get_td_map_from_batch(
            meta_obs0, meta_act, meta_rew, meta_obs1, meta_done))
        td_map.update(self.worker.get_td_map_from_batch(
            worker_obs0, worker_act, worker_rew, worker_obs1, worker_done))

        return td_map

    def _setup_connected_gradients(self):
        """Create the updated manager optimization with connected gradients."""
        goal_dim = self.manager.ac_space.shape[0]

        if self.relative_goals:
            # The observation from the perspective of the manager can be
            # collected from the first goal_dim elements of the observation. We
            # use goal_dim in case the goal-specific observations are not the
            # entire observation space.
            obs_t = self.manager.obs_ph[:, :goal_dim]
            # We collect the observation of the worker in a similar fashion as
            # above.
            obs_tpi = self.worker.obs_ph[:, :goal_dim]
            # Relative goal formulation as per HIRO.
            goal = obs_t + self.manager.actor_tf - obs_tpi
        else:
            # Goal is the direct output from the manager in this case.
            goal = self.manager.actor_tf

        # concatenate the output from the manager with the worker policy.
        obs_shape = self.worker.ob_space.shape[0]
        obs = tf.concat([self.worker.obs_ph[:, :obs_shape], goal], axis=-1)

        # create the worker policy with inputs directly from the manager
        with tf.compat.v1.variable_scope("Worker/model"):
            worker_with_manager_obs = self.worker.make_critic(
                obs, self.worker.action_ph, reuse=True, scope="qf_0")

        # create a tensorflow operation that mimics the reward function that is
        # used to provide feedback to the worker
        if self.relative_goals:
            reward_fn = -tf.compat.v1.losses.mean_squared_error(
                self.worker.obs_ph[:, :goal_dim] + goal,
                self.worker.obs1_ph[:, :goal_dim])
        else:
            reward_fn = -tf.compat.v1.losses.mean_squared_error(
                goal, self.worker.obs1_ph[:, :goal_dim])

        # compute the worker loss with respect to the manager actions
        self.cg_loss = - tf.reduce_mean(worker_with_manager_obs) - reward_fn

        # create the optimizer object
        optimizer = tf.compat.v1.train.AdamOptimizer(self.manager.actor_lr)
        self.cg_optimizer = optimizer.minimize(
            self.manager.actor_loss + self.cg_weights * self.cg_loss,
            var_list=get_trainable_vars("Manager/model/pi/"),
        )

    def _connected_gradients_update(self,
                                    obs0,
                                    actions,
                                    rewards,
                                    obs1,
                                    terminals1,
                                    worker_obs0,
                                    worker_obs1,
                                    worker_actions,
                                    update_actor=True):
        """Perform the gradient update procedure for the HRL-CG algorithm.

        This procedure is similar to self.manager.update_from_batch, expect it
        runs the self.cg_optimizer operation instead of self.manager.optimizer,
        and utilizes some information from the worker samples as well.

        Parameters
        ----------
        obs0 : np.ndarray
            batch of manager observations
        actions : numpy float
            batch of manager actions executed given obs_batch
        rewards : numpy float
            manager rewards received as results of executing act_batch
        obs1 : np.ndarray
            set of next manager observations seen after executing act_batch
        terminals1 : numpy bool
            done_mask[i] = 1 if executing act_batch[i] resulted in the end of
            an episode and 0 otherwise.
        worker_obs0 : array_like
            batch of worker observations
        worker_obs1 : array_like
            batch of next worker observations
        worker_actions : array_like
            batch of worker actions
        update_actor : bool
            specifies whether to update the actor policy of the manager. The
            critic policy is still updated if this value is set to False.

        Returns
        -------
        float
            manager critic loss
        float
            manager actor loss
        """
        # Reshape to match previous behavior and placeholder shape.
        rewards = rewards.reshape(-1, 1)
        terminals1 = terminals1.reshape(-1, 1)

        # Update operations for the critic networks.
        step_ops = [self.manager.critic_loss,
                    self.manager.critic_optimizer[0],
                    self.manager.critic_optimizer[1]]

        feed_dict = {
            self.manager.obs_ph: obs0,
            self.manager.action_ph: actions,
            self.manager.rew_ph: rewards,
            self.manager.obs1_ph: obs1,
            self.manager.terminals1: terminals1
        }

        if update_actor:
            # Actor updates and target soft update operation.
            step_ops += [self.manager.actor_loss,
                         self.cg_optimizer,  # This is what's replaced.
                         self.manager.target_soft_updates]

            feed_dict.update({
                self.worker.obs_ph: worker_obs0,
                self.worker.action_ph: worker_actions,
                self.worker.obs1_ph: worker_obs1,
            })

        # Perform the update operations and collect the critic loss.
        critic_loss, *_vals = self.sess.run(step_ops, feed_dict=feed_dict)

        # Extract the actor loss.
        actor_loss = _vals[2] if update_actor else 0

        return critic_loss, actor_loss
