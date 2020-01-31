"""Base goal-conditioned hierarchical policy."""
import tensorflow as tf
import numpy as np
from copy import deepcopy
import random

from hbaselines.fcnet.base import ActorCriticPolicy
from hbaselines.goal_conditioned.replay_buffer import HierReplayBuffer
from hbaselines.utils.reward_fns import negative_distance
from hbaselines.utils.misc import get_manager_ac_space, get_state_indices
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.utils.tf_util import gaussian_likelihood


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
    manager : hbaselines.fcnet.base.ActorCriticPolicy
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
    hindsight : bool
        whether to use hindsight action and goal transitions, as well as
        subgoal testing. See: https://arxiv.org/abs/1712.00948
    subgoal_testing_rate : float
        rate at which the original (non-hindsight) sample is stored in the
        replay buffer as well. Used only if `hindsight` is set to True.
    connected_gradients : bool
        whether to connect the graph between the manager and worker
    cg_weights : float
        weights for the gradients of the loss of the worker with respect to the
        parameters of the manager. Only used if `connected_gradients` is set to
        True.
    multistep_llp : bool
        whether to use the multi-step LLP update procedure. See: TODO
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
    prev_meta_obs : array_like
        previous observation by the Manager
    meta_action : array_like
        current action by the Manager
    meta_reward : float
        current meta reward, counting as the cumulative environment reward
        during the meta period
    batch_size : int
        SGD batch size
    worker : hbaselines.fcnet.base.ActorCriticPolicy
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
                 layer_norm,
                 layers,
                 act_fun,
                 use_huber,
                 meta_period,
                 worker_reward_scale,
                 relative_goals,
                 off_policy_corrections,
                 hindsight,
                 subgoal_testing_rate,
                 connected_gradients,
                 cg_weights,
                 multistep_llp,
                 use_fingerprints,
                 fingerprint_range,
                 centralized_value_functions,
                 env_name="",
                 meta_policy=None,
                 worker_policy=None,
                 additional_params=None):
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
        hindsight : bool
            whether to include hindsight action and goal transitions in the
            replay buffer. See: https://arxiv.org/abs/1712.00948
        subgoal_testing_rate : float
            rate at which the original (non-hindsight) sample is stored in the
            replay buffer as well. Used only if `hindsight` is set to True.
        connected_gradients : bool
            whether to connect the graph between the manager and worker
        cg_weights : float
            weights for the gradients of the loss of the worker with respect to
            the parameters of the manager. Only used if `connected_gradients`
            is set to True.
        multistep_llp : bool
            whether to use the multi-step LLP update procedure. See: TODO
        use_fingerprints : bool
            specifies whether to add a time-dependent fingerprint to the
            observations
        fingerprint_range : (list of float, list of float)
            the low and high values for each fingerprint element, if they are
            being used
        centralized_value_functions : bool
            specifies whether to use centralized value functions for the
            Manager and Worker critic functions
        meta_policy : type [ hbaselines.fcnet.base.ActorCriticPolicy ]
            the policy model to use for the Manager
        worker_policy : type [ hbaselines.fcnet.base.ActorCriticPolicy ]
            the policy model to use for the Worker
        additional_params : dict
            additional algorithm-specific policy parameters. Used internally by
            the class when instantiating other (child) policies.
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
        self.hindsight = hindsight
        self.subgoal_testing_rate = subgoal_testing_rate
        self.connected_gradients = connected_gradients
        self.cg_weights = cg_weights
        self.multistep_llp = multistep_llp
        self.use_fingerprints = use_fingerprints
        self.fingerprint_range = fingerprint_range
        self.fingerprint_dim = (len(self.fingerprint_range[0]),)
        self.centralized_value_functions = centralized_value_functions

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
            meta_period=meta_period,
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
            self.manager = meta_policy(
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
                zero_fingerprint=False,
                fingerprint_dim=self.fingerprint_dim[0],
                **(additional_params or {}),
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
            self.worker = worker_policy(
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
                zero_fingerprint=self.use_fingerprints,
                fingerprint_dim=self.fingerprint_dim[0],
                **(additional_params or {}),
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

        if self.multistep_llp:
            self._setup_multistep_llp()

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
         ([float, float], [float, float])
            manager critic loss, worker critic loss
        (float, float)
            manager actor loss, worker actor loss
        """
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample():
            return ([0, 0], [0, 0]), (0, 0)

        # Specifies whether to remove additional data from the replay buffer
        # sampling procedure. Since only a subset of algorithms use additional
        # data, removing it can speedup the other algorithms.
        with_additional = self.off_policy_corrections or self.multistep_llp

        # Get a batch.
        meta_obs0, meta_obs1, meta_act, meta_rew, meta_done, worker_obs0, \
            worker_obs1, worker_act, worker_rew, worker_done, additional = \
            self.replay_buffer.sample(with_additional=with_additional)

        if self.multistep_llp:
            # Perform model-based relabeling.
            w_obses, w_actions, w_rewards = self._multistep_llp_update(
                meta_action=meta_act,
                worker_obses=additional["worker_obses"],
                worker_actions=additional["worker_actions"]
            )

            # Update the samples from the batch.  TODO: meta_obs1?
            meta_obs1, worker_obs0, worker_obs1, worker_act, worker_rew = \
                self._sample_from_relabeled(
                    worker_obses=w_obses,
                    worker_actions=w_actions,
                    worker_rewards=w_rewards
                )
            additional["worker_obses"] = w_obses
            additional["worker_actions"] = w_actions

        # Update the Manager policy.
        if kwargs['update_meta']:
            # Replace the goals with the most likely goals.
            if self.off_policy_corrections:
                meta_act = self._sample_best_meta_action(
                    meta_obs0=meta_obs0,
                    meta_obs1=meta_obs1,
                    meta_action=meta_act,
                    worker_obses=additional["worker_obses"],
                    worker_actions=additional["worker_actions"],
                    k=8
                )

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
                # Perform the regular manager update procedure.
                m_critic_loss, m_actor_loss = self.manager.update_from_batch(
                    obs0=meta_obs0,
                    actions=meta_act,
                    rewards=meta_rew,
                    obs1=meta_obs1,
                    terminals1=meta_done,
                    update_actor=kwargs['update_meta_actor'],
                )
        else:
            m_critic_loss, m_actor_loss = [0, 0], 0

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

            # Avoid storing samples when performing evaluations.
            if not evaluate:
                if not self.hindsight \
                        or random.random() < self.subgoal_testing_rate:
                    # Store a sample in the replay buffer.
                    self.replay_buffer.add(
                        obs_t=self._observations,
                        goal_t=self._meta_actions[0],
                        action_t=self._worker_actions,
                        reward_t=self._worker_rewards,
                        done=self._dones,
                        meta_obs_t=(self.prev_meta_obs, meta_obs1),
                        meta_reward_t=self.meta_reward,
                    )

                if self.hindsight:
                    # Implement hindsight action and goal transitions.
                    goal, obs, rewards = self._hindsight_actions_goals(
                        meta_action=self.meta_action,
                        initial_observations=self._observations,
                        initial_rewards=self._worker_rewards
                    )

                    # Store the hindsight sample in the replay buffer.
                    self.replay_buffer.add(
                        obs_t=obs,
                        goal_t=goal,
                        action_t=self._worker_actions,
                        reward_t=rewards,
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

    def get_td_map(self):
        """See parent class."""
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample():
            return {}

        # Get a batch.
        meta_obs0, meta_obs1, meta_act, meta_rew, meta_done, worker_obs0, \
            worker_obs1, worker_act, worker_rew, worker_done, _ = \
            self.replay_buffer.sample()

        td_map = {}
        td_map.update(self.manager.get_td_map_from_batch(
            meta_obs0, meta_act, meta_rew, meta_obs1, meta_done))
        td_map.update(self.worker.get_td_map_from_batch(
            worker_obs0, worker_act, worker_rew, worker_obs1, worker_done))

        return td_map

    # ======================================================================= #
    #                       Auxiliary methods for HIRO                        #
    # ======================================================================= #

    def _sample_best_meta_action(self,
                                 meta_obs0,
                                 meta_obs1,
                                 meta_action,
                                 worker_obses,
                                 worker_actions,
                                 k=10):
        """Return meta-actions that approximately maximize low-level log-probs.

        Parameters
        ----------
        meta_obs0 : array_like
            (batch_size, m_obs_dim) matrix of Manager observations
        meta_obs1 : array_like
            (batch_size, m_obs_dim) matrix of next time step Manager
            observations
        meta_action : array_like
            (batch_size, m_ac_dim) matrix of Manager actions
        worker_obses : array_like
            (batch_size, w_obs_dim, meta_period+1) matrix of current Worker
            state observations
        worker_actions : array_like
            (batch_size, w_ac_dim, meta_period) matrix of current Worker
            environmental actions
        k : int, optional
            number of goals returned, excluding the initial goal and the mean
            value

        Returns
        -------
        array_like
            (batch_size, m_ac_dim) matrix of most likely Manager actions
        """
        batch_size, goal_dim = meta_action.shape

        # Collect several samples of potentially optimal goals.
        sampled_actions = self._sample(meta_obs0, meta_obs1, meta_action, k)
        assert sampled_actions.shape == (batch_size, goal_dim, k)

        # Compute the fitness of each candidate goal. The fitness is the sum of
        # the log-probabilities of each action for the given goal.
        fitness = self._log_probs(
            sampled_actions, worker_obses, worker_actions)
        assert fitness.shape == (batch_size, k)

        # For each sample, choose the meta action that maximizes the fitness.
        indx = np.argmax(fitness, 1)
        best_goals = np.asarray(
            [sampled_actions[i, :, indx[i]] for i in range(batch_size)])

        return best_goals

    def _sample(self, meta_obs0, meta_obs1, meta_action, num_samples, sc=0.5):
        """Sample different goals.

        The goals are sampled as follows:

        * The first num_samples-2 goals are acquired from a random Gaussian
          distribution centered at s_{t+c} - s_t.
        * The second to last goal is s_{t+c} - s_t.
        * The last goal is the originally sampled goal g_t.

        Parameters
        ----------
        meta_obs0 : array_like
            (batch_size, m_obs_dim) matrix of Manager observations
        meta_obs1 : array_like
            (batch_size, m_obs_dim) matrix of next time step Manager
            observations
        meta_action : array_like
            (batch_size, m_ac_dim) matrix of Manager actions
        num_samples : int
            number of samples
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
        batch_size, goal_dim = meta_action.shape
        goal_space = self.manager.ac_space
        spec_range = goal_space.high - goal_space.low
        random_samples = num_samples - 2

        # Compute the mean and std for the Gaussian distribution to sample
        # from, and well as the maxima and minima.
        loc = meta_obs1[:, :goal_dim] - meta_obs0[:, :goal_dim]
        scale = [sc * spec_range / 2]
        minimum, maximum = [goal_space.low], [goal_space.high]

        new_loc = np.zeros((batch_size, goal_dim, random_samples))
        new_scale = np.zeros((batch_size, goal_dim, random_samples))
        for i in range(random_samples):
            new_loc[:, :, i] = loc
            new_scale[:, :, i] = scale

        new_minimum = np.zeros((batch_size, goal_dim, num_samples))
        new_maximum = np.zeros((batch_size, goal_dim, num_samples))
        for i in range(num_samples):
            new_minimum[:, :, i] = minimum
            new_maximum[:, :, i] = maximum

        # Generate random samples for the above distribution.
        normal_samples = np.random.normal(
            size=(random_samples * batch_size * goal_dim))
        normal_samples = normal_samples.reshape(
            (batch_size, goal_dim, random_samples))

        samples = np.zeros((batch_size, goal_dim, num_samples))
        samples[:, :, :-2] = new_loc + normal_samples * new_scale
        samples[:, :, -2] = loc
        samples[:, :, -1] = meta_action

        # Clip the values based on the Manager action space range.
        samples = np.minimum(np.maximum(samples, new_minimum), new_maximum)

        return samples

    def _log_probs(self, meta_actions, worker_obses, worker_actions):
        """Calculate the log probability of the next goal by the Manager.

        Parameters
        ----------
        meta_actions : array_like
            (batch_size, m_ac_dim, num_samples) matrix of candidate Manager
            actions
        worker_obses : array_like
            (batch_size, w_obs_dim, meta_period + 1) matrix of Worker
            observations
        worker_actions : array_like
            (batch_size, w_ac_dim, meta_period) list of Worker actions

        Returns
        -------
        array_like
            (batch_size, num_samples) fitness associated with every state /
            action / goal pair

        Helps
        -----
        * _sample_best_meta_action(self):
        """
        raise NotImplementedError

    # ======================================================================= #
    #                       Auxiliary methods for HAC                         #
    # ======================================================================= #

    def _hindsight_actions_goals(self,
                                 meta_action,
                                 initial_observations,
                                 initial_rewards):
        """Calculate hindsight goal and action transitions.

        These are then stored in the replay buffer along with the original
        (non-hindsight) sample.

        See the README at the front page of this repository for an in-depth
        description of this procedure.

        Parameters
        ----------
        meta_action : array_like
            the original Manager actions (goal)
        initial_observations : array_like
            the original worker observations with the non-hindsight goals
            appended to them
        initial_rewards : array_like
            the original worker rewards

        Returns
        -------
        array_like
            the Manager action (goal) in hindsight
        array_like
            the modified Worker observations with the hindsight goals appended
            to them
        array_like
            the modified Worker rewards taking into account the hindsight goals

        Helps
        -----
        * store_transition(self):
        """
        goal_dim = meta_action.shape[0]
        observations = deepcopy(initial_observations)
        rewards = deepcopy(initial_rewards)
        hindsight_goal = 0 if self.relative_goals \
            else observations[-1][:goal_dim]
        obs_tp1 = observations[-1]

        for i in range(1, len(observations) + 1):
            obs_t = observations[-i]

            # Calculate the hindsight goal in using relative goals.
            # If not, the hindsight goal is simply a subset of the
            # final state observation.
            if self.relative_goals:
                hindsight_goal += obs_tp1[:goal_dim] - obs_t[:goal_dim]

            # Modify the Worker intrinsic rewards based on the new
            # hindsight goal.
            if i > 1:
                rewards[-(i - 1)] = self.worker_reward_scale \
                    * self.worker_reward_fn(obs_t, hindsight_goal, obs_tp1)

            obs_tp1 = deepcopy(obs_t)

            # Replace the goal with the goal that the worker
            # actually achieved.
            observations[-i][-goal_dim:] = hindsight_goal

        return hindsight_goal, observations, rewards

    # ======================================================================= #
    #                      Auxiliary methods for HRL-CG                       #
    # ======================================================================= #

    def _setup_connected_gradients(self):
        """Create the updated manager optimization with connected gradients."""
        raise NotImplementedError

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
        [float, float]
            manager critic loss
        float
            manager actor loss
        """
        raise NotImplementedError

    # ======================================================================= #
    #                  Auxiliary methods for Multi-Step LLP                   #
    # ======================================================================= #

    def _setup_multistep_llp(self):
        """Create the trainable features of the multi-step LLP algorithm."""
        with tf.compat.v1.variable_scope("multistep_llp", reuse=False):
            # Create placeholders for the model.
            self.worker_obs_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + self.worker.ob_space.shape,
                name="worker_obs0")
            self.worker_obs1_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + self.worker.ob_space.shape,
                name="worker_obs1")
            self.worker_action_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + self.worker.ac_space.shape,
                name="worker_action")

            # Create a trainable model of the Worker dynamics.
            self.worker_model, logp = self._setup_worker_model(
                obs=self.worker_obs_ph,
                obs1=self.worker_obs1_ph,
                action=self.worker_action_ph,
                ob_space=self.ob_space,
            )

            # Create the model loss.
            self.worker_model_loss = -tf.reduce_mean(logp)

            # Create an optimizer object.
            optimizer = tf.train.AdamOptimizer(self.actor_lr)

            # Create the model optimization technique.
            self.worker_model_optimizer = optimizer.minimize(
                self.worker_model_loss,
                var_list=get_trainable_vars('multistep_llp'))

    def _setup_worker_model(self,
                            obs,
                            obs1,
                            action,
                            ob_space,
                            reuse=False,
                            scope="rho"):
        """Create the trainable parameters of the Worker dynamics model.

        Parameters
        ----------
        obs : tf.compat.v1.placeholder
            the last step observation, not including the context
        obs1 : tf.compat.v1.placeholder
            the current step observation, not including the context
        action : tf.compat.v1.placeholder
            the action from the Worker policy. May be a function of the
            Manager's trainable parameters
        ob_space : gym.spaces.*
            the observation space, not including the context space
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor

        Returns
        -------
        tf.Variable
            the output from the Worker dynamics model
        """
        if self.verbose >= 2:
            print('setting up Worker dynamics model')

        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Concatenate the observations and actions.
            rho_h = tf.concat([obs, action], axis=-1)

            # Create the hidden layers.
            for i, layer_size in enumerate(self.layers):
                rho_h = self._layer(
                    rho_h,  layer_size, 'fc{}'.format(i),
                    act_fun=self.act_fun,
                    layer_norm=self.layer_norm
                )

            # Create the output mean.
            rho_mean = self._layer(
                rho_h, ob_space.shape[0], 'rho_mean',
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3)
            )

            # Create the output logstd term.
            rho_logstd = self._layer(
                rho_h, ob_space.shape[0], 'rho_logstd',
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3)
            )
            rho_std = tf.exp(rho_logstd)

            # The model samples from its distribution.
            rho = rho_mean + tf.random.normal(tf.shape(rho_mean)) * rho_std

            # The worker model is trained to learn the change in state between
            # two time-steps.
            delta = obs1 - obs

            # Computes the log probability of choosing a specific output - used
            # by the loss
            rho_logp = gaussian_likelihood(delta, rho_mean, rho_logstd)

        return rho, rho_logp

    def _multistep_llp_update(self, meta_action, worker_obses, worker_actions):
        """Perform the multi-step LLP update procedure.

        The Worker states and actions, as well as the intrinsic rewards, are
        relabeled using a trained dynamics model. The last Worker observation
        is also used to relabel the Manager next step observation.

        Parameters
        ----------
        meta_action : array_like
            (batch_size, m_ac_dim) matrix of Manager actions (goals)
            actions
        worker_obses : array_like
            (batch_size, w_obs_dim, meta_period + 1) matrix of Worker
            observations
        worker_actions : array_like
            (batch_size, w_ac_dim, meta_period) matrix of Worker actions

        Returns
        -------
        array_like
            (batch_size, w_obs_dim, meta_period + 1) matrix of relabeled Worker
            observations
        array_like
            (batch_size, w_ac_dim, meta_period) matrix of relabeled Worker
            actions
        array_like
            (batch_size, meta_period) matrix of relabeled Worker rewards
        """
        # Collect dimensions.
        batch_size, m_ac_dim = meta_action.shape
        _, w_obs_dim, _ = worker_obses.shape
        _, w_ac_dim, meta_period = worker_actions.shape

        # The relabeled samples will be stored here.
        new_worker_obses = np.zeros((batch_size, w_obs_dim, meta_period + 1))
        new_worker_actions = np.zeros((batch_size, w_ac_dim, meta_period))
        new_worker_rewards = np.zeros((batch_size, meta_period))

        for i in range(batch_size):
            # Collect the initial samples.
            obs0 = worker_obses[i, :, 0]
            action, obs1, rew = self._get_model_next_step(obs0)

            # Add the initial samples.
            new_worker_actions[i, :, 0] = obs0
            new_worker_obses[i, :, 0] = obs0
            new_worker_obses[i, :, 1] = obs1
            new_worker_rewards[i, 0] = rew
            obs0 = obs1.copy()

            # TODO: if it ends prematurely?
            for j in range(1, meta_period):
                # Compute the next step variables.
                action, obs1, rew = self._get_model_next_step(obs0)

                # Add the next step samples.
                new_worker_actions[i, :, j] = obs0
                new_worker_obses[i, :, j] = obs0
                new_worker_obses[i, :, j + 1] = obs1
                new_worker_rewards[i, j] = rew
                obs0 = obs1.copy()

        return new_worker_obses, new_worker_actions, new_worker_rewards

    def _get_model_next_step(self, obs0):
        """Compute the next-step information using the trained model.

        Parameters
        ----------
        obs0 : array_like
            the current step worker observation, including the goal

        Returns
        -------
        array_like
            current step Worker action
        array_like
            next step Worker observation
        float
            the intrinsic reward
        """
        # Separate the observation and goal.
        goal_dim = self.manager.ac_space.shape[0]
        worker_obs0 = obs0[:goal_dim]
        goal = obs0[:goal_dim]

        # Compute the action using the current instantiation of the policy.
        worker_action = self.worker.get_action(worker_obs0, goal, False, False)

        # Use the model to compute the next step observation.
        delta = self.sess.run(
            self.worker_model,
            feed_dict={
                self.worker_obs_ph: worker_obs0,
                self.worker_action_ph: worker_action
            }
        )
        worker_obs1 = worker_obs0 + delta

        # Compute the intrinsic reward.
        worker_reward = self.worker_reward_scale * \
            self.worker_reward_fn(worker_obs0, goal, worker_obs1)

        # Compute the next step goal and add it to the observation.
        next_goal = self.goal_transition_fn(worker_obs0, goal, worker_obs1)
        worker_obs1 = np.append(worker_obs1, next_goal)

        return worker_action, worker_obs1, worker_reward

    def _sample_from_relabeled(self,
                               worker_obses,
                               worker_actions,
                               worker_rewards):
        """Collect a batch of samples from the relabeled samples.

        Parameters
        ----------
        worker_obses : array_like
            (batch_size, w_obs_dim, meta_period + 1) matrix of relabeled Worker
            observations
        worker_actions : array_like
            (batch_size, w_ac_dim, meta_period) matrix of relabeled Worker
            actions
        worker_rewards : array_like
            (batch_size, meta_period) matrix of relabeled Worker rewards

        Returns
        -------
        array_like
            (batch_size, m_obs_dim) matrix of relabeled next step Manager
            observations
        array_like
            (batch_size, worker_obs) matrix of relabeled Worker observations
        array_like
            (batch_size, worker_obs) matrix of relabeled next step Worker
            observations
        array_like
            (batch_size, worker_ac) matrix of relabeled Worker actions
        array_like
            (batch_size,) vector of relabeled Worker rewards
        """
        batch_size, w_obs_dim, _ = worker_obses.shape
        _, w_ac_dim, _ = worker_actions.shape
        goal_dim = self.manager.ac_space.shape[0]

        meta_obs1 = np.array((batch_size, goal_dim))
        worker_obs0 = np.array((batch_size, w_obs_dim))
        worker_obs1 = np.array((batch_size, w_obs_dim))
        worker_act = np.array((batch_size, w_ac_dim))
        worker_rew = np.array(batch_size)

        for i in range(batch_size):
            indx_val = random.randint(0, worker_obses.shape[2] - 2)
            meta_obs1 = worker_obses[i, -goal_dim:, -1]
            worker_obs0[i, :] = worker_obses[i, :, indx_val]
            worker_obs1[i, :] = worker_obses[i, :, indx_val + 1]
            worker_act[i, :] = worker_actions[i, :, indx_val]
            worker_rew[i] = worker_rewards[i, indx_val]

        return meta_obs1, worker_obs0, worker_obs1, worker_act, worker_rew
