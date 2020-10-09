"""Base goal-conditioned hierarchical policy."""
import tensorflow as tf
import numpy as np
from copy import deepcopy
import random

from hbaselines.goal_conditioned.base import GoalConditionedPolicy \
    as BasePolicy
from hbaselines.goal_conditioned.replay_buffer import HierReplayBuffer
from hbaselines.utils.env_util import get_meta_ac_space


class GoalConditionedPolicy(BasePolicy):
    """Actor-critic variant of the goal-conditioned hierarchical policy."""

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
                 num_levels,
                 meta_period,
                 intrinsic_reward_type,
                 intrinsic_reward_scale,
                 relative_goals,
                 off_policy_corrections,
                 hindsight,
                 subgoal_testing_rate,
                 cooperative_gradients,
                 cg_weights,
                 pretrain_worker,
                 pretrain_path,
                 pretrain_ckpt,
                 scope=None,
                 env_name="",
                 num_envs=1,
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
        use_huber : bool
            specifies whether to use the huber distance function as the loss
            for the critic. If set to False, the mean-squared error metric is
            used instead
        model_params : dict
            dictionary of model-specific parameters. See parent class.
        num_levels : int
            number of levels within the hierarchy. Must be greater than 1. Two
            levels correspond to a Manager/Worker paradigm.
        meta_period : int
            meta-policy action period
        intrinsic_reward_type : str
            the reward function to be used by the worker. Must be one of:

            * "negative_distance": the negative two norm between the states and
              desired absolute or relative goals.
            * "scaled_negative_distance": similar to the negative distance
              reward where the states, goals, and next states are scaled by the
              inverse of the action space of the manager policy
            * "non_negative_distance": the negative two norm between the states
              and desired absolute or relative goals offset by the maximum goal
              space (to ensure non-negativity)
            * "scaled_non_negative_distance": similar to the non-negative
              distance reward where the states, goals, and next states are
              scaled by the inverse of the action space of the manager policy
            * "exp_negative_distance": equal to exp(-negative_distance^2). The
              result is a reward between 0 and 1. This is useful for policies
              that terminate early.
            * "scaled_exp_negative_distance": similar to the previous worker
              reward type but with states, actions, and next states that are
              scaled.
        intrinsic_reward_scale : float
            the value that the intrinsic reward should be scaled by
        relative_goals : bool
            specifies whether the goal issued by the higher-level policies is
            meant to be a relative or absolute goal, i.e. specific state or
            change in state
        off_policy_corrections : bool
            whether to use off-policy corrections during the update procedure.
            See: https://arxiv.org/abs/1805.08296
        hindsight : bool
            whether to include hindsight action and goal transitions in the
            replay buffer. See: https://arxiv.org/abs/1712.00948
        subgoal_testing_rate : float
            rate at which the original (non-hindsight) sample is stored in the
            replay buffer as well. Used only if `hindsight` is set to True.
        cooperative_gradients : bool
            whether to use the cooperative gradient update procedure for the
            higher-level policy. See: https://arxiv.org/abs/1912.02368v1
        cg_weights : float
            weights for the gradients of the loss of the lower-level policies
            with respect to the parameters of the higher-level policies. Only
            used if `cooperative_gradients` is set to True.
        pretrain_worker : bool
            specifies whether you are pre-training the lower-level policies.
            Actions by the high-level policy are randomly sampled from the
            action space.
        pretrain_path : str or None
            path to the pre-trained worker policy checkpoints
        pretrain_ckpt : int or None
            checkpoint number to use within the worker policy path. If set to
            None, the most recent checkpoint is used.
        meta_policy : type [ hbaselines.base_policies.ActorCriticPolicy ]
            the policy model to use for the meta policies
        worker_policy : type [ hbaselines.base_policies.ActorCriticPolicy ]
            the policy model to use for the worker policy
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
            use_huber=use_huber,
            l2_penalty=l2_penalty,
            model_params=model_params,
            num_levels=num_levels,
            meta_period=meta_period,
            intrinsic_reward_type=intrinsic_reward_type,
            intrinsic_reward_scale=intrinsic_reward_scale,
            relative_goals=relative_goals,
            off_policy_corrections=off_policy_corrections,
            hindsight=hindsight,
            subgoal_testing_rate=subgoal_testing_rate,
            cooperative_gradients=cooperative_gradients,
            cg_weights=cg_weights,
            pretrain_worker=pretrain_worker,
            pretrain_path=pretrain_path,
            pretrain_ckpt=pretrain_ckpt,
            scope=scope,
            env_name=env_name,
            num_envs=num_envs,
            meta_policy=meta_policy,
            worker_policy=worker_policy,
            additional_params=additional_params,
        )

        # =================================================================== #
        # Create attributes for the replay buffer.                            #
        # =================================================================== #

        # Get the observation and action space of the higher level policies.
        meta_ac_space = get_meta_ac_space(
            ob_space=ob_space,
            relative_goals=relative_goals,
            env_name=env_name,
        )

        # Create the replay buffer.
        self.replay_buffer = HierReplayBuffer(
            buffer_size=int(buffer_size/meta_period),
            batch_size=batch_size,
            meta_period=meta_period,
            obs_dim=ob_space.shape[0],
            ac_dim=ac_space.shape[0],
            co_dim=None if co_space is None else co_space.shape[0],
            goal_dim=meta_ac_space.shape[0],
            num_levels=num_levels
        )

        # a list of all the actions performed by each level in the hierarchy,
        # ordered from highest to lowest level policy. A separate element is
        # used for each environment.
        self._actions = [[[] for _ in range(self.num_levels)]
                         for _ in range(num_envs)]

        # a list of the rewards (intrinsic or other) experienced by every level
        # in the hierarchy, ordered from highest to lowest level policy. A
        # separate element is used for each environment.
        self._rewards = [[[0]] + [[] for _ in range(self.num_levels - 1)]
                         for _ in range(num_envs)]

        # a list of observations that stretch as long as the dilated horizon
        # chosen for the highest level policy. A separate element is used for
        # each environment.
        self._observations = [[] for _ in range(num_envs)]

        # the first and last contextual term. A separate element is used for
        # each environment.
        self._contexts = [[] for _ in range(num_envs)]

        # a list of done masks at every time step. A separate element is used
        # for each environment.
        self._dones = [[] for _ in range(num_envs)]

    def _sample_buffer(self, with_additional=False):
        """See parent class."""
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample():
            return None

        # Get a batch.
        obs0, obs1, act, rew, done, additional = self.replay_buffer.sample(
            with_additional)

        # Do not use done masks for lower-level policies with negative
        # intrinsic rewards (these cause the policies to terminate early).
        if self._negative_reward_fn():
            for i in range(self.num_levels - 1):
                done[i+1] = np.array([False] * done[i+1].shape[0])

        return obs0, obs1, act, rew, done, additional

    def store_transition(self, obs0, context0, action, reward, obs1, context1,
                         done, is_final_step, env_num=0, evaluate=False):
        """See parent class."""
        # the time since the most recent sample began collecting step samples
        t_start = len(self._observations[env_num])

        # Flatten the observations.
        obs0 = obs0.flatten()
        obs1 = obs1.flatten()

        for i in range(1, self.num_levels):
            # Actions and intrinsic rewards for the high-level policies are
            # only updated when the action is recomputed by the graph.
            if t_start % self.meta_period ** (i-1) == 0:
                self._rewards[env_num][-i].append(0)
                self._actions[env_num][-i-1].append(
                    self.meta_action[env_num][-i].flatten())

            # Compute the intrinsic rewards and append them to the list of
            # rewards.
            self._rewards[env_num][-i][-1] += \
                self.intrinsic_reward_scale / self.meta_period ** (i-1) * \
                self.intrinsic_reward_fn(
                    states=obs0,
                    goals=self.meta_action[env_num][-i].flatten(),
                    next_states=obs1
                )

        # The highest level policy receives the sum of environmental rewards.
        self._rewards[env_num][0][0] += reward

        # The lowest level policy's actions are received from the algorithm.
        self._actions[env_num][-1].append(action)

        # Add the environmental observations and contextual terms to their
        # respective lists.
        self._observations[env_num].append(obs0)
        if t_start == 0:
            self._contexts[env_num].append(context0)

        # Modify the done mask in accordance with the TD3 algorithm. Done masks
        # that correspond to the final step are set to False.
        self._dones[env_num].append(done and not is_final_step)

        # Add a sample to the replay buffer.
        if len(self._observations[env_num]) == \
                self.meta_period ** (self.num_levels - 1) or done:
            # Add the last observation and context.
            self._observations[env_num].append(obs1)
            self._contexts[env_num].append(context1)

            # Compute the current state goals to add to the final observation.
            for i in range(self.num_levels - 1):
                self._actions[env_num][i].append(self.goal_transition_fn(
                    obs0=obs0[self.goal_indices],
                    goal=self.meta_action[env_num][i],
                    obs1=obs1[self.goal_indices]
                ).flatten())

            # Avoid storing samples when performing evaluations.
            if not evaluate:
                if not self.hindsight \
                        or random.random() < self.subgoal_testing_rate:
                    # Store a sample in the replay buffer.
                    self.replay_buffer.add(
                        obs_t=self._observations[env_num],
                        context_t=self._contexts[env_num],
                        action_t=self._actions[env_num],
                        reward_t=self._rewards[env_num],
                        done_t=self._dones[env_num],
                    )

                if self.hindsight:
                    # Some temporary attributes.
                    worker_obses = [
                        self._get_obs(self._observations[env_num][i],
                                      self._actions[env_num][0][i], 0)
                        for i in range(len(self._observations[env_num]))]
                    intrinsic_rewards = self._rewards[env_num][-1]

                    # Implement hindsight action and goal transitions.
                    goal, rewards = self._hindsight_actions_goals(
                        initial_observations=worker_obses,
                        initial_rewards=intrinsic_rewards
                    )
                    new_actions = deepcopy(self._actions[env_num])
                    new_actions[0] = goal
                    new_rewards = deepcopy(self._rewards[env_num])
                    new_rewards[-1] = rewards

                    # Store the hindsight sample in the replay buffer.
                    self.replay_buffer.add(
                        obs_t=self._observations[env_num],
                        context_t=self._contexts[env_num],
                        action_t=new_actions,
                        reward_t=new_rewards,
                        done_t=self._dones[env_num],
                    )

            # Clear the memory that has been stored in the replay buffer.
            self.clear_memory(env_num)

    def _update_meta(self, level, env_num):
        """Determine whether a meta-policy should update its action.

        This is done by checking the length of the observation lists that are
        passed to the replay buffer, which are cleared whenever the highest
        level meta-period has been met or the environment has been reset.

        Parameters
        ----------
        level : int
            the level of the policy
        env_num : int
            the environment number. Used to handle situations when multiple
            parallel environments are being used.

        Returns
        -------
        bool
            True if the action should be updated by the meta-policy at the
            given level
        """
        return len(self._observations[env_num]) % \
            (self.meta_period ** (self.num_levels - level - 1)) == 0

    def clear_memory(self, env_num):
        """Clear internal memory that is used by the replay buffer."""
        self._actions[env_num] = [[] for _ in range(self.num_levels)]
        self._rewards[env_num] = \
            [[0]] + [[] for _ in range(self.num_levels - 1)]
        self._observations[env_num] = []
        self._contexts[env_num] = []
        self._dones[env_num] = []

    def _negative_reward_fn(self):
        """Return True if the intrinsic reward returns negative values.

        Intrinsic reward functions with negative rewards incentivize early
        terminations, which we attempt to mitigate in the training operation by
        preventing early terminations from return an expected return of 0.
        """
        return "exp" not in self.intrinsic_reward_type \
            and "non" not in self.intrinsic_reward_type

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
            (batch_size, m_obs_dim) matrix of meta observations
        meta_obs1 : array_like
            (batch_size, m_obs_dim) matrix of next time step meta observations
        meta_action : array_like
            (batch_size, m_ac_dim) matrix of meta actions
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
            (batch_size, m_ac_dim) matrix of most likely meta actions
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
            (batch_size, m_obs_dim) matrix of meta observations
        meta_obs1 : array_like
            (batch_size, m_obs_dim) matrix of next time step meta observations
        meta_action : array_like
            (batch_size, m_ac_dim) matrix of meta actions
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
        goal_space = self.policy[0].ac_space
        spec_range = goal_space.high - goal_space.low
        random_samples = num_samples - 2

        # Compute the mean and std for the Gaussian distribution to sample
        # from, and well as the maxima and minima.
        loc = meta_obs1[:, self.goal_indices] - meta_obs0[:, self.goal_indices]
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

        # Clip the values based on the meta action space range.
        samples = np.minimum(np.maximum(samples, new_minimum), new_maximum)

        return samples

    def _log_probs(self, meta_actions, worker_obses, worker_actions):
        """Calculate the log probability of the next goal by the meta-policies.

        Parameters
        ----------
        meta_actions : array_like
            (batch_size, m_ac_dim, num_samples) matrix of candidate higher-
            level policy actions
        worker_obses : array_like
            (batch_size, w_obs_dim, meta_period + 1) matrix of lower-level
            policy observations
        worker_actions : array_like
            (batch_size, w_ac_dim, meta_period) list of lower-level policy
            actions

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
    #                       Auxiliary methods for CHER                        #
    # ======================================================================= #

    def _setup_cooperative_gradients(self):
        """Create the cooperative gradients meta-policy optimizer."""
        raise NotImplementedError

    def _cooperative_gradients_update(self,
                                      obs0,
                                      actions,
                                      rewards,
                                      obs1,
                                      terminals1,
                                      level_num,
                                      update_actor=True):
        """Perform the gradient update procedure for the CHER algorithm.

        This procedure is similar to update_from_batch, expect it runs the
        self.cg_optimizer operation instead of the policy object's optimizer,
        and utilizes some information from the worker samples as well.

        Parameters
        ----------
        obs0 : list of array_like
            (batch_size, obs_dim) matrix of observations for every level in the
            hierarchy
        actions : list of array_like
            (batch_size, ac_dim) matrix of actions for every level in the
            hierarchy
        obs1 : list of array_like
            (batch_size, obs_dim) matrix of next step observations for every
            level in the hierarchy
        rewards : list of array_like
            (batch_size,) vector of rewards for every level in the hierarchy
        terminals1 : list of numpy bool
            (batch_size,) vector of done masks for every level in the hierarchy
        level_num : int
            the hierarchy level number of the policy to optimize
        update_actor : bool
            specifies whether to update the actor policy of the meta policy.
            The critic policy is still updated if this value is set to False.

        Returns
        -------
        [float, float]
            meta-policy critic loss
        float
            meta-policy actor loss
        """
        raise NotImplementedError
