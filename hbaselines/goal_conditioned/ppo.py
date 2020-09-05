"""Base goal-conditioned hierarchical policy."""
import tensorflow as tf
import numpy as np
from copy import deepcopy

from hbaselines.base_policies import OnPolicyPolicy
from hbaselines.fcnet.ppo import FeedForwardPolicy
from hbaselines.utils.reward_fns import negative_distance
from hbaselines.utils.env_util import get_meta_ac_space, get_state_indices


class GoalConditionedPolicy(OnPolicyPolicy):
    """Goal-conditioned hierarchical reinforcement learning model.

    Attributes
    ----------
    meta_period : int
        meta-policy action period
    intrinsic_reward_type : str
        the reward function to be used by the worker. Must be one of:

        * "negative_distance": the negative two norm between the states and
          desired absolute or relative goals.
        * "scaled_negative_distance": similar to the negative distance reward
          where the states, goals, and next states are scaled by the inverse of
          the action space of the manager policy
        * "non_negative_distance": the negative two norm between the states and
          desired absolute or relative goals offset by the maximum goal space
          (to ensure non-negativity)
        * "scaled_non_negative_distance": similar to the non-negative distance
          reward where the states, goals, and next states are scaled by the
          inverse of the action space of the manager policy
        * "exp_negative_distance": equal to exp(-negative_distance^2). The
          result is a reward between 0 and 1. This is useful for policies that
          terminate early.
        * "scaled_exp_negative_distance": similar to the previous worker reward
          type but with states, actions, and next states that are scaled.
    intrinsic_reward_scale : float
        the value that the intrinsic reward should be scaled by
    relative_goals : bool
        specifies whether the goal issued by the higher-level policies is meant
        to be a relative or absolute goal, i.e. specific state or change in
        state
    off_policy_corrections : bool
        whether to use off-policy corrections during the update procedure. See:
        https://arxiv.org/abs/1805.08296.
    hindsight : bool
        whether to use hindsight action and goal transitions, as well as
        subgoal testing. See: https://arxiv.org/abs/1712.00948
    subgoal_testing_rate : float
        rate at which the original (non-hindsight) sample is stored in the
        replay buffer as well. Used only if `hindsight` is set to True.
    cooperative_gradients : bool
        whether to use the cooperative gradient update procedure for the
        higher-level policy. See: https://arxiv.org/abs/1912.02368v1
    cg_weights : float
        weights for the gradients of the loss of the lower-level policies with
        respect to the parameters of the higher-level policies. Only used if
        `cooperative_gradients` is set to True.
    policy : list of hbaselines.base_policies.ActorCriticPolicy
        a list of policy object for each level in the hierarchy, order from
        highest to lowest level policy
    goal_indices : list of int
        the state indices for the intrinsic rewards
    intrinsic_reward_fn : function
        reward function for the lower-level policies
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 co_space,
                 verbose,
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
                 scope=None,
                 env_name="",
                 num_envs=1,
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
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
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
        additional_params : dict
            additional algorithm-specific policy parameters. Used internally by
            the class when instantiating other (child) policies.
        """
        meta_policy = FeedForwardPolicy
        worker_policy = FeedForwardPolicy

        super(GoalConditionedPolicy, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            co_space=co_space,
            verbose=verbose,
            model_params=model_params,
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
            num_envs=num_envs,
        )

        # Run assertions.
        assert num_levels >= 2, "num_levels must be greater than or equal to 2"
        assert not off_policy_corrections, \
            "The `off_policy_corrections` is not available for PPO. All " \
            "samples are on-policy, and hence do not require corrections."

        self.num_levels = num_levels
        self.meta_period = meta_period
        self.intrinsic_reward_type = intrinsic_reward_type
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.relative_goals = relative_goals
        self.off_policy_corrections = off_policy_corrections
        self.hindsight = hindsight
        self.subgoal_testing_rate = subgoal_testing_rate
        self.cooperative_gradients = cooperative_gradients
        self.cg_weights = cg_weights

        # Get the observation and action space of the higher level policies.
        meta_ac_space = get_meta_ac_space(
            ob_space=ob_space,
            relative_goals=relative_goals,
            env_name=env_name,
        )

        # =================================================================== #
        # Step 1: Create the policies for the individual levels.              #
        # =================================================================== #

        self.policy = []

        # The policies are ordered from the highest level to lowest level
        # policies in the hierarchy.
        for i in range(num_levels):
            # Determine the appropriate parameters to use for the policy in the
            # current level.
            policy_fn = meta_policy if i < (num_levels - 1) else worker_policy
            ac_space_i = meta_ac_space if i < (num_levels - 1) else ac_space
            co_space_i = co_space if i == 0 else meta_ac_space
            ob_space_i = ob_space

            # The policies are ordered from the highest level to lowest level
            # policies in the hierarchy.
            with tf.compat.v1.variable_scope("level_{}".format(i)):
                # Compute the scope name based on any outer scope term.
                scope_i = "level_{}".format(i)
                if scope is not None:
                    scope_i = "{}/{}".format(scope, scope_i)

                # Create the next policy.
                self.policy.append(policy_fn(
                    sess=sess,
                    ob_space=ob_space_i,
                    ac_space=ac_space_i,
                    co_space=co_space_i,
                    verbose=verbose,
                    model_params=model_params,
                    scope=scope_i,
                    **(additional_params or {}),
                ))

        # =================================================================== #
        # Step 2: Create attributes for storing on-policy data.               #
        # =================================================================== #

        # Create variables to store on-policy data.
        storage_list = [
            [[] for _ in range(num_envs)] for _ in range(num_levels)]
        self.mb_rewards = deepcopy(storage_list)
        self.mb_obs = deepcopy(storage_list)
        self.mb_contexts = deepcopy(storage_list)
        self.mb_actions = deepcopy(storage_list)
        self.mb_values = deepcopy(storage_list)
        self.mb_neglogpacs = deepcopy(storage_list)
        self.mb_dones = deepcopy(storage_list)
        self.mb_all_obs = deepcopy(storage_list)
        self.mb_returns = deepcopy(storage_list)
        self.last_obs = deepcopy(storage_list)
        self.mb_advs = None

        # the time since the most recent sample began collecting step samples
        self._t_start = [0 for _ in range(num_levels)]

        # current action by the meta-level policies
        self._meta_action = [[None for _ in range(num_levels - 1)]
                             for _ in range(num_envs)]

        # Collect the state indices for the intrinsic rewards.
        self.goal_indices = get_state_indices(ob_space, env_name)

        # Define the intrinsic reward function.
        if intrinsic_reward_type in ["negative_distance",
                                     "scaled_negative_distance",
                                     "non_negative_distance",
                                     "scaled_non_negative_distance",
                                     "exp_negative_distance",
                                     "scaled_exp_negative_distance"]:
            # Offset the distance measure by the maximum possible distance to
            # ensure non-negativity.
            if "non_negative" in intrinsic_reward_type:
                offset = np.sqrt(np.sum(np.square(
                    meta_ac_space.high - meta_ac_space.low), -1))
            else:
                offset = 0

            # Scale the outputs from the state by the meta-action space if you
            # wish to scale the worker reward.
            if intrinsic_reward_type.startswith("scaled"):
                scale = 0.5 * (meta_ac_space.high - meta_ac_space.low)
            else:
                scale = 1

            def intrinsic_reward_fn(states, goals, next_states):
                return negative_distance(
                    states=states[self.goal_indices] / scale,
                    goals=goals / scale,
                    next_states=next_states[self.goal_indices] / scale,
                    relative_context=relative_goals,
                    offset=0.0
                ) + offset

            # Perform the exponential and squashing operations to keep the
            # intrinsic reward between 0 and 1.
            if "exp" in intrinsic_reward_type:
                def exp_intrinsic_reward_fn(states, goals, next_states):
                    return np.exp(
                        -intrinsic_reward_fn(states, goals, next_states) ** 2)
                self.intrinsic_reward_fn = exp_intrinsic_reward_fn
            else:
                self.intrinsic_reward_fn = intrinsic_reward_fn
        else:
            raise ValueError("Unknown intrinsic reward type: {}".format(
                intrinsic_reward_type))

        # =================================================================== #
        # Step 3: Create algorithm-specific features.                         #
        # =================================================================== #

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

        if self.cooperative_gradients:
            with tf.compat.v1.variable_scope(scope):
                self._setup_cooperative_gradients()

    def initialize(self):
        """See parent class.

        This method calls the initialization methods of the policies at every
        level of the hierarchy.
        """
        for i in range(self.num_levels):
            self.policy[i].initialize()

    def update(self, update_actor=True, **kwargs):
        """Perform a gradient update step.

        This is done both at every level of the hierarchy.

        The kwargs argument for this method contains two additional terms:

        * update_meta (bool): specifies whether to perform a gradient update
          step for the meta-policies
        * update_meta_actor (bool): similar to the `update_policy` term, but
          for the meta-policy. Note that, if `update_meta` is set to False,
          this term is void.

        **Note**; The target update soft updates for all policies occur at the
        same frequency as their respective actor update frequencies.

        Parameters
        ----------
        update_actor : bool
            specifies whether to update the actor policy. The critic policy is
            still updated if this value is set to False.

        Returns
        -------
         ([float, float], [float, float])
            the critic loss for every policy in the hierarchy
        (float, float)
            the actor loss for every policy in the hierarchy
        """
        # Perform the update for every level in the hierarchy.
        for level in range(self.num_levels):
            # Compute the last estimated value.
            last_values = [
                self.sess.run(
                    self.policy[level].value_flat,
                    {self.policy[level].obs_ph: self.last_obs[level][env_num]}
                ) for env_num in range(self.num_envs)
            ]

            (self.mb_obs[level],
             self.mb_contexts[level],
             self.mb_actions[level],
             self.mb_values[level],
             self.mb_neglogpacs[level],
             self.mb_all_obs[level],
             self.mb_rewards[level],
             self.mb_returns[level],
             self.mb_dones[level],
             self.mb_advs[level], n_steps) = self.process_minibatch(
                mb_obs=self.mb_obs[level],
                mb_contexts=self.mb_contexts[level],
                mb_actions=self.mb_actions[level],
                mb_values=self.mb_values[level],
                mb_neglogpacs=self.mb_neglogpacs[level],
                mb_all_obs=self.mb_all_obs[level],
                mb_rewards=self.mb_rewards[level],
                mb_returns=self.mb_returns[level],
                mb_dones=self.mb_dones[level],
                last_values=last_values,
            )

            # Run the optimization procedure.
            batch_size = n_steps // self.n_minibatches

            inds = np.arange(n_steps)
            for _ in range(self.n_opt_epochs):
                np.random.shuffle(inds)
                for start in range(0, n_steps, batch_size):
                    end = start + batch_size
                    mbinds = inds[start:end]
                    self.policy[level].update_from_batch(
                        obs=self.mb_obs[level][mbinds],
                        context=None if self.mb_contexts[level][0] is None
                        else self.mb_contexts[level][mbinds],
                        returns=self.mb_returns[level][mbinds],
                        actions=self.mb_actions[level][mbinds],
                        values=self.mb_values[level][mbinds],
                        advs=self.mb_advs[level][mbinds],
                        neglogpacs=self.mb_neglogpacs[level][mbinds],
                    )

    def get_action(self, obs, context, apply_noise, random_actions, env_num=0):
        """See parent class."""
        # Loop through the policies in the hierarchy.
        for i in range(self.num_levels - 1):
            if self._update_meta(i, env_num):
                context_i = context if i == 0 \
                    else self._meta_action[env_num][i - 1]

                # Update the meta action based on the output from the policy if
                # the time period requires is.
                self._meta_action[env_num][i] = self.policy[i].get_action(
                    obs, context_i, apply_noise, random_actions)
            else:
                # Update the meta-action in accordance with a fixed transition
                # function.
                self._meta_action[env_num][i] = self.goal_transition_fn(
                    obs0=np.array(
                        [self.last_obs[0][env_num][:, self.goal_indices]]),
                    goal=self._meta_action[env_num][i],
                    obs1=obs[:, self.goal_indices]
                )

        # Return the action to be performed within the environment (i.e. the
        # action by the lowest level policy).
        action = self.policy[-1].get_action(
            obs, self._meta_action[env_num][-1], apply_noise, random_actions)

        return action

    def store_transition(self, obs0, context0, action, reward, obs1, context1,
                         done, is_final_step, env_num=0, evaluate=False):
        """See parent class."""
        for level in range(self.num_levels):
            # Actions and intrinsic rewards for the high-level policies are
            # only updated when the action is recomputed by the graph.
            if self._t_start[env_num] % self.meta_period ** level == 0 or done:
                # Update the minibatch of samples.
                self.mb_rewards[level][env_num].append(0)
                self.mb_obs[level][env_num].append([obs0])
                self.mb_contexts[level][env_num].append(context0)
                self.mb_actions[level][env_num].append(
                    [action] if level == 0
                    else self._meta_action[env_num][level])
                # FIXME: non-negative rewards?
                self.mb_dones[level][env_num].append(
                    1 if level < self.num_levels - 1
                    and not self._negative_reward_fn() else done)

                # Update the last observation (to compute the last value for
                # the GAE expected returns).
                self.last_obs[level][env_num] = self._get_obs([obs1], context1)

            # Add to the most recent reward the return from the current step.
            if level == self.num_levels - 1:
                self.mb_rewards[level][env_num][-1] += reward
            else:
                self.mb_rewards[level][env_num][-1] += \
                    self.intrinsic_reward_scale / \
                    self.meta_period ** (level - 1) * \
                    self.intrinsic_reward_fn(
                        states=obs0,
                        goals=self._meta_action[env_num][level].flatten(),
                        next_states=obs1
                    )

        # Increment the time since the highest level meta period started.
        self._t_start[env_num] += 1

        # Check if the final meta period is done.
        if self._t_start[env_num] == self.meta_period ** self.num_levels \
                or done:
            self._t_start[env_num] = 0

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
        return self._t_start[env_num] % self.meta_period ** level == 0

    def clear_memory(self, env_num):
        """Do nothing."""
        pass

    def get_td_map(self):
        """See parent class."""
        td_map = {}
        for level in range(self.num_levels):
            # Add the contextual observation, if applicable.
            context = \
                None if self.mb_contexts[level][0] is None \
                else self.mb_contexts[level]
            obs = self._get_obs(self.mb_obs[level], context, axis=1)

            td_map.update(self.policy[level].get_td_map_from_batch(
                obs=obs.copy(),
                mb_actions=self.mb_actions[level],
                mb_advs=self.mb_advs[level],
                mb_returns=self.mb_returns[level],
                mb_neglogpacs=self.mb_neglogpacs[level],
                mb_values=self.mb_values[level]
            ))

        # Clear memory.
        storage_list = [
            [[] for _ in range(self.num_envs)] for _ in range(self.num_levels)]
        self.mb_rewards = deepcopy(storage_list)
        self.mb_obs = deepcopy(storage_list)
        self.mb_contexts = deepcopy(storage_list)
        self.mb_actions = deepcopy(storage_list)
        self.mb_values = deepcopy(storage_list)
        self.mb_neglogpacs = deepcopy(storage_list)
        self.mb_dones = deepcopy(storage_list)
        self.mb_all_obs = deepcopy(storage_list)
        self.mb_returns = deepcopy(storage_list)
        self.last_obs = deepcopy(storage_list)
        self.mb_advs = None

        return td_map

    def _negative_reward_fn(self):
        """Return True if the intrinsic reward returns negative values.

        Intrinsic reward functions with negative rewards incentivize early
        terminations, which we attempt to mitigate in the training operation by
        preventing early terminations from return an expected return of 0.
        """
        return "exp" not in self.intrinsic_reward_type \
            and "non" not in self.intrinsic_reward_type

    # ======================================================================= #
    #                       Auxiliary methods for HAC                         #
    # ======================================================================= #

    def _hindsight_actions_goals(self, initial_observations, initial_rewards):
        """Calculate hindsight goal and action transitions.

        These are then stored in the replay buffer along with the original
        (non-hindsight) sample.

        See the README at the front page of this repository for an in-depth
        description of this procedure.

        Parameters
        ----------
        initial_observations : array_like
            the original worker observations with the non-hindsight goals
            appended to them
        initial_rewards : array_like
            the original intrinsic rewards

        Returns
        -------
        array_like
            the goal at every step in hindsight
        array_like
            the modified intrinsic rewards taking into account the hindsight
            goals

        Helps
        -----
        * store_transition(self):
        """
        new_goals = []
        observations = deepcopy(initial_observations)
        rewards = deepcopy(initial_rewards)
        hindsight_goal = 0 if self.relative_goals \
            else observations[-1][self.goal_indices]
        obs_tp1 = observations[-1]

        for i in range(1, len(observations) + 1):
            obs_t = observations[-i]

            # Calculate the hindsight goal in using relative goals.
            # If not, the hindsight goal is simply a subset of the
            # final state observation.
            if self.relative_goals:
                hindsight_goal += \
                    obs_tp1[self.goal_indices] - obs_t[self.goal_indices]

            # Modify the Worker intrinsic rewards based on the new
            # hindsight goal.
            if i > 1:
                rewards[-(i - 1)] = self.intrinsic_reward_scale \
                    * self.intrinsic_reward_fn(obs_t, hindsight_goal, obs_tp1)

            obs_tp1 = deepcopy(obs_t)
            new_goals = [deepcopy(hindsight_goal)] + new_goals

        return new_goals, rewards

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
