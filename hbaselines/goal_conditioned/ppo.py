"""PPO-compatible goal-conditioned hierarchical policy."""
import tensorflow as tf
import numpy as np
from copy import deepcopy

from hbaselines.goal_conditioned.base import GoalConditionedPolicy \
    as BasePolicy
from hbaselines.fcnet.ppo import FeedForwardPolicy
from hbaselines.utils.tf_util import process_minibatch


class GoalConditionedPolicy(BasePolicy):
    """PPO-compatible goal-conditioned hierarchical policy."""

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
                 num_envs=1):
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
        l2_penalty : float
            L2 regularization penalty. This is applied to the policy network.
        model_params : dict
            dictionary of model-specific parameters. See parent class.
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
        """
        super(GoalConditionedPolicy, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            co_space=co_space,
            verbose=verbose,
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
            meta_policy=FeedForwardPolicy,
            worker_policy=FeedForwardPolicy,
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

        # Run assertions.
        assert not off_policy_corrections, \
            "The `off_policy_corrections` is not available for PPO. All " \
            "samples are on-policy, and hence do not require corrections."

        # =================================================================== #
        # Create attributes for storing on-policy data.                       #
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
        self.mb_advs = [[] for _ in range(self.num_levels)]

        # the time since the most recent sample began collecting step samples
        self._t_start = [0 for _ in range(num_levels)]

    def update(self, update_actor=True, **kwargs):
        """See parent class."""
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
             self.mb_advs[level], n_steps) = process_minibatch(
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
                gamma=self.gamma,
                lam=self.lam,
                num_envs=self.num_envs,
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

    def store_transition(self, obs0, context0, action, reward, obs1, context1,
                         done, is_final_step, env_num=0, evaluate=False):
        """See parent class."""
        for i in range(self.num_levels):
            level = self.num_levels - (i + 1)
            # Actions and intrinsic rewards for the high-level policies are
            # only updated when the action is recomputed by the graph.
            if self._t_start[env_num] % self.meta_period ** i == 0 or done:
                # Update the minibatch of samples.
                self.mb_rewards[level][env_num].append(0)
                self.mb_obs[level][env_num].append([obs0])
                self.mb_contexts[level][env_num].append(
                    context0 if level == 0
                    else list(self.meta_action[env_num][level - 1].flatten()))
                self.mb_actions[level][env_num].append(
                    [action] if i == 0
                    else self.meta_action[env_num][level])
                self.mb_dones[level][env_num].append(done)
                self.mb_values[level][env_num].append(
                    self.policy[level].mb_values[env_num][-1])
                self.mb_neglogpacs[level][env_num].append(
                    self.policy[level].mb_neglogpacs[env_num][-1])

                # Update the last observation (to compute the last value for
                # the GAE expected returns).
                self.last_obs[level][env_num] = self._get_obs(
                    [obs1],
                    context1 if level == 0
                    else self.meta_action[env_num][level - 1], axis=1)

            # Add to the most recent reward the return from the current step.
            if level == 0:
                self.mb_rewards[level][env_num][-1] += reward
            else:
                self.mb_rewards[level][env_num][-1] += \
                    self.intrinsic_reward_scale / \
                    self.meta_period ** (level - 1) * \
                    self.intrinsic_reward_fn(
                        states=obs0,
                        goals=self.meta_action[env_num][level - 1].flatten(),
                        next_states=obs1
                    )

        # Increment the time since the highest level meta period started.
        self._t_start[env_num] += 1

        # Check if the final meta period is done.
        if self._t_start[env_num] == self.meta_period ** self.num_levels \
                or done:
            self._t_start[env_num] = 0

    def _update_meta(self, level, env_num):
        """See parent class."""
        return self._t_start[env_num] % self.meta_period ** level == 0

    def clear_memory(self, env_num):
        """See parent class."""
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
                mb_rewards=self.mb_rewards[level],
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
        self.mb_advs = [[] for _ in range(self.num_levels)]

        # Reset the meta-policy timers.
        self._t_start = [0 for _ in range(self.num_levels)]

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
        """See parent class."""
        raise NotImplementedError

    # ======================================================================= #
    #                       Auxiliary methods for CHER                        #
    # ======================================================================= #

    def _setup_cooperative_gradients(self):
        """See parent class."""
        raise NotImplementedError

    def _cooperative_gradients_update(self,
                                      obs0,
                                      actions,
                                      rewards,
                                      obs1,
                                      terminals1,
                                      level_num,
                                      update_actor=True):
        """See parent class."""
        raise NotImplementedError
