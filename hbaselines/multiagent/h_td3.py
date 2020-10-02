"""TD3-compatible multi-agent goal-conditioned hierarchical policy."""
from hbaselines.multiagent.base import MultiActorCriticPolicy as BasePolicy
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy


class MultiGoalConditionedPolicy(BasePolicy):
    """TD3-compatible multi-agent goal-conditioned hierarchical policy."""

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
                 shared,
                 maddpg,
                 env_name="",
                 num_envs=1,
                 all_ob_space=None,
                 n_agents=1,
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
            Actions by the high-level policy are randomly sampled from its
            action space.
        pretrain_path : str or None
            path to the pre-trained worker policy checkpoints
        pretrain_ckpt : int or None
            checkpoint number to use within the worker policy path. If set to
            None, the most recent checkpoint is used.
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
        super(MultiGoalConditionedPolicy, self).__init__(
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
            shared=shared,
            maddpg=maddpg,
            all_ob_space=all_ob_space,
            n_agents=n_agents,
            base_policy=GoalConditionedPolicy,
            scope=scope,
            additional_params=dict(
                noise=noise,
                target_policy_noise=target_policy_noise,
                target_noise_clip=target_noise_clip,
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
                env_name=env_name,
                num_envs=n_agents * num_envs if shared else num_envs,
            ),
        )

    def _setup_maddpg(self, scope):
        """See setup."""
        raise NotImplementedError(
            "This policy does not support MADDPG-variants of the training "
            "operation.")

    def _initialize_maddpg(self):
        """See initialize."""
        raise NotImplementedError(
            "This policy does not support MADDPG-variants of the training "
            "operation.")

    def _update_maddpg(self, update_actor=True, **kwargs):
        """See update."""
        raise NotImplementedError(
            "This policy does not support MADDPG-variants of the training "
            "operation.")

    def _get_action_maddpg(self,
                           obs,
                           context,
                           apply_noise,
                           random_actions,
                           env_num):
        """See get_action."""
        raise NotImplementedError(
            "This policy does not support MADDPG-variants of the training "
            "operation.")

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
        raise NotImplementedError(
            "This policy does not support MADDPG-variants of the training "
            "operation.")

    def _get_td_map_maddpg(self):
        """See get_td_map."""
        raise NotImplementedError(
            "This policy does not support MADDPG-variants of the training "
            "operation.")
