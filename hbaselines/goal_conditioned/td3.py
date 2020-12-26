"""TD3-compatible goal-conditioned hierarchical policy."""
import tensorflow as tf
import numpy as np

from hbaselines.goal_conditioned.base import GoalConditionedPolicy as \
    BaseGoalConditionedPolicy
from hbaselines.fcnet.td3 import FeedForwardPolicy
from hbaselines.utils.tf_util import get_trainable_vars


class GoalConditionedPolicy(BaseGoalConditionedPolicy):
    """TD3-compatible goal-conditioned hierarchical policy."""

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
                 cg_delta,
                 pretrain_worker,
                 pretrain_path,
                 pretrain_ckpt,
                 total_steps,
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
        use_huber : bool
            specifies whether to use the huber distance function as the loss
            for the critic. If set to False, the mean-squared error metric is
            used instead
        l2_penalty : float
            L2 regularization penalty. This is applied to the policy network.
        model_params : dict
            dictionary of model-specific parameters. See parent class.
        num_levels : int
            number of levels within the hierarchy. Must be greater than 1. Two
            levels correspond to a Manager/Worker paradigm.
        meta_period : int
            meta-policy action period
        intrinsic_reward_type : str
            the reward function to be used by the lower-level policies. See the
            base goal-conditioned policy for a description.
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
        cg_delta : float
            the desired lower-level expected returns. If set to None, a fixed
            Lagrangian specified by cg_weights is used instead. Only used if
            `cooperative_gradients` is set to True.
        pretrain_worker : bool
            specifies whether you are pre-training the lower-level policies.
            Actions by the high-level policy are randomly sampled from its
            action space.
        pretrain_path : str or None
            path to the pre-trained worker policy checkpoints
        pretrain_ckpt : int or None
            checkpoint number to use within the worker policy path. If set to
            None, the most recent checkpoint is used.
        total_steps : int
            Total number of timesteps used during training. Used by a subset of
            algorithms.
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
            cg_delta=cg_delta,
            scope=scope,
            env_name=env_name,
            pretrain_worker=pretrain_worker,
            pretrain_path=pretrain_path,
            pretrain_ckpt=pretrain_ckpt,
            total_steps=total_steps,
            num_envs=num_envs,
            meta_policy=FeedForwardPolicy,
            worker_policy=FeedForwardPolicy,
            additional_params=dict(
                noise=noise,
                target_policy_noise=target_policy_noise,
                target_noise_clip=target_noise_clip,
            ),
        )

    # ======================================================================= #
    #                       Auxiliary methods for HIRO                        #
    # ======================================================================= #

    # TODO: add support for multi-level hierarchies
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
        fitness = []
        batch_size, goal_dim, num_samples = meta_actions.shape
        _, _, meta_period = worker_actions.shape

        # Loop through the elements of the batch.
        for i in range(batch_size):
            # Extract the candidate goals for the current element in the batch.
            # The worker observations and actions from the meta period of the
            # current batch are also collected to compute the log-probability
            # of a given candidate goal.
            goals_per_sample = meta_actions[i, :, :].T
            worker_obses_per_sample = worker_obses[i, :, :].T
            worker_actions_per_sample = worker_actions[i, :, :].T

            # This will be used to store the cumulative log-probabilities of a
            # given candidate goal for the entire meta-period.
            fitness_per_sample = np.zeros(num_samples)

            # Create repeated representations of each worker observation for
            # each candidate goal. The indexing of worker_obses_per_sample is
            # meant to do the following:
            #  1. We remove the last observation since it does not correspond
            #     to any action for the current meta-period.
            #  2. Since the worker observations contain the goal (context) for
            #     the last `goal_dim` elements, these elements are removed to
            #     only provide the environmental observation.
            tiled_worker_obses_per_sample = np.tile(
                worker_obses_per_sample[:-1, :-goal_dim],
                (num_samples, 1)
            )

            # Create repeated representations of each candidate goal for each
            # worker observation in a meta period.
            tiled_goals_per_sample = np.tile(
                goals_per_sample, meta_period).reshape(
                (num_samples * meta_period, goal_dim))

            # If relative goals are being used, update the later goals to match
            # what they would be under the relative goals difference approach.
            if self.relative_goals:
                goal_diff = worker_obses_per_sample[:-1, :] - np.tile(
                    worker_obses_per_sample[0, :], (meta_period, 1))
                tiled_goals_per_sample += \
                    np.tile(goal_diff, (num_samples, 1))[:, self.goal_indices]

            # Compute the actions the Worker would perform given a specific
            # observation/goal for the current instantiation of the policy.
            pred_actions = self.policy[-1].get_action(
                tiled_worker_obses_per_sample,
                tiled_goals_per_sample,
                apply_noise=False,
                random_actions=False
            )

            # Compute error as the distance between expected and actual actions
            normalized_error = -np.mean(
                np.square(
                    np.tile(worker_actions_per_sample, (num_samples, 1))
                    - pred_actions
                ),
                axis=1
            )

            # Sum the different normalized errors to get the fitness of each
            # candidate goal.
            for j in range(num_samples):
                fitness_per_sample[j] = np.sum(
                    normalized_error[j * meta_period: (j+1) * meta_period])

            fitness.append(fitness_per_sample)

        return np.array(fitness)

    # ======================================================================= #
    #                       Auxiliary methods for CHER                        #
    # ======================================================================= #

    def _setup_cooperative_gradients(self):
        """Create the cooperative gradients meta-policy optimizer."""
        self._n_train_steps = 0

        if self.cg_delta is not None:
            # placeholder for the lambda term.
            self.cg_weights = [
                tf.compat.v1.Variable(initial_value=-4.20, trainable=True)
                for _ in range(self.num_levels - 1)]
        else:
            self.cg_weights = [
                np.log(self.cg_weights).astype(np.float32)
                for _ in range(self.num_levels - 1)]

        self.cg_loss = []
        self.cg_optimizer = []
        for level in range(self.num_levels - 1):
            # Index relevant variables based on self.goal_indices
            meta_obs0 = self.crop_to_goal(self.policy[level].obs_ph)
            meta_obs1 = self.crop_to_goal(self.policy[level].obs1_ph)
            worker_obs0 = self.crop_to_goal(self.policy[level + 1].obs_ph)
            worker_obs1 = self.crop_to_goal(self.policy[level + 1].obs1_ph)

            if self.relative_goals:
                # Relative goal formulation as per HIRO.
                goal = meta_obs0 + self.policy[level].actor_tf - meta_obs1
            else:
                # Goal is the direct output from the meta policy in this case.
                goal = self.policy[level].actor_tf

            # Concatenate the output from the manager with the worker policy.
            obs_shape = self.policy[level + 1].ob_space.shape[0]
            obs = tf.concat(
                [self.policy[level + 1].obs_ph[:, :obs_shape], goal], axis=-1)

            # Create the worker policy with inputs directly from the manager.
            with tf.compat.v1.variable_scope("level_{}/model".format(level+1)):
                worker_with_meta_obs = self.policy[level + 1].make_critic(
                    obs,
                    self.policy[level + 1].action_ph,
                    reuse=True, scope="qf_0")

            # Create a tensorflow operation that mimics the reward function
            # that is used to provide feedback to the worker.
            if self.intrinsic_reward_type.startswith("scaled"):
                # Scale the observations/goals by the action space of the
                # upper-level policy if requested.
                ac_space = self.policy[level].ac_space
                scale = 0.5 * (ac_space.high - ac_space.low)
                worker_obs0 /= scale
                goal /= scale
                worker_obs1 /= scale

            if self.relative_goals:
                # Implement relative goals if requested.
                goal += worker_obs0

            if self.intrinsic_reward_type.endswith("exp_negative_distance"):
                reward_fn = tf.reduce_mean(tf.exp(-tf.reduce_sum(
                    tf.square(worker_obs0 + goal - worker_obs1), axis=1)))
            elif self.intrinsic_reward_type.endswith("negative_distance"):
                reward_fn = -tf.compat.v1.losses.mean_squared_error(
                    worker_obs0 + goal, worker_obs1)
            else:
                raise ValueError("Unknown intrinsic reward type: {}".format(
                    self.intrinsic_reward_type))

            # Scale by the worker reward scale.
            reward_fn *= self.intrinsic_reward_scale

            # Compute the worker loss with respect to the meta policy actions.
            cg_loss = - (tf.reduce_mean(worker_with_meta_obs) + reward_fn)
            self.cg_loss.append(cg_loss)

            # Create the optimizer object.
            optimizer = tf.compat.v1.train.AdamOptimizer(
                self.policy[level].actor_lr)
            self.cg_optimizer.append(optimizer.minimize(
                self.policy[level].actor_loss
                + tf.exp(self.cg_weights[level]) * cg_loss,
                var_list=get_trainable_vars(
                    "level_{}/model/pi/".format(level)),
            ))

            if self.cg_delta is not None:
                cg_weights_loss = \
                    tf.reduce_mean(
                        tf.exp(self.cg_weights[level]) * tf.stop_gradient(
                            worker_with_meta_obs + reward_fn - self.cg_delta
                        )
                    )
                optimizer = tf.compat.v1.train.AdamOptimizer(self.actor_lr)
                self.cg_weights_optimizer = optimizer.minimize(
                    cg_weights_loss,
                    var_list=[self.cg_weights[level]])

                # Add to tensorboard.
                tf.compat.v1.summary.scalar(
                    "level_{}/cg_weights_log".format(level),
                    self.cg_weights[level])
                tf.compat.v1.summary.scalar(
                    "level_{}/cg_weights".format(level),
                    tf.exp(self.cg_weights[level]))
                tf.compat.v1.summary.scalar(
                    "level_{}/cg_weights_loss".format(level),
                    cg_weights_loss)
                tf.compat.v1.summary.scalar(
                    "level_{}/worker_with_meta_obs".format(level),
                    tf.reduce_mean(worker_with_meta_obs))

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
        """
        self._n_train_steps += 1

        # Reshape to match previous behavior and placeholder shape.
        rewards[level_num] = rewards[level_num].reshape(-1, 1)
        terminals1[level_num] = terminals1[level_num].reshape(-1, 1)

        # Update operations for the critic networks.
        step_ops = [
            self.policy[level_num].critic_optimizer[0],
            self.policy[level_num].critic_optimizer[1],
        ]

        feed_dict = {
            self.policy[level_num].obs_ph: obs0[level_num],
            self.policy[level_num].action_ph: actions[level_num],
            self.policy[level_num].rew_ph: rewards[level_num],
            self.policy[level_num].obs1_ph: obs1[level_num],
            self.policy[level_num].terminals1: terminals1[level_num],
            self.policy[level_num + 1].action_ph: actions[level_num + 1],
            self.policy[level_num + 1].obs_ph: obs0[level_num + 1],
        }

        # Update the cg_weights terms.
        if self._n_train_steps > 1000 and self.cg_delta is not None:
            step_ops += [self.cg_weights_optimizer]

        if update_actor:
            # Actor updates and target soft update operation.
            step_ops += [
                self.cg_optimizer[level_num],  # This is what's replaced.
                self.policy[level_num].target_soft_updates,
            ]

            feed_dict.update({
                self.policy[level_num + 1].obs_ph: obs0[level_num + 1],
                self.policy[level_num + 1].action_ph: actions[level_num + 1],
                self.policy[level_num + 1].obs1_ph: obs1[level_num + 1],
            })

        # Perform the update operations.
        self.sess.run(step_ops, feed_dict=feed_dict)
