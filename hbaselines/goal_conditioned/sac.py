"""SAC-compatible goal-conditioned hierarchical policy."""
import numpy as np

from hbaselines.goal_conditioned.base import GoalConditionedPolicy as \
    BaseGoalConditionedPolicy
from hbaselines.fcnet.sac import FeedForwardPolicy


class GoalConditionedPolicy(BaseGoalConditionedPolicy):
    """SAC-compatible goal-conditioned hierarchical policy.

    TODO: description of off-policy corrections

    TODO: description of connected gradients

    Descriptions of the base goal-conditioned policy can be found in
    hbaselines/goal_conditioned/base.py.
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
                 target_entropy,
                 meta_period,
                 worker_reward_scale,
                 relative_goals,
                 off_policy_corrections,
                 hindsight,
                 subgoal_testing_rate,
                 connected_gradients,
                 use_fingerprints,
                 fingerprint_range,
                 centralized_value_functions,
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
        target_entropy : float
            target entropy used when learning the entropy coefficient. If set
            to None, a heuristic value is used.
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
        use_fingerprints : bool
            specifies whether to add a time-dependent fingerprint to the
            observations
        fingerprint_range : (list of float, list of float)
            the low and high values for each fingerprint element, if they are
            being used
        centralized_value_functions : bool
            specifies whether to use centralized value functions for the
            Manager and Worker critic functions
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
            use_huber=use_huber,
            meta_period=meta_period,
            worker_reward_scale=worker_reward_scale,
            relative_goals=relative_goals,
            off_policy_corrections=off_policy_corrections,
            hindsight=hindsight,
            subgoal_testing_rate=subgoal_testing_rate,
            connected_gradients=connected_gradients,
            cg_weights=cg_weights,
            use_fingerprints=use_fingerprints,
            fingerprint_range=fingerprint_range,
            centralized_value_functions=centralized_value_functions,
            env_name=env_name,
            meta_policy=FeedForwardPolicy,
            worker_policy=FeedForwardPolicy,
            additional_params=dict(
                target_entropy=target_entropy,
            ),
        )

    # ======================================================================= #
    #                       Auxiliary methods for HIRO                        #
    # ======================================================================= #

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

            # Create repeated representations of each worker action for each
            # candidate goal.
            tiled_worker_actions_per_sample = np.tile(
                worker_actions_per_sample, (num_samples, 1))

            # Create repeated representations of each worker observation for
            # each candidate goal. The indexing of worker_obses_per_sample is
            # meant to do the following:
            #  1. We remove the last observation since it does not correspond
            #     to any action for the current meta-period.
            #  2. Unlike the TD3 implementation, we keep the trailing context
            #     (goal) terms since they are needed to compute the log-prob
            #     of a given action when feeding to logp_action.
            tiled_worker_obses_per_sample = np.tile(
                worker_obses_per_sample[:-1, :], (num_samples, 1))

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
                    np.tile(goal_diff, (num_samples, 1))[:, :goal_dim]

            # Compute the log-probability of each action using the logp_action
            # attribute of the SAC Worker policy.
            normalized_error = self.sess.run(
                self.worker.logp_action,
                feed_dict={
                    self.worker.obs_ph: tiled_worker_obses_per_sample,
                    self.worker.action_ph: tiled_worker_actions_per_sample,
                }
            )

            # Sum the different normalized errors to get the fitness of each
            # candidate goal.
            for j in range(num_samples):
                fitness_per_sample[j] = np.sum(
                    normalized_error[j * meta_period: (j+1) * meta_period])

            fitness.append(fitness_per_sample)

        return np.array(fitness)

    # ======================================================================= #
    #                      Auxiliary methods for HRL-CG                       #
    # ======================================================================= #

    def _setup_connected_gradients(self):
        """Create the updated manager optimization with connected gradients."""
        raise NotImplementedError  # TODO

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
        raise NotImplementedError  # TODO
