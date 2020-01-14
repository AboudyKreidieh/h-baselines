"""TD3-compatible goal-conditioned hierarchical policy."""
import tensorflow as tf
import numpy as np

from hbaselines.goal_conditioned.base import GoalConditionedPolicy as \
    BaseGoalConditionedPolicy
from hbaselines.fcnet.td3 import FeedForwardPolicy
from hbaselines.utils.tf_util import get_trainable_vars


class GoalConditionedPolicy(BaseGoalConditionedPolicy):
    """TD3-compatible goal-conditioned hierarchical policy.

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
                 hindsight,
                 connected_gradients,
                 cg_weights,
                 use_fingerprints,
                 fingerprint_range,
                 centralized_value_functions,
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
        hindsight : bool
            whether to include hindsight action transitions in the replay
            buffer. See: https://arxiv.org/abs/1712.00948
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
            connected_gradients=connected_gradients,
            cg_weights=cg_weights,
            use_fingerprints=use_fingerprints,
            fingerprint_range=fingerprint_range,
            centralized_value_functions=centralized_value_functions,
            env_name=env_name,
            meta_policy=FeedForwardPolicy,
            worker_policy=FeedForwardPolicy,
            additional_params=dict(
                noise=noise,
                target_policy_noise=target_policy_noise,
                target_noise_clip=target_noise_clip,
            ),
        )

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

    def _log_probs(self, meta_actions, worker_obses, worker_actions):
        """Calculate the log probability of the next goal by the Manager.

        Parameters
        ----------
        meta_actions : array_like
            (batch_size, m_ac_dim, num_samples) matrix of candidate Manager
            actions
        worker_obses : array_like
            (bath_size, w_obs_dim, meta_period+1) matrix of Worker observations
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
                    np.tile(goal_diff, (num_samples, 1))[:, :goal_dim]

            # Compute the actions the Worker would perform given a specific
            # observation/goal for the current instantiation of the policy.
            pred_actions = self.worker.get_action(
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
        [float, float]
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
