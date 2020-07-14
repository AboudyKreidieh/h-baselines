"""Script algorithm contain the base on-policy RL algorithm class.

Supported algorithms through this class:

* Proximal Policy Optimization (PPO): see https://arxiv.org/pdf/1707.06347.pdf

This algorithm class also contains modifications to support contextual
environments as well as multi-agent and hierarchical policies.
"""
import ray
import os
import time
import csv
import random
import numpy as np
import tensorflow as tf
import math
from collections import deque
from copy import deepcopy
from gym.spaces import Box

from hbaselines.algorithms.utils import is_feedforward_policy
from hbaselines.algorithms.utils import is_goal_conditioned_policy
from hbaselines.algorithms.utils import is_multiagent_policy
from hbaselines.algorithms.utils import add_fingerprint
from hbaselines.algorithms.utils import get_obs
from hbaselines.utils.tf_util import make_session
from hbaselines.utils.misc import ensure_dir
from hbaselines.utils.env_util import create_env
from hbaselines.utils.misc import explained_variance

from stable_baselines.common.runners import AbstractEnvRunner


# =========================================================================== #
#                   Policy parameters for FeedForwardPolicy                   #
# =========================================================================== #

FEEDFORWARD_PARAMS = dict(
    # discount factor
    gamma=0.99,
    # the learning rate
    learning_rate=3e-4,
    # entropy coefficient for the loss calculation
    ent_coef=0.01,
    # value function coefficient for the loss calculation
    vf_coef=0.5,
    # the maximum value for the gradient clipping
    max_grad_norm=0.5,
    # factor for trade-off of bias vs variance for Generalized Advantage
    # Estimator
    lam=0.95,
    # clipping parameter, it can be a function
    cliprange=0.2,
    # clipping parameter for the value function, it can be a function. This is
    # a parameter specific to the OpenAI implementation. If None is passed
    # (default), then `cliprange` (that is used for the policy) will be used.
    # IMPORTANT: this clipping depends on the reward scaling. To deactivate
    # value function clipping (and recover the original PPO implementation),
    # you have to pass a negative value (e.g. -1).
    cliprange_vf=None,
    # enable layer normalisation
    layer_norm=False,
    # the size of the neural network for the policy
    layers=[64, 64],
    # the activation function to use in the neural network
    act_fun=tf.nn.relu,
    # specifies whether to use the huber distance function as the loss for the
    # critic. If set to False, the mean-squared error metric is used instead
    use_huber=False,
)


# =========================================================================== #
#                 Policy parameters for GoalConditionedPolicy                 #
# =========================================================================== #

GOAL_CONDITIONED_PARAMS = FEEDFORWARD_PARAMS.copy()
GOAL_CONDITIONED_PARAMS.update(dict(
    # number of levels within the hierarchy. Must be greater than 1. Two levels
    # correspond to a Manager/Worker paradigm.
    num_levels=2,
    # meta-policy action period
    meta_period=10,
    # the reward function to be used by lower-level policies. See the base
    # goal-conditioned policy for a description.
    intrinsic_reward_type="negative_distance",
    # the value that the intrinsic reward should be scaled by
    intrinsic_reward_scale=1,
    # specifies whether the goal issued by the higher-level policies is meant
    # to be a relative or absolute goal, i.e. specific state or change in state
    relative_goals=False,
    # whether to use off-policy corrections during the update procedure. See:
    # https://arxiv.org/abs/1805.08296
    off_policy_corrections=False,
    # whether to include hindsight action and goal transitions in the replay
    # buffer. See: https://arxiv.org/abs/1712.00948
    hindsight=False,
    # rate at which the original (non-hindsight) sample is stored in the
    # replay buffer as well. Used only if `hindsight` is set to True.
    subgoal_testing_rate=0.3,
    # whether to use the connected gradient update actor update procedure to
    # the higher-level policies. See: https://arxiv.org/abs/1912.02368v1
    connected_gradients=False,
    # weights for the gradients of the loss of the lower-level policies with
    # respect to the parameters of the higher-level policies. Only used if
    # `connected_gradients` is set to True.
    cg_weights=0.0005,
))


# =========================================================================== #
#                Policy parameters for MultiActorCriticPolicy                 #
# =========================================================================== #

MULTI_FEEDFORWARD_PARAMS = FEEDFORWARD_PARAMS.copy()
MULTI_FEEDFORWARD_PARAMS.update(dict(
    # whether to use a shared policy for all agents
    shared=False,
    # whether to use an algorithm-specific variant of the MADDPG algorithm
    maddpg=False,
))


class OnPolicyRLAlgorithm(object):
    """On-policy RL algorithm class.

    Supports the training of PPO policies.

    Attributes
    ----------
    policy : ActorCriticPolicy or str
        The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    env : Gym environment or str
        The environment to learn from (if registered in Gym, can be str)
    n_steps : int
        The number of steps to run for each environment per update (i.e. batch
        size is n_steps * n_env where n_env is number of environment copies
        running in parallel)
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    _init_setup_model : bool
        Whether or not to build the network at the creation of the instance
    policy_kwargs : dict
        additional arguments to be passed to the policy on creation
    """
    def __init__(self,
                 policy,
                 env,
                 eval_env=None,
                 n_steps=128,
                 n_eval_episodes=50,
                 n_minibatches=4,
                 n_opt_epochs=4,  # TODO: remove
                 meta_update_freq=10,
                 reward_scale=1.,
                 render=False,
                 render_eval=False,
                 eval_deterministic=True,
                 num_envs=1,
                 verbose=0,
                 policy_kwargs=None,
                 _init_setup_model=True):
        """Instantiate the algorithm object.

        Parameters
        ----------
        policy : TODO
            TODO
        env : TODO
            TODO
        eval_env : TODO
            TODO
        n_steps : TODO
            TODO
        n_minibatches : int
            number of training minibatches per update
        n_opt_epochs : int
            number of epoch when optimizing the surrogate
        n_eval_episodes : TODO
            TODO
        meta_update_freq : TODO
            TODO
        reward_scale : TODO
            TODO
        render : TODO
            TODO
        render_eval : TODO
            TODO
        eval_deterministic : TODO
            TODO
        num_envs : TODO
            TODO
        verbose : TODO
            TODO
        _init_setup_model : TODO
            TODO
        policy_kwargs : TODO
            TODO
        """
        shared = False if policy_kwargs is None else \
            policy_kwargs.get("shared", False)
        maddpg = False if policy_kwargs is None else \
            policy_kwargs.get("maddpg", False)

        # Run assertions.
        assert num_envs <= n_steps, \
            "num_envs must be less than or equal to nb_rollout_steps"

        # Instantiate the ray instance.
        if num_envs > 1:
            ray.init(num_cpus=num_envs+1, ignore_reinit_error=True)

        self.policy = policy
        self.env_name = deepcopy(env) if isinstance(env, str) \
            else env.__str__()
        self.eval_env, _ = create_env(
            eval_env, render_eval, shared, maddpg, evaluate=True)
        self.n_steps = n_steps
        self.n_minibatches = n_minibatches
        self.n_opt_epochs = n_opt_epochs
        self.n_eval_episodes = n_eval_episodes
        self.n_eval_episodes = n_eval_episodes
        self.meta_update_freq = meta_update_freq
        self.reward_scale = reward_scale
        self.render = render
        self.render_eval = render_eval
        self.eval_deterministic = eval_deterministic
        self.num_envs = num_envs
        self.verbose = verbose
        self.policy_kwargs = {'verbose': verbose}

        assert self.n_steps % self.n_minibatches == 0, \
            "The number of minibatches (`n_minibatches`) is not a factor of " \
            "the total number of samples collected per rollout (`n_steps`), " \
            "some samples won't be used."

        self.initial_state = None  # TODO: remove

        # Create the environment and collect the initial observations.
        self.sampler, self.obs, self.all_obs = self.setup_sampler(
            env, render, shared, maddpg)

        # Collect the spaces of the environments.
        self.ac_space, self.ob_space, self.co_space, all_ob_space = \
            self.get_spaces()

        # Add the default policy kwargs to the policy_kwargs term.
        if is_feedforward_policy(policy):
            self.policy_kwargs.update(FEEDFORWARD_PARAMS.copy())

        if is_goal_conditioned_policy(policy):
            self.policy_kwargs.update(GOAL_CONDITIONED_PARAMS.copy())
            self.policy_kwargs['env_name'] = self.env_name.__str__()
            self.policy_kwargs['num_envs'] = num_envs

        if is_multiagent_policy(policy):
            self.policy_kwargs.update(MULTI_FEEDFORWARD_PARAMS.copy())
            self.policy_kwargs["all_ob_space"] = all_ob_space

        self.policy_kwargs.update(policy_kwargs or {})

        # init
        self.graph = None
        self.policy_tf = None
        self.sess = None
        self.summary = None
        self.episode_step = [0 for _ in range(num_envs)]
        self.episodes = 0
        self.total_steps = 0
        self.epoch_episode_steps = []
        self.epoch_episode_rewards = []
        self.epoch_episodes = 0
        self.epoch = 0
        self.episode_rew_history = deque(maxlen=100)
        self.episode_reward = [0 for _ in range(num_envs)]
        self.rew_ph = None
        self.rew_history_ph = None
        self.eval_rew_ph = None
        self.eval_success_ph = None
        self.saver = None

        # TODO: remove
        self.value = None
        self.step = None
        self.env = env
        self.num_timesteps = 0

        if self.policy_kwargs.get("use_fingerprints", False):
            # Append the fingerprint dimension to the observation dimension.
            fingerprint_range = self.policy_kwargs["fingerprint_range"]
            low = np.concatenate((self.ob_space.low, fingerprint_range[0]))
            high = np.concatenate((self.ob_space.high, fingerprint_range[1]))
            self.ob_space = Box(low, high, dtype=np.float32)

            # Add the fingerprint term to the first observation.
            self.obs = [add_fingerprint(obs, 0, 1, True) for obs in self.obs]

        # Create the model variables and operations.
        if _init_setup_model:
            self.trainable_vars = self.setup_model()

        # TODO: remove
        self.runner = Runner(
            env=self.env,
            model=self,
            n_steps=self.n_steps,
            gamma=self.policy_kwargs["gamma"],
            lam=self.policy_kwargs["lam"],
        )

    def setup_sampler(self, env, render, shared, maddpg):
        """Create the environment and collect the initial observations.

        Parameters
        ----------
        env : str
            the name of the environment
        render : bool
            whether to render the environment
        shared : bool
            specifies whether agents in an environment are meant to share
            policies. This is solely used by multi-agent Flow environments.
        maddpg : bool
            whether to use an environment variant that is compatible with the
            MADDPG algorithm

        Returns
        -------
        list of Sampler or list of RaySampler
            the sampler objects
        list of array_like or list of dict < str, array_like >
            the initial observation. If the environment is multi-agent, this
            will be a dictionary of observations for each agent, indexed by the
            agent ID. One element for each environment.
        list of array_like or list of None
            additional information, used by MADDPG variants of the multi-agent
            policy to pass full-state information. One element for each
            environment
        """
        if self.num_envs > 1:
            from hbaselines.utils.sampler import RaySampler
            sampler = [
                RaySampler.remote(
                    env_name=env,
                    render=render,
                    shared=shared,
                    maddpg=maddpg,
                    env_num=env_num,
                    evaluate=False,
                )
                for env_num in range(self.num_envs)
            ]
            ob = ray.get([s.get_init_obs.remote() for s in sampler])
        else:
            from hbaselines.utils.sampler import Sampler
            sampler = [
                Sampler(
                    env_name=env,
                    render=render,
                    shared=shared,
                    maddpg=maddpg,
                    env_num=0,
                    evaluate=False,
                )
            ]
            ob = [s.get_init_obs() for s in sampler]

        # Separate the observation and full-state observation.
        obs = [get_obs(o)[0] for o in ob]
        all_obs = [get_obs(o)[1] for o in ob]

        return sampler, obs, all_obs

    def get_spaces(self):
        """Collect the spaces of the environments.

        Returns
        -------
        gym.spaces.*
            the action space of the training environment
        gym.spaces.*
            the observation space of the training environment
        gym.spaces.* or None
            the context space of the training environment (i.e. the same of the
            desired environmental goal)
        gym.spaces.* or None
            the full-state observation space of the training environment
        """
        sampler = self.sampler[0]

        if self.num_envs > 1:
            ac_space = ray.get(sampler.action_space.remote())
            ob_space = ray.get(sampler.observation_space.remote())
            co_space = ray.get(sampler.context_space.remote())
            all_ob_space = ray.get(sampler.all_observation_space.remote())
        else:
            ac_space = sampler.action_space()
            ob_space = sampler.observation_space()
            co_space = sampler.context_space()
            all_ob_space = sampler.all_observation_space()

        return ac_space, ob_space, co_space, all_ob_space

    def setup_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Create the tensorflow session.
            self.sess = make_session(num_cpu=3, graph=self.graph)

            # Create the policy.
            self.policy_tf = self.policy(
                sess=self.sess,
                ob_space=self.ob_space,
                ac_space=self.ac_space,
                co_space=self.co_space,
                reuse=False,
                **self.policy_kwargs
            )

            # for tensorboard logging
            with tf.compat.v1.variable_scope("Train"):
                self.rew_ph = tf.compat.v1.placeholder(tf.float32)
                self.rew_history_ph = tf.compat.v1.placeholder(tf.float32)

            # Add tensorboard scalars for the return, return history, and
            # success rate.
            tf.compat.v1.summary.scalar("Train/return", self.rew_ph)
            tf.compat.v1.summary.scalar("Train/return_history",
                                        self.rew_history_ph)

            # Create the tensorboard summary.
            self.summary = tf.compat.v1.summary.merge_all()

            self.model = self.policy_tf  # TODO: remove
            self.step = self.policy_tf.step  # TODO: remove
            self.value = self.policy_tf.value  # TODO: remove

            # Initialize the model parameters and optimizers.
            with self.sess.as_default():
                self.sess.run(tf.compat.v1.global_variables_initializer())
                self.policy_tf.initialize()

            return tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

    def learn(self,
              total_steps,
              log_dir=None,
              seed=None,
              eval_interval=50000,
              save_interval=50000):
        """Perform the complete training operation.

        Parameters
        ----------
        total_steps : int
            the total number of samples to train on
        log_dir : str
            the directory where the training and evaluation statistics, as well
            as the tensorboard log, should be stored
        seed : int or None
            the initial seed for training, if None: keep current seed
        eval_interval : int
            number of simulation steps in the training environment before an
            evaluation is performed
        save_interval : int
            number of simulation steps in the training environment before the
            model is saved
        """
        # Create a saver object.
        self.saver = tf.compat.v1.train.Saver(
            self.trainable_vars,
            max_to_keep=total_steps // save_interval)

        # Make sure that the log directory exists, and if not, make it.
        ensure_dir(log_dir)
        ensure_dir(os.path.join(log_dir, "checkpoints"))

        # Create a tensorboard object for logging.
        save_path = os.path.join(log_dir, "tb_log")
        writer = tf.compat.v1.summary.FileWriter(save_path)

        # file path for training and evaluation results
        train_filepath = os.path.join(log_dir, "train.csv")
        eval_filepath = os.path.join(log_dir, "eval.csv")

        # Setup the seed value.
        random.seed(seed)
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

        if self.verbose >= 2:
            print('Using agent with the following configuration:')
            print(str(self.__dict__.items()))

        eval_steps_incr = 0
        save_steps_incr = 0
        start_time = time.time()

        rewards_buffer = deque(maxlen=100)  # TODO: remove

        with self.sess.as_default(), self.graph.as_default():
            n_updates = total_steps // self.n_steps

            for update in range(1, n_updates + 1):
                batch_size = self.n_steps // self.n_minibatches
                t_start = time.time()

                # Reset epoch-specific variables.
                self.epoch_episodes = 0
                self.epoch_episode_steps = []
                self.epoch_episode_rewards = []

                # Perform rollouts.
                # rollout = self._collect_samples(total_steps)

                # Train.
                # self._train()

                # true_reward is the reward without discount
                rollout = self.runner.run()
                # Unpack
                obs, returns, masks, actions, values, neglogpacs, states, \
                    ep_infos, true_reward, total_reward = rollout
                rewards_buffer.extend(total_reward)
                self.total_steps += self.n_steps

                mb_loss_vals = []
                inds = np.arange(self.n_steps)
                for epoch_num in range(self.n_opt_epochs):
                    np.random.shuffle(inds)
                    for start in range(0, self.n_steps, batch_size):
                        end = start + batch_size
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (
                            obs, returns, masks, actions, values, neglogpacs))
                        mb_loss_vals.append(self.model.update(*slices))

                # Log statistics.
                self._log_training(
                    train_filepath,
                    mb_loss_vals,
                    t_start,
                    values,
                    returns,
                    update,
                    rewards_buffer,
                    start_time,
                )

                # Run and store summary.
                if writer is not None:
                    td_map = self.policy_tf.get_td_map(
                        obs,
                        returns,
                        actions,
                        values,
                        neglogpacs
                    )

                    # Check if td_map is empty.
                    if not td_map:
                        break

                    td_map.update({
                        self.rew_ph: np.mean(self.epoch_episode_rewards),
                        self.rew_history_ph: np.mean(self.episode_rew_history),
                    })
                    summary = self.sess.run(self.summary, td_map)
                    writer.add_summary(summary, self.total_steps)

                # Save a checkpoint of the model.
                if (self.total_steps - save_steps_incr) >= save_interval:
                    save_steps_incr += save_interval
                    self.save(os.path.join(log_dir, "checkpoints/itr"))

                # Update the epoch count.
                self.epoch += 1

    def save(self, save_path):
        """Save the parameters of a tensorflow model.

        Parameters
        ----------
        save_path : str
            Prefix of filenames created for the checkpoint
        """
        self.saver.save(self.sess, save_path, global_step=self.total_steps)

    def load(self, load_path):
        """Load model parameters from a checkpoint.

        Parameters
        ----------
        load_path : str
            location of the checkpoint
        """
        self.saver.restore(self.sess, load_path)

    def _collect_samples(self, total_steps):
        """Perform the sample collection operation over multiple steps.

        This method calls collect_sample for a multiple steps, and attempts to
        run the operation in parallel if multiple environments are available.

        Parameters
        ----------
        total_steps : int
            the total number of samples to train on. Used by the fingerprint
            element
        """
        context_list = [[] for _ in range(self.num_envs)]
        action_list = [[] for _ in range(self.num_envs)]
        reward_list = [[] for _ in range(self.num_envs)]
        obs_list = [[] for _ in range(self.num_envs)]
        done_list = [[] for _ in range(self.num_envs)]
        all_obs_list = [[] for _ in range(self.num_envs)]

        # Loop through the sampling procedure the number of times it would
        # require to run through each environment in parallel until the number
        # of required steps have been collected.
        n_itr = math.ceil(self.n_steps / self.num_envs)
        for itr in range(n_itr):
            n_steps = self.num_envs if itr < n_itr - 1 \
                else self.n_steps - (n_itr - 1) * self.num_envs

            # Collect the most recent contextual term from every environment.
            if self.num_envs > 1:
                context = [ray.get(self.sampler[env_num].get_context.remote())
                           for env_num in range(self.num_envs)]
            else:
                context = [self.sampler[0].get_context()]

            # Predict next action. Use random actions when initializing the
            # replay buffer.
            action = [self._policy(
                obs=self.obs[env_num],
                context=context[env_num],
                apply_noise=True,
                env_num=env_num,
            ) for env_num in range(n_steps)]

            # Update the environment.
            if self.num_envs > 1:
                ret = ray.get([
                    self.sampler[env_num].collect_sample.remote(
                        action=action[env_num],
                        multiagent=is_multiagent_policy(self.policy),
                        steps=self.total_steps,
                        total_steps=total_steps,
                        use_fingerprints=self.policy_kwargs.get(
                            "use_fingerprints", False)
                    )
                    for env_num in range(n_steps)
                ])
            else:
                ret = [
                    self.sampler[0].collect_sample(
                        action=action[0],
                        multiagent=is_multiagent_policy(self.policy),
                        steps=self.total_steps,
                        total_steps=total_steps,
                        use_fingerprints=self.policy_kwargs.get(
                            "use_fingerprints", False)
                    )
                ]

            for ret_i in ret:
                num = ret_i["env_num"]
                context = ret_i["context"]
                action = ret_i["action"]
                reward = ret_i["reward"]
                obs = ret_i["obs"]
                done = ret_i["done"]
                all_obs = ret_i["all_obs"]

                # Store the new data.
                context_list[num].append(context)
                action_list[num].append(action)
                reward_list[num].append(reward)
                obs_list[num].append(obs)
                done_list[num].append(done)
                all_obs_list[num].append(all_obs)

                # Book-keeping.
                self.total_steps += 1
                self.episode_step[num] += 1
                if isinstance(reward, dict):
                    self.episode_reward[num] += sum(
                        reward[k] for k in reward.keys())
                else:
                    self.episode_reward[num] += reward

                # Update the current observation.
                self.obs[num] = (obs[1] if done else obs).copy()
                self.all_obs[num] = all_obs[1] if done else all_obs

                # Handle episode done.
                if done:
                    self.epoch_episode_rewards.append(self.episode_reward[num])
                    self.episode_rew_history.append(self.episode_reward[num])
                    self.epoch_episode_steps.append(self.episode_step[num])
                    self.episode_reward[num] = 0
                    self.episode_step[num] = 0
                    self.epoch_episodes += 1
                    self.episodes += 1

        # Concatenate the stored data.
        # TODO

        # Compute the advantages.
        # TODO

    def _train(self):
        pass

    def _log_training(self,
                      file_path,
                      mb_loss_vals,
                      t_start,
                      values,
                      returns,
                      update,
                      rewards_buffer,
                      start_time):
        loss_names = ['policy_loss', 'value_loss', 'policy_entropy',
                      'approxkl', 'clipfrac']
        loss_vals = np.mean(mb_loss_vals, axis=0)
        t_now = time.time()
        fps = int(self.n_steps / (t_now - t_start))

        explained_var = explained_variance(values, returns)

        combined_stats = {
            "serial_timesteps": update * self.n_steps,
            "n_updates": update,
            "total_timesteps": self.num_timesteps,
            "fps": fps,
            "explained_variance": float(explained_var),
            'ep_reward_mean': np.mean(rewards_buffer),
            'time_elapsed': t_start - start_time,
        }
        for (loss_val, loss_name) in zip(loss_vals, loss_names):
            combined_stats[loss_name] = loss_val

        # # Log statistics.
        # duration = time.time() - start_time
        #
        # combined_stats = {
        #     # Rollout statistics.
        #     'rollout/episodes': self.epoch_episodes,
        #     'rollout/episode_steps': np.mean(self.epoch_episode_steps),
        #     'rollout/return': np.mean(self.epoch_episode_rewards),
        #     'rollout/return_history': np.mean(self.episode_rew_history),
        #
        #     # Total statistics.
        #     'total/epochs': self.epoch + 1,
        #     'total/steps': self.total_steps,
        #     'total/duration': duration,
        #     'total/steps_per_second': self.total_steps / duration,
        #     'total/episodes': self.episodes,
        # }

        # Save combined_stats in a csv file.
        if file_path is not None:
            exists = os.path.exists(file_path)
            with open(file_path, 'a') as f:
                w = csv.DictWriter(f, fieldnames=combined_stats.keys())
                if not exists:
                    w.writeheader()
                w.writerow(combined_stats)

        # Print statistics.
        print("-" * 67)
        for key in sorted(combined_stats.keys()):
            val = combined_stats[key]
            print("| {:<30} | {:<30} |".format(key, val))
        print("-" * 67)
        print('')


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for
        Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.lam = lam
        self.gamma = gamma
        self.total_reward = 0

    def _run(self):
        """Run a learning step of the model.

        :return:
        - observations: (np.ndarray) the observations
        - rewards: (np.ndarray) the rewards
        - masks: (numpy bool) whether an episode is over or not
        - actions: (np.ndarray) the actions
        - values: (np.ndarray) the value function output
        - negative log probabilities: (np.ndarray)
        - states: (np.ndarray) the internal states of the recurrent policies
        - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch
        mb_obs = []
        mb_rewards = []
        mb_actions = []
        mb_values = []
        mb_dones = []
        mb_neglogpacs = []
        mb_states = self.states
        ep_infos = []
        total_reward = []
        for _ in range(self.n_steps):
            actions, values, neglogpacs = self.model.step(self.obs)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, Box):
                clipped_actions = np.clip(
                    actions,
                    self.env.action_space.low,
                    self.env.action_space.high)
            self.obs[:], rewards, self.dones, infos = self.env.step(
                clipped_actions)
            self.total_reward += rewards[0]
            if self.dones:
                total_reward.append(self.total_reward)
                self.total_reward = 0

            self.model.num_timesteps += 1

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
            mb_rewards.append(rewards)

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs)

        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] \
                + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta \
                + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, \
            true_reward = map(
                swap_and_flatten,
                (mb_obs, mb_returns, mb_dones, mb_actions, mb_values,
                 mb_neglogpacs, true_reward)
            )

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, \
            mb_neglogpacs, mb_states, ep_infos, true_reward, total_reward


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
