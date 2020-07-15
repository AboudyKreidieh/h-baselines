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


# =========================================================================== #
#                   Policy parameters for FeedForwardPolicy                   #
# =========================================================================== #

FEEDFORWARD_PARAMS = dict(
    # the learning rate
    learning_rate=3e-4,
    # entropy coefficient for the loss calculation
    ent_coef=0.01,
    # value function coefficient for the loss calculation
    vf_coef=0.5,
    # the maximum value for the gradient clipping
    max_grad_norm=0.5,
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
    policy : type [ hbaselines.base_policies.Policy ]
        the policy model to use
    env_name : str
        name of the environment. Affects the action bounds of the higher-level
        policies
    eval_env : gym.Env or list of gym.Env
        the environment(s) to evaluate from
    n_steps : int
        number of steps to sample in between training operations
    n_minibatches : int
        number of training minibatches per update
    n_opt_epochs : int
        number of epoch when optimizing the surrogate
    n_eval_episodes : int
        the number of evaluation episodes
    gamma : float
        the discount factor
    lam : float
        factor for trade-off of bias vs variance for Generalized Advantage
        Estimator
    reward_scale : float
        the value the reward should be scaled by
    render : bool
        enable rendering of the training environment
    render_eval : bool
        enable rendering of the evaluation environment
    eval_deterministic : bool
        if set to True, the policy provides deterministic actions to the
        evaluation environment. Otherwise, stochastic or noisy actions are
        returned.
    num_envs : int
        number of environments used to run simulations in parallel. Each
        environment is run on a separate CPUS and uses the same policy as the
        rest. Must be less than or equal to nb_rollout_steps.
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    policy_kwargs : dict
        policy-specific hyperparameters
    sampler : list of hbaselines.utils.sampler.Sampler
        the training environment sampler object. One environment is provided
        for each CPU
    obs : list of array_like or list of dict < str, array_like >
        the most recent training observation. If you are using a multi-agent
        environment, this will be a dictionary of observations for each agent,
        indexed by the agent ID. One element for each environment.
    all_obs : list of array_like or list of None
        additional information, used by MADDPG variants of the multi-agent
        policy to pass full-state information. One element for each environment
    ac_space : gym.spaces.*
        the action space of the training environment
    ob_space : gym.spaces.*
        the observation space of the training environment
    co_space : gym.spaces.*
        the context space of the training environment (i.e. the same of the
        desired environmental goal)
    graph : tf.Graph
        the current tensorflow graph
    policy_tf : hbaselines.base_policies.Policy
        the policy object
    sess : tf.compat.v1.Session
        the current tensorflow session
    summary : tf.Summary
        tensorboard summary object
    episode_step : list of int
        the number of steps since the most recent rollout began. One for each
        environment.
    episodes : int
        the total number of rollouts performed since training began
    total_steps : int
        the total number of steps that have been executed since training began
    epoch_episode_steps : list of int
        a list of rollout lengths from the most recent training iterations
    epoch_episode_rewards : list of float
        a list of cumulative rollout rewards from the most recent training
        iterations
    epoch_episodes : int
        the total number of rollouts performed since the most recent training
        iteration began
    epoch : int
        the total number of training iterations
    episode_rew_history : list of float
        the cumulative return from the last 100 training episodes
    episode_reward : list of float
        the cumulative reward since the most reward began. One for each
        environment.
    rew_ph : tf.compat.v1.placeholder
        a placeholder for the average training return for the last epoch. Used
        for logging purposes.
    rew_history_ph : tf.compat.v1.placeholder
        a placeholder for the average training return for the last 100
        episodes. Used for logging purposes.
    eval_rew_ph : tf.compat.v1.placeholder
        placeholder for the average evaluation return from the last time
        evaluations occurred. Used for logging purposes.
    eval_success_ph : tf.compat.v1.placeholder
        placeholder for the average evaluation success rate from the last time
        evaluations occurred. Used for logging purposes.
    saver : tf.compat.v1.train.Saver
        tensorflow saver object
    trainable_vars : list of str
        the trainable variables
    """

    def __init__(self,
                 policy,
                 env,
                 eval_env=None,
                 n_steps=128,
                 n_eval_episodes=50,
                 n_minibatches=4,
                 n_opt_epochs=4,
                 gamma=0.99,
                 lam=0.95,
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
        policy : type [ hbaselines.base_policies.Policy ]
            the policy model to use
        env : gym.Env or str
            the environment to learn from (if registered in Gym, can be str)
        eval_env : gym.Env or str
            the environment to evaluate from (if registered in Gym, can be str)
        n_steps : int
            number of steps to sample in between training operations
        n_minibatches : int
            number of training minibatches per update
        n_opt_epochs : int
            number of epoch when optimizing the surrogate
        n_eval_episodes : int
            the number of evaluation episodes
        gamma : float
            discount factor
        lam : float
            factor for trade-off of bias vs variance for Generalized Advantage
            Estimator
        reward_scale : float
            the value the reward should be scaled by
        render : bool
            enable rendering of the training environment
        render_eval : bool
            enable rendering of the evaluation environment
        eval_deterministic : bool
            if set to True, the policy provides deterministic actions to the
            evaluation environment. Otherwise, stochastic or noisy actions are
            returned.
        num_envs : int
            number of environments used to run simulations in parallel. Each
            environment is run on a separate CPUS and uses the same policy as
            the rest. Must be less than or equal to nb_rollout_steps.
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        policy_kwargs : dict
            policy-specific hyperparameters
        _init_setup_model : bool
            Whether or not to build the network at the creation of the instance
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
        self.gamma = gamma
        self.lam = lam
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

        with self.sess.as_default(), self.graph.as_default():
            n_updates = total_steps // self.n_steps

            for update in range(1, n_updates + 1):
                # Reset epoch-specific variables.
                self.epoch_episodes = 0
                self.epoch_episode_steps = []
                self.epoch_episode_rewards = []

                # Perform rollouts.
                rollout = self._collect_samples(total_steps)

                # Train.
                self._train(**rollout)

                # Log statistics.
                self._log_training(train_filepath, start_time)

                # Run and store summary.
                if writer is not None:
                    td_map = self.policy_tf.get_td_map(
                        obs=rollout["mb_obs"],
                        context=None if rollout["mb_contexts"][0] is None
                        else rollout["mb_contexts"],
                        returns=rollout["mb_returns"],
                        actions=rollout["mb_actions"],
                        values=rollout["mb_values"],
                        neglogpacs=rollout["mb_neglogpacs"],
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

    def _policy(self, obs, context, apply_noise=True, env_num=0):
        """Get the actions and critic output, from a given observation.

        Parameters
        ----------
        obs : array_like
            the observation
        context : array_like or None
            the contextual term. Set to None if no context is provided by the
            environment.
        apply_noise : bool
            enable the noise
        env_num : int
            the environment number. Used to handle situations when multiple
            parallel environments are being used.

        Returns
        -------
        list of float
            the action value
        """
        # Reshape the observation to match the input structure of the policy.
        if isinstance(obs, dict):
            # In multi-agent environments, observations come in dict form
            for key in obs.keys():
                # Shared policies with have one observation space, while
                # independent policies have a different observation space based
                # on their agent ID.
                if isinstance(self.ob_space, dict):
                    ob_shape = self.ob_space[key].shape
                else:
                    ob_shape = self.ob_space.shape
                obs[key] = np.array(obs[key]).reshape((-1,) + ob_shape)
        else:
            obs = np.array(obs).reshape((-1,) + self.ob_space.shape)

        action, value, neglogpac = self.policy_tf.get_action(
            obs, context,
            random_actions=False,
            apply_noise=apply_noise,
            env_num=env_num)

        # Flatten the actions. Dictionaries correspond to multi-agent policies.
        if isinstance(action, dict):
            action = {key: action[key].flatten() for key in action.keys()}
        else:
            action = action.flatten()

        return action, value, neglogpac

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
        mb_rewards = [[] for _ in range(self.num_envs)]
        mb_obs = [[] for _ in range(self.num_envs)]
        mb_contexts = [[] for _ in range(self.num_envs)]
        mb_actions = [[] for _ in range(self.num_envs)]
        mb_values = [[] for _ in range(self.num_envs)]
        mb_neglogpacs = [[] for _ in range(self.num_envs)]
        mb_dones = [[] for _ in range(self.num_envs)]
        mb_all_obs = [[] for _ in range(self.num_envs)]

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
            output = [self._policy(
                obs=self.obs[env_num],
                context=context[env_num],
                apply_noise=True,
                env_num=env_num,
            ) for env_num in range(n_steps)]
            action = [o[0] for o in output]
            values = [o[1] for o in output]
            neglogpacs = [o[2] for o in output]

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
                reward = ret_i["reward"]
                obs = ret_i["obs"]
                done = ret_i["done"]
                all_obs = ret_i["all_obs"]

                # Store the new data.
                mb_rewards[num].append(ret_i["reward"])
                mb_obs[num].append([self.obs[num]])
                mb_contexts[num].append(ret_i["context"])
                mb_actions[num].append([ret_i["action"]])
                mb_values[num].append(values[num])
                mb_neglogpacs[num].append(neglogpacs[num])
                mb_dones[num].append(ret_i["done"])
                mb_all_obs[num].append(ret_i["all_obs"])

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

        # Compute the bootstrapped/discounted returns.
        mb_returns = []
        for num in range(self.num_envs):
            mb_obs[num] = np.concatenate(mb_obs[num], axis=0)
            mb_rewards[num] = np.asarray(mb_rewards[num])
            mb_actions[num] = np.concatenate(mb_actions[num], axis=0)
            mb_values[num] = np.concatenate(mb_values[num], axis=0)
            mb_neglogpacs[num] = np.concatenate(mb_neglogpacs[num], axis=0)
            mb_dones[num] = np.asarray(mb_dones[num])

            mb_returns.append(self._gae_returns(
                mb_rewards=mb_rewards[num],
                mb_values=mb_values[num],
                mb_dones=mb_dones[num],
                obs=self.obs[num],
                context=context[num],
            ))

        # Concatenate the stored data.
        if self.num_envs > 1:
            mb_obs = np.concatenate(mb_rewards, axis=0)
            mb_contexts = np.concatenate(mb_contexts, axis=0)
            mb_actions = np.concatenate(mb_actions, axis=0)
            mb_values = np.concatenate(mb_values, axis=0)
            mb_neglogpacs = np.concatenate(mb_neglogpacs, axis=0)
            mb_all_obs = np.concatenate(mb_all_obs, axis=0)
            mb_returns = np.concatenate(mb_returns, axis=0)
        else:
            mb_obs = mb_obs[0]
            mb_contexts = mb_contexts[0]
            mb_actions = mb_actions[0]
            mb_values = mb_values[0]
            mb_neglogpacs = mb_neglogpacs[0]
            mb_all_obs = mb_all_obs[0]
            mb_returns = mb_returns[0]

        return {
            "mb_returns": mb_returns,
            "mb_obs": mb_obs,
            "mb_contexts": mb_contexts,
            "mb_actions": mb_actions,
            "mb_values": mb_values,
            "mb_neglogpacs": mb_neglogpacs,
            "mb_all_obs": mb_all_obs,
        }

    def _gae_returns(self, mb_rewards, mb_values, mb_dones, obs, context):
        """Compute the bootstrapped/discounted returns.

        Parameters
        ----------
        mb_rewards : array_like
            a minibatch of rewards from a given environment
        mb_values : array_like
            a minibatch of values computed by the policy from a given
            environment
        mb_dones : array_like
            a minibatch of done masks from a given environment
        obs : array_like
            the current observation within the environment
        context : array_like or None
            the current contextual term within the environment

        Returns
        -------
        array_like
            GAE-style expected discounted returns.
        """
        # Compute the last estimated value.
        last_values = self.policy_tf.value([obs], context)

        # Discount/bootstrap off value fn.
        mb_advs = np.zeros_like(mb_rewards)
        mb_vactual = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextnonterminal = 1.0 - mb_dones[-1]
                nextvalues = last_values
                mb_vactual[t] = mb_rewards[t]
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
                mb_vactual[t] = mb_rewards[t] \
                    + self.gamma * nextnonterminal * nextvalues
            delta = mb_rewards[t] \
                + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta \
                + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return mb_returns

    def _train(self,
               mb_obs,
               mb_contexts,
               mb_returns,
               mb_actions,
               mb_values,
               mb_neglogpacs,
               mb_all_obs):
        """Perform the training operation.

        Parameters
        ----------
        mb_obs : array_like
            a minibatch of observations
        mb_contexts : array_like
            a minibatch of contextual terms
        mb_returns : array_like
            a minibatch of contextual expected discounted retuns
        mb_actions : array_like
            a minibatch of actions
        mb_values : array_like
            a minibatch of estimated values by the policy
        mb_neglogpacs : array_like
            a minibatch of the negative log-likelihood of performed actions
        mb_all_obs : array_like or None
            a minibatch of full-state observations. Used by the multi-agent
            MADDPG policies.

        Returns
        -------
        TODO
            TODO
        """
        batch_size = self.n_steps // self.n_minibatches

        inds = np.arange(self.n_steps)
        for epoch_num in range(self.n_opt_epochs):
            np.random.shuffle(inds)
            for start in range(0, self.n_steps, batch_size):
                end = start + batch_size
                mbinds = inds[start:end]
                self.policy_tf.update(
                    obs=mb_obs[mbinds],
                    context=None if mb_contexts[0] is None
                    else mb_contexts[mbinds],
                    returns=mb_returns[mbinds],
                    actions=mb_actions[mbinds],
                    values=mb_values[mbinds],
                    neglogpacs=mb_neglogpacs[mbinds],
                )

    def _log_training(self, file_path, start_time):
        """Log training statistics.

        Parameters
        ----------
        file_path : str
            the list of cumulative rewards from every episode in the evaluation
            phase
        start_time : float
            the time when training began. This is used to print the total
            training time.
        """
        # Log statistics.
        duration = time.time() - start_time

        combined_stats = {
            # Rollout statistics.
            'rollout/episodes': self.epoch_episodes,
            'rollout/episode_steps': np.mean(self.epoch_episode_steps),
            'rollout/return': np.mean(self.epoch_episode_rewards),
            'rollout/return_history': np.mean(self.episode_rew_history),

            # Total statistics.
            'total/epochs': self.epoch + 1,
            'total/steps': self.total_steps,
            'total/duration': duration,
            'total/steps_per_second': self.total_steps / duration,
            'total/episodes': self.episodes,
        }

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
