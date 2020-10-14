"""Script algorithm contain the base RL algorithm class.

Supported algorithms through this class:

* Twin Delayed Deep Deterministic Policy Gradient (TD3): see
  https://arxiv.org/pdf/1802.09477.pdf
* Soft Actor Critic (SAC): see https://arxiv.org/pdf/1801.01290.pdf
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

from hbaselines.algorithms.utils import is_td3_policy
from hbaselines.algorithms.utils import is_sac_policy
from hbaselines.algorithms.utils import is_ppo_policy
from hbaselines.algorithms.utils import is_feedforward_policy
from hbaselines.algorithms.utils import is_goal_conditioned_policy
from hbaselines.algorithms.utils import is_multiagent_policy
from hbaselines.algorithms.utils import get_obs
from hbaselines.utils.tf_util import make_session
from hbaselines.utils.misc import ensure_dir
from hbaselines.utils.misc import recursive_update
from hbaselines.utils.env_util import create_env


# =========================================================================== #
#                          Policy parameters for TD3                          #
# =========================================================================== #

TD3_PARAMS = dict(
    # the max number of transitions to store
    buffer_size=200000,
    # the size of the batch for learning the policy
    batch_size=128,
    # the actor learning rate
    actor_lr=3e-4,
    # the critic learning rate
    critic_lr=3e-4,
    # the soft update coefficient (keep old values, between 0 and 1)
    tau=0.005,
    # the discount rate
    gamma=0.99,
    # specifies whether to use the huber distance function as the loss for the
    # critic. If set to False, the mean-squared error metric is used instead
    use_huber=False,
    # scaling term to the range of the action space, that is subsequently used
    # as the standard deviation of Gaussian noise added to the action if
    # `apply_noise` is set to True in `get_action`
    noise=0.1,
    # standard deviation term to the noise from the output of the target actor
    # policy. See TD3 paper for more.
    target_policy_noise=0.2,
    # clipping term for the noise injected in the target actor policy
    target_noise_clip=0.5,
)


# =========================================================================== #
#                          Policy parameters for SAC                          #
# =========================================================================== #

SAC_PARAMS = dict(
    # the max number of transitions to store
    buffer_size=200000,
    # the size of the batch for learning the policy
    batch_size=128,
    # the actor learning rate
    actor_lr=3e-4,
    # the critic learning rate
    critic_lr=3e-4,
    # the soft update coefficient (keep old values, between 0 and 1)
    tau=0.005,
    # the discount rate
    gamma=0.99,
    # specifies whether to use the huber distance function as the loss for the
    # critic. If set to False, the mean-squared error metric is used instead
    use_huber=False,
    # target entropy used when learning the entropy coefficient. If set to
    # None, a heuristic value is used.
    target_entropy=None,
)


# =========================================================================== #
#                          Policy parameters for PPO                          #
# =========================================================================== #

PPO_PARAMS = dict(
    # the learning rate
    learning_rate=3e-4,
    # number of training minibatches per update
    n_minibatches=10,
    # number of training epochs per update procedure
    n_opt_epochs=10,
    # the discount factor
    gamma=0.99,
    # factor for trade-off of bias vs variance for Generalized Advantage
    # Estimator
    lam=0.95,
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
)


# =========================================================================== #
#                   Policy parameters for FeedForwardPolicy                   #
# =========================================================================== #

FEEDFORWARD_PARAMS = dict(
    # L2 regularization penalty. This is applied to the policy network.
    l2_penalty=0,
    # dictionary of model-specific parameters
    model_params=dict(
        # the type of model to use. Must be one of {"fcnet", "conv"}.
        model_type="fcnet",
        # the size of the neural network for the policy
        layers=[256, 256],
        # enable layer normalisation
        layer_norm=False,
        # the activation function to use in the neural network
        act_fun=tf.nn.relu,

        # --------------- Model parameters for "conv" models. --------------- #

        # channels of the proprioceptive state to be ignored
        ignore_flat_channels=[],
        # observation includes an image but should it be ignored
        ignore_image=False,
        # the height of the image in the observation
        image_height=32,
        # the width of the image in the observation
        image_width=32,
        # the number of channels of the image in the observation
        image_channels=3,
        # the channels of the neural network conv layers for the policy
        filters=[16, 16, 16],
        # the kernel size of the neural network conv layers for the policy
        kernel_sizes=[5, 5, 5],
        # the kernel size of the neural network conv layers for the policy
        strides=[2, 2, 2],
    )
)


# =========================================================================== #
#                 Policy parameters for GoalConditionedPolicy                 #
# =========================================================================== #

GOAL_CONDITIONED_PARAMS = recursive_update(FEEDFORWARD_PARAMS.copy(), dict(
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
    # whether to use the cooperative gradient update procedure for the
    # higher-level policies. See: https://arxiv.org/abs/1912.02368v1
    cooperative_gradients=False,
    # weights for the gradients of the loss of the lower-level policies with
    # respect to the parameters of the higher-level policies. Only used if
    # `cooperative_gradients` is set to True.
    cg_weights=0.0005,
    # specifies whether you are pre-training the lower-level policies. Actions
    # by the high-level policy are randomly sampled from its action space.
    pretrain_worker=False,
    # path to the pre-trained worker policy checkpoints
    pretrain_path=None,
    # the checkpoint number to use within the worker policy path. If set to
    # None, the most recent checkpoint is used.
    pretrain_ckpt=None,
))


# =========================================================================== #
#                Policy parameters for MultiActorCriticPolicy                 #
# =========================================================================== #

MULTIAGENT_PARAMS = recursive_update(FEEDFORWARD_PARAMS.copy(), dict(
    # whether to use a shared policy for all agents
    shared=False,
    # whether to use an algorithm-specific variant of the MADDPG algorithm
    maddpg=False,
))


class RLAlgorithm(object):
    """RL algorithm class.

    Supports the training of TD3, SAC, and PPO policies.

    Attributes
    ----------
    policy : type [ hbaselines.base_policies.Policy ]
        the policy model to use
    env_name : str
        name of the environment. Affects the action bounds of the higher-level
        policies
    sampler : list of hbaselines.utils.sampler.Sampler
        the training environment sampler object. One environment is provided
        for each CPU
    eval_env : gym.Env or list of gym.Env
        the environment(s) to evaluate from
    nb_train_steps : int
        the number of training steps
    nb_rollout_steps : int
        the number of rollout steps
    nb_eval_episodes : int
        the number of evaluation episodes
    actor_update_freq : int
        number of training steps per actor policy update step. The critic
        policy is updated every training step.
    meta_update_freq : int
        number of training steps per meta policy update step. The actor policy
        of the meta-policy is further updated at the frequency provided by the
        actor_update_freq variable. Note that this value is only relevant when
        using the GoalConditionedPolicy policy.
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
    save_replay_buffer : bool
        whether to save the data from the replay buffer, at the frequency that
        the model is saved. Only the most recent replay buffer is stored.
    num_envs : int
        number of environments used to run simulations in parallel. Each
        environment is run on a separate CPUS and uses the same policy as the
        rest. Must be less than or equal to nb_rollout_steps.
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    ac_space : gym.spaces.*
        the action space of the training environment
    ob_space : gym.spaces.*
        the observation space of the training environment
    co_space : gym.spaces.*
        the context space of the training environment (i.e. the same of the
        desired environmental goal)
    policy_kwargs : dict
        policy-specific hyperparameters
    horizon : int
        time horizon, which is used to check if an environment terminated early
        and used to compute the done mask as per TD3 implementation (see
        appendix A of their paper). If the horizon cannot be found, it is
        assumed to be 500 (default value for most gym environments).
    graph : tf.Graph
        the current tensorflow graph
    policy_tf : hbaselines.base_policies.ActorCriticPolicy
        the policy object
    sess : tf.compat.v1.Session
        the current tensorflow session
    summary : tf.Summary
        tensorboard summary object
    obs : list of array_like or list of dict < str, array_like >
        the most recent training observation. If you are using a multi-agent
        environment, this will be a dictionary of observations for each agent,
        indexed by the agent ID. One element for each environment.
    all_obs : list of array_like or list of None
        additional information, used by MADDPG variants of the multi-agent
        policy to pass full-state information. One element for each environment
    episode_step : list of int
        the number of steps since the most recent rollout began. One for each
        environment.
    episodes : int
        the total number of rollouts performed since training began
    total_steps : int
        the total number of steps that have been executed since training began
    epoch_episode_rewards : list of float
        a list of cumulative rollout rewards from the most recent training
        iterations
    epoch_episode_steps : list of int
        a list of rollout lengths from the most recent training iterations
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
    saver : tf.compat.v1.train.Saver
        tensorflow saver object
    trainable_vars : list of str
        the trainable variables
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
    """

    def __init__(self,
                 policy,
                 env,
                 eval_env=None,
                 nb_train_steps=1,
                 nb_rollout_steps=1,
                 nb_eval_episodes=50,
                 actor_update_freq=2,
                 meta_update_freq=10,
                 reward_scale=1.,
                 render=False,
                 render_eval=False,
                 eval_deterministic=True,
                 save_replay_buffer=False,
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
        nb_train_steps : int
            the number of training steps
        nb_rollout_steps : int
            the number of rollout steps
        nb_eval_episodes : int
            the number of evaluation episodes
        actor_update_freq : int
            number of training steps per actor policy update step. The critic
            policy is updated every training step.
        meta_update_freq : int
            number of training steps per meta policy update step. The actor
            policy of the meta-policy is further updated at the frequency
            provided by the actor_update_freq variable. Note that this value is
            only relevant when using the GoalConditionedPolicy policy.
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
        save_replay_buffer : bool
            whether to save the data from the replay buffer, at the frequency
            that the model is saved. Only the most recent replay buffer is
            stored.
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

        Raises
        ------
        AssertionError
            if num_envs > nb_rollout_steps
        """
        shared = False if policy_kwargs is None else \
            policy_kwargs.get("shared", False)
        maddpg = False if policy_kwargs is None else \
            policy_kwargs.get("maddpg", False)

        # Run assertions.
        assert num_envs <= nb_rollout_steps, \
            "num_envs must be less than or equal to nb_rollout_steps"

        # Include warnings if using PPO.
        if is_ppo_policy(policy):
            if actor_update_freq is not None:
                print("WARNING: actor_update_freq is not utilized when running"
                      " PPO. Ignoring.")
            if meta_update_freq is not None:
                print("WARNING: meta_update_freq is not utilized when running"
                      " PPO. Ignoring.")
            if nb_train_steps is not None:
                print("WARNING: nb_train_steps is not utilized when running"
                      " PPO. Ignoring.")

        # Instantiate the ray instance.
        if num_envs > 1:
            ray.init(num_cpus=num_envs+1, ignore_reinit_error=True)

        self.policy = policy
        self.env_name = deepcopy(env) if isinstance(env, str) \
            else env.__str__()
        self.eval_env, _ = create_env(
            eval_env, render_eval, shared, maddpg, evaluate=True)
        self.nb_train_steps = nb_train_steps
        self.nb_rollout_steps = nb_rollout_steps
        self.nb_eval_episodes = nb_eval_episodes
        self.actor_update_freq = actor_update_freq
        self.meta_update_freq = meta_update_freq
        self.reward_scale = reward_scale
        self.render = render
        self.render_eval = render_eval
        self.eval_deterministic = eval_deterministic
        self.save_replay_buffer = save_replay_buffer
        self.num_envs = num_envs
        self.verbose = verbose
        self.policy_kwargs = {'verbose': verbose}

        # Create the environment and collect the initial observations.
        self.sampler, self.obs, self.all_obs, self._info_keys = \
            self.setup_sampler(env, render, shared, maddpg)

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
            self.policy_kwargs.update(MULTIAGENT_PARAMS.copy())
            self.policy_kwargs["all_ob_space"] = all_ob_space

        if is_td3_policy(policy):
            self.policy_kwargs.update(TD3_PARAMS.copy())
        elif is_sac_policy(policy):
            self.policy_kwargs.update(SAC_PARAMS.copy())
        elif is_ppo_policy(policy):
            self.policy_kwargs.update(PPO_PARAMS.copy())
            self.policy_kwargs['num_envs'] = num_envs

        self.policy_kwargs = recursive_update(
            self.policy_kwargs, policy_kwargs or {})

        # Compute the time horizon, which is used to check if an environment
        # terminated early and used to compute the done mask for TD3.
        if self.num_envs > 1:
            self.horizon = ray.get(self.sampler[0].horizon.remote())
        else:
            self.horizon = self.sampler[0].horizon()

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
        self.info_at_done = {key: deque(maxlen=100) for key in self._info_keys}
        self.info_ph = {}
        self.rew_ph = None
        self.rew_history_ph = None
        self.eval_rew_ph = None
        self.eval_success_ph = None
        self.saver = None

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
            ob = [o[0] for o in ob]
            info_key = ray.get(sampler[0].get_init_obs.remote())[1]
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
            ob = [s.get_init_obs()[0] for s in sampler]
            info_key = sampler[0].get_init_obs()[1]

        # Separate the observation and full-state observation.
        obs = [get_obs(o)[0] for o in ob]
        all_obs = [get_obs(o)[1] for o in ob]

        return sampler, obs, all_obs, info_key

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
        """Create the graph, session, policy, and summary objects."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Create the tensorflow session.
            self.sess = make_session(num_cpu=3, graph=self.graph)

            # Create the policy.
            self.policy_tf = self.policy(
                self.sess,
                self.ob_space,
                self.ac_space,
                self.co_space,
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

            # Add the info_dict various to tensorboard as well.
            with tf.compat.v1.variable_scope("info_at_done"):
                for key in self._info_keys:
                    self.info_ph[key] = tf.compat.v1.placeholder(
                        tf.float32, name="{}".format(key))
                    tf.compat.v1.summary.scalar(
                        "{}".format(key), self.info_ph[key])

            # Create the tensorboard summary.
            self.summary = tf.compat.v1.summary.merge_all()

            # Initialize the model parameters and optimizers.
            with self.sess.as_default():
                self.sess.run(tf.compat.v1.global_variables_initializer())
                self.policy_tf.initialize()

            return tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

    def _policy(self,
                obs,
                context,
                apply_noise=True,
                random_actions=False,
                env_num=0):
        """Get the actions from a given observation.

        Parameters
        ----------
        obs : array_like
            the observation
        context : array_like or None
            the contextual term. Set to None if no context is provided by the
            environment.
        apply_noise : bool
            enable the noise
        random_actions : bool
            if set to True, actions are sampled randomly from the action space
            instead of being computed by the policy. This is used for
            exploration purposes.
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

        action = self.policy_tf.get_action(
            obs, context,
            apply_noise=apply_noise,
            random_actions=random_actions,
            env_num=env_num,
        )

        # Flatten the actions. Dictionaries correspond to multi-agent policies.
        if isinstance(action, dict):
            action = {key: action[key].flatten() for key in action.keys()}
        else:
            action = action.flatten()

        return action

    def _store_transition(self,
                          obs0,
                          context0,
                          action,
                          reward,
                          obs1,
                          context1,
                          terminal1,
                          is_final_step,
                          env_num=0,
                          evaluate=False,
                          **kwargs):
        """Store a transition in the replay buffer.

        Parameters
        ----------
        obs0 : array_like
            the last observation
        action : array_like
            the action
        reward : float
            the reward
        obs1 : array_like
            the current observation
        terminal1 : bool
            is the episode done
        is_final_step : bool
            whether the time horizon was met in the step corresponding to the
            current sample. This is used by the TD3 algorithm to augment the
            done mask.
        env_num : int
            the environment number. Used to handle situations when multiple
            parallel environments are being used.
        evaluate : bool
            whether the sample is being provided by the evaluation environment.
            If so, the data is not stored in the replay buffer.
        kwargs : dict
            additional parameters, containing the current and next-step full
            observations for policies using MADDPG
        """
        # Scale the rewards by the provided term. Rewards are dictionaries when
        # training independent multi-agent policies.
        if isinstance(reward, dict):
            reward = {k: self.reward_scale * reward[k] for k in reward.keys()}
        else:
            reward *= self.reward_scale

        self.policy_tf.store_transition(
            obs0=obs0,
            context0=context0,
            action=action,
            reward=reward,
            obs1=obs1,
            context1=context1,
            done=terminal1,
            is_final_step=is_final_step,
            env_num=env_num,
            evaluate=evaluate,
            **(kwargs if self.policy_kwargs.get("maddpg", False) else {}),
        )

    def learn(self,
              total_steps,
              log_dir=None,
              seed=None,
              log_interval=2000,
              eval_interval=50000,
              save_interval=10000,
              initial_exploration_steps=10000):
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
        log_interval : int
            the number of training steps before logging training results
        eval_interval : int
            number of simulation steps in the training environment before an
            evaluation is performed
        save_interval : int
            number of simulation steps in the training environment before the
            model is saved
        initial_exploration_steps : int
            number of timesteps that the policy is run before training to
            initialize the replay buffer with samples
        """
        # Include warnings if using PPO.
        if is_ppo_policy(self.policy):
            if log_interval is not None:
                print("WARNING: log_interval for PPO policies set to after "
                      "every training iteration.")
            log_interval = self.nb_rollout_steps

            if initial_exploration_steps > 0:
                print("WARNING: initial_exploration_steps set to 0 for PPO "
                      "policies.")
                initial_exploration_steps = 0

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
            # Collect preliminary random samples.
            if initial_exploration_steps > 0:
                print("Collecting initial exploration samples...")
                self._collect_samples(run_steps=initial_exploration_steps,
                                      random_actions=True)
                print("Done!")

            # Reset total statistics variables.
            self.episodes = 0
            self.total_steps = 0
            self.episode_rew_history = deque(maxlen=100)
            self.info_at_done = {
                key: deque(maxlen=100) for key in self._info_keys}

            while True:
                # Reset epoch-specific variables.
                self.epoch_episodes = 0
                self.epoch_episode_steps = []
                self.epoch_episode_rewards = []

                for _ in range(round(log_interval / self.nb_rollout_steps)):
                    # If the requirement number of time steps has been met,
                    # terminate training.
                    if self.total_steps >= total_steps:
                        return

                    # Perform rollouts.
                    self._collect_samples()

                    # Train.
                    self._train()

                # Log statistics.
                self._log_training(train_filepath, start_time)

                # Evaluate.
                if self.eval_env is not None and \
                        (self.total_steps - eval_steps_incr) >= eval_interval:
                    eval_steps_incr += eval_interval

                    # Run the evaluation operations over the evaluation env(s).
                    # Note that multiple evaluation envs can be provided.
                    if isinstance(self.eval_env, list):
                        eval_rewards = []
                        eval_successes = []
                        eval_info = []
                        for env in self.eval_env:
                            rew, suc, inf = self._evaluate(env)
                            eval_rewards.append(rew)
                            eval_successes.append(suc)
                            eval_info.append(inf)
                    else:
                        eval_rewards, eval_successes, eval_info = \
                            self._evaluate(self.eval_env)

                    # Log the evaluation statistics.
                    self._log_eval(eval_filepath, start_time, eval_rewards,
                                   eval_successes, eval_info)

                # Run and store summary.
                if writer is not None:
                    td_map = self.policy_tf.get_td_map()

                    # Check if td_map is empty.
                    if not td_map:
                        break

                    td_map.update({
                        self.rew_ph: np.mean(self.epoch_episode_rewards),
                        self.rew_history_ph: np.mean(self.episode_rew_history),
                    })
                    td_map.update({
                        self.info_ph[key]: np.mean(self.info_at_done[key])
                        for key in self.info_ph.keys()
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

        # Save data from the replay buffer.
        if self.save_replay_buffer:
            self.policy_tf.replay_buffer.save(
                save_path + "-{}.rb".format(self.total_steps))

    def load(self, load_path):
        """Load model parameters from a checkpoint.

        Parameters
        ----------
        load_path : str
            location of the checkpoint
        """
        self.saver.restore(self.sess, load_path)

        # Load pre-existing replay buffers.
        if self.save_replay_buffer:
            self.policy_tf.replay_buffer.load(load_path + ".rb")

    def _collect_samples(self, run_steps=None, random_actions=False):
        """Perform the sample collection operation over multiple steps.

        This method calls collect_sample for a multiple steps, and attempts to
        run the operation in parallel if multiple environments are available.

        Parameters
        ----------
        run_steps : int, optional
            number of steps to collect samples from. If not provided, the value
            defaults to `self.nb_rollout_steps`.
        random_actions : bool
            if set to True, actions are sampled randomly from the action space
            instead of being computed by the policy. This is used for
            exploration purposes.
        """
        # Loop through the sampling procedure the number of times it would
        # require to run through each environment in parallel until the number
        # of required steps have been collected.
        run_steps = run_steps or self.nb_rollout_steps
        n_itr = math.ceil(run_steps / self.num_envs)
        for itr in range(n_itr):
            n_steps = self.num_envs if itr < n_itr - 1 \
                else run_steps - (n_itr - 1) * self.num_envs

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
                random_actions=random_actions,
                env_num=env_num,
            ) for env_num in range(n_steps)]

            # Update the environment.
            if self.num_envs > 1:
                ret = ray.get([
                    self.sampler[env_num].collect_sample.remote(
                        action=action[env_num])
                    for env_num in range(n_steps)
                ])
            else:
                ret = [self.sampler[0].collect_sample(action=action[0])]

            for ret_i in ret:
                num = ret_i["env_num"]
                context = ret_i["context"]
                action = ret_i["action"]
                reward = ret_i["reward"]
                obs = ret_i["obs"]
                done = ret_i["done"]
                all_obs = ret_i["all_obs"]
                info = ret_i["info"]

                # Store a transition in the replay buffer.
                self._store_transition(
                    obs0=self.obs[num],
                    context0=context,
                    action=action,
                    reward=reward,
                    obs1=obs[0] if done else obs,
                    context1=context,
                    terminal1=done,
                    is_final_step=(self.episode_step[num] >= self.horizon - 1),
                    all_obs0=self.all_obs[num],
                    all_obs1=all_obs[0] if done else all_obs,
                    env_num=num,
                )

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

                    # Store the info value at the end of the rollout.
                    for key in info.keys():
                        self.info_at_done[key].append(info[key])

    def _train(self):
        """Perform the training operation."""
        if is_td3_policy(self.policy) or is_sac_policy(self.policy):
            # Added to adjust the actor update frequency based on the rate at
            # which training occurs.
            train_itr = int(self.total_steps / self.nb_rollout_steps)
            num_levels = getattr(self.policy_tf, "num_levels", 2)

            if is_goal_conditioned_policy(self.policy):
                # specifies whether to update the meta actor and critic
                # policies based on the meta and actor update frequencies
                kwargs = {
                    "update_meta": [
                        train_itr % self.meta_update_freq ** i == 0
                        for i in range(1, num_levels)
                    ],
                    "update_meta_actor": [
                        train_itr % (self.meta_update_freq ** i *
                                     self.actor_update_freq) == 0
                        for i in range(1, num_levels)
                    ]
                }
            else:
                kwargs = {}

            # Specifies whether to update the actor policy, base on the actor
            # update frequency.
            update = train_itr % self.actor_update_freq == 0

            # Run a step of training from batch.
            for _ in range(self.nb_train_steps):
                self.policy_tf.update(update_actor=update, **kwargs)
        else:
            # for PPO policies
            self.policy_tf.update()

    def _evaluate(self, env):
        """Perform the evaluation operation.

        This method runs the evaluation environment for a number of episodes
        and returns the cumulative rewards and successes from each environment.

        Parameters
        ----------
        env : gym.Env
            the evaluation environment that the policy is meant to be tested on

        Returns
        -------
        list of float
            the list of cumulative rewards from every episode in the evaluation
            phase
        list of bool
            a list of boolean terms representing if each episode ended in
            success or not. If the list is empty, then the environment did not
            output successes or failures, and the success rate will be set to
            zero.
        dict
            additional information that is meant to be logged
        """
        num_steps = deepcopy(self.total_steps)
        eval_episode_rewards = []
        eval_episode_successes = []
        ret_info = {'initial': [], 'final': [], 'average': []}

        if self.verbose >= 1:
            for _ in range(3):
                print("-------------------")
            print("Running evaluation for {} episodes:".format(
                self.nb_eval_episodes))

        # Clear replay buffer-related memory in the policy to allow for the
        # meta-actions to properly updated.
        if is_goal_conditioned_policy(self.policy):
            self.policy_tf.clear_memory(0)

        for i in range(self.nb_eval_episodes):
            # Reset the environment.
            eval_obs = env.reset()
            eval_obs, eval_all_obs = get_obs(eval_obs)

            # Reset rollout-specific variables.
            eval_episode_reward = 0.
            eval_episode_step = 0

            rets = np.array([])
            while True:
                # Collect the contextual term. None if it is not passed.
                context = [env.current_context] \
                    if hasattr(env, "current_context") else None

                eval_action = self._policy(
                    obs=eval_obs,
                    context=context,
                    apply_noise=not self.eval_deterministic,
                    random_actions=False,
                    env_num=0,
                )

                # Update the environment.
                obs, eval_r, done, info = env.step(eval_action)
                obs, all_obs = get_obs(obs)

                if self.env_name == "HumanoidMaze":
                    eval_r = 0.72 * np.log(eval_r)

                # Visualize the current step.
                if self.render_eval:
                    self.eval_env.render()  # pragma: no cover

                # Add the distance to this list for logging purposes (applies
                # only to the Ant* environments).
                if hasattr(env, "current_context"):
                    context = getattr(env, "current_context")
                    reward_fn = getattr(env, "contextual_reward")
                    rets = np.append(rets, reward_fn(eval_obs, context, obs))
                    if self.env_name == "HumanoidMaze":
                        rets[-1] = 0.72 * np.log(rets[-1])

                # Get the contextual term.
                context0 = context1 = getattr(env, "current_context", None)

                # Store a transition in the replay buffer. This is just for the
                # purposes of calling features in the store_transition method
                # of the policy.
                self._store_transition(
                    obs0=eval_obs,
                    context0=context0,
                    action=eval_action,
                    reward=eval_r,
                    obs1=obs,
                    context1=context1,
                    terminal1=False,
                    is_final_step=False,
                    all_obs0=eval_all_obs,
                    all_obs1=all_obs,
                    evaluate=True,
                )

                # Update the previous step observation.
                eval_obs = obs.copy()
                eval_all_obs = all_obs

                # Increment the reward and step count.
                num_steps += 1
                eval_episode_reward += eval_r
                eval_episode_step += 1

                if done:
                    eval_episode_rewards.append(eval_episode_reward)
                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        eval_episode_successes.append(float(maybe_is_success))

                    if self.verbose >= 1:
                        if rets.shape[0] > 0:
                            print("%d/%d: initial: %.3f, final: %.3f, average:"
                                  " %.3f, success: %d"
                                  % (i + 1, self.nb_eval_episodes, rets[0],
                                     rets[-1], float(rets.mean()),
                                     int(info.get('is_success'))))
                        else:
                            print("%d/%d" % (i + 1, self.nb_eval_episodes))

                    if hasattr(env, "current_context"):
                        ret_info['initial'].append(rets[0])
                        ret_info['final'].append(rets[-1])
                        ret_info['average'].append(float(rets.mean()))

                    # Exit the loop.
                    break

        if self.verbose >= 1:
            print("Done.")
            print("Average return: {}".format(np.mean(eval_episode_rewards)))
            if len(eval_episode_successes) > 0:
                print("Success rate: {}".format(
                    np.mean(eval_episode_successes)))
            for _ in range(3):
                print("-------------------")
            print("")

        # get the average of the reward information
        ret_info['initial'] = np.mean(ret_info['initial'])
        ret_info['final'] = np.mean(ret_info['final'])
        ret_info['average'] = np.mean(ret_info['average'])

        # Clear replay buffer-related memory in the policy once again so that
        # it does not affect the training procedure.
        if is_goal_conditioned_policy(self.policy):
            self.policy_tf.clear_memory(0)

        return eval_episode_rewards, eval_episode_successes, ret_info

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

        # Information passed by the environment.
        combined_stats.update({
            'info_at_done/{}'.format(key): np.mean(self.info_at_done[key])
            for key in self.info_at_done.keys()
        })

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

    def _log_eval(self, file_path, start_time, rewards, successes, info):
        """Log evaluation statistics.

        Parameters
        ----------
        file_path : str
            path to the evaluation csv file
        start_time : float
            the time when training began. This is used to print the total
            training time.
        rewards : array_like
            the list of cumulative rewards from every episode in the evaluation
            phase
        successes : list of bool
            a list of boolean terms representing if each episode ended in
            success or not. If the list is empty, then the environment did not
            output successes or failures, and the success rate will be set to
            zero.
        info : dict
            additional information that is meant to be logged
        """
        duration = time.time() - start_time

        if isinstance(info, dict):
            rewards = [rewards]
            successes = [successes]
            info = [info]

        for i, (rew, suc, info_i) in enumerate(zip(rewards, successes, info)):
            if len(suc) > 0:
                success_rate = np.mean(suc)
            else:
                success_rate = 0  # no success rate to log

            evaluation_stats = {
                "duration": duration,
                "total_step": self.total_steps,
                "success_rate": success_rate,
                "average_return": np.mean(rew)
            }
            # Add additional evaluation information.
            evaluation_stats.update(info_i)

            if file_path is not None:
                # Add an evaluation number to the csv file in case of multiple
                # evaluation environments.
                eval_fp = file_path[:-4] + "_{}.csv".format(i)
                exists = os.path.exists(eval_fp)

                # Save evaluation statistics in a csv file.
                with open(eval_fp, "a") as f:
                    w = csv.DictWriter(f, fieldnames=evaluation_stats.keys())
                    if not exists:
                        w.writeheader()
                    w.writerow(evaluation_stats)
