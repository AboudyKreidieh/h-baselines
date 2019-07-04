"""Deep Deterministic Policy Gradient (DDPG) algorithm.

See: https://arxiv.org/pdf/1509.02971.pdf

A large portion of this code is adapted from the following repository:
https://github.com/hill-a/stable-baselines
"""
import os
import time
from collections import deque
import csv
import warnings
import random

from gym.spaces import Box
import numpy as np
import tensorflow as tf
from mpi4py import MPI

from hbaselines.utils.train import ensure_dir
from hbaselines.hiro.tf_util import get_trainable_vars
from hbaselines.envs.efficient_hrl.envs import AntMaze, AntFall, AntPush
from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity
from stable_baselines.ddpg.policies import DDPGPolicy


def as_scalar(scalar):
    """Check and return the input if it is a scalar.

    If it is not scalar, raise a ValueError.

    Parameters
    ----------
    scalar : Any
        the object to check

    Returns
    -------
    float
        the scalar if x is a scalar
    """
    if isinstance(scalar, np.ndarray):
        assert scalar.size == 1
        return scalar[0]
    elif np.isscalar(scalar):
        return scalar
    else:
        raise ValueError('expected scalar, got %s' % scalar)


class DDPG(OffPolicyRLModel):
    """Deep Deterministic Policy Gradient (DDPG) model.

    See: https://arxiv.org/pdf/1509.02971.pdf

    Attributes
    ----------
    policy : type [ hbaselines.hiro.policy.ActorCriticPolicy ]
        the policy model to use
    env : gym.Env or str
        the environment to learn from (if registered in Gym, can be str)
    gamma : float
        the discount rate
    eval_env : gym.Env or str
        the environment to evaluate from (if registered in Gym, can be str)
    nb_train_steps : int
        the number of training steps
    nb_rollout_steps : int
        the number of rollout steps
    nb_eval_episodes : int
        the number of evaluation episodes
    normalize_observations : bool
        should the observation be normalized
    tau : float
        the soft update coefficient (keep old values, between 0 and 1)
    batch_size : int
        the size of the batch for learning the policy
    normalize_returns : bool
        should the critic output be normalized
    observation_range : (float, float)
        the bounding values for the observation
    critic_l2_reg : float
        l2 regularizer coefficient
    return_range : (float, float)
        the bounding values for the critic output
    actor_lr : float
        the actor learning rate
    critic_lr : float
        the critic learning rate
    clip_norm : float
        clip the gradients (disabled if None)
    reward_scale : float
        the value the reward should be scaled by
    render : bool
        enable rendering of the training environment
    render_eval : bool
        enable rendering of the evaluation environment
    buffer_size : int
        the max number of transitions to store
    random_exploration : float
        fraction of actions that are randomly selected between the action range
        (to be used in HER+DDPG)
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    tensorboard_log : str
        the log location for tensorboard (if None, no logging)
    policy_kwargs : dict
        additional policy parameters
    graph : tf.Graph
        the current tensorflow graph
    policy_tf : hbaselines.hiro.policy.ActorCriticPolicy
        the policy object
    sess : tf.Session
        the current tensorflow session
    params : list of str
        the names of the trainable parameters
    summary : TODO
        TODO
    target_params : list of str
        the names of the parameters in the target actor/critic
    obs_rms_params : list of str
        TODO
    ret_rms_params : list of str
        TODO
    obs : array_like
        the most recent training observation
    episode_step : int
        the number of steps since the most recent rollout began
    episodes : int
        the total number of rollouts performed since training began
    total_steps : int
        the total number of steps that have been executed since training began
    epoch_episode_rewards : list of float
        a list of cumulative rollout rewards from the most recent training
        iterations
    epoch_episode_steps : list of int
        a list of rollout lengths from the most recent training iterations
    epoch_actor_losses : list of float
        the actor loss values from each SGD step in the most recent training
        iteration
    epoch_critic_losses : list of float
        the critic loss values from each SGD step in the most recent training
        iteration
    epoch_actions : list of array_like
        a list of the actions that were performed during the most recent
        training iteration
    epoch_qs : list of float
        a list of the Q values that were calculated during the most recent
        training iteration
    epoch_episodes : int
        the total number of rollouts performed since the most recent training
        iteration began
    epoch : int
        the total number of training iterations
    episode_rewards_history : list of float
        the cumulative return from the last 100 training episodes
    episode_reward : float
        the cumulative reward since the most reward began
    """

    def __init__(self,
                 policy,
                 env,
                 gamma=0.99,
                 eval_env=None,
                 nb_train_steps=50,
                 nb_rollout_steps=100,
                 nb_eval_episodes=50,
                 normalize_observations=False,
                 tau=0.001,
                 batch_size=128,
                 normalize_returns=False,
                 observation_range=(-5., 5.),
                 critic_l2_reg=0.,
                 return_range=(-np.inf, np.inf),
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 clip_norm=None,
                 reward_scale=1.,
                 render=False,
                 render_eval=False,
                 memory_limit=None,
                 buffer_size=50000,
                 random_exploration=0.0,
                 verbose=0,
                 tensorboard_log=None,
                 _init_setup_model=True,
                 policy_kwargs=None):
        """Instantiate the algorithm object.

        Parameters
        ----------
        policy : type [ hbaselines.hiro.policy.ActorCriticPolicy ]
            the policy model to use
        env : gym.Env or str
            the environment to learn from (if registered in Gym, can be str)
        gamma : float
            the discount rate
        eval_env : gym.Env or str
            the environment to evaluate from (if registered in Gym, can be str)
        nb_train_steps : int
            the number of training steps
        nb_rollout_steps : int
            the number of rollout steps
        nb_eval_episodes : int
            the number of evaluation episodes
        normalize_observations : bool
            should the observation be normalized
        tau : float
            the soft update coefficient (keep old values, between 0 and 1)
        batch_size : int
            the size of the batch for learning the policy
        normalize_returns : bool
            should the critic output be normalized
        observation_range : (float, float)
            the bounding values for the observation
        critic_l2_reg : float
            l2 regularizer coefficient
        return_range : (float, float)
            the bounding values for the critic output
        actor_lr : float
            the actor learning rate
        critic_lr : float
            the critic learning rate
        clip_norm : float
            clip the gradients (disabled if None)
        reward_scale : float
            the value the reward should be scaled by
        render : bool
            enable rendering of the training environment
        render_eval : bool
            enable rendering of the evaluation environment
        buffer_size : int
            the max number of transitions to store
        random_exploration : float
            fraction of actions that are randomly selected between the action
            range (to be used in HER+DDPG)
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        tensorboard_log : str
            the log location for tensorboard (if None, no logging)
        _init_setup_model : bool
            Whether or not to build the network at the creation of the instance
        policy_kwargs : dict
            additional policy parameters
        """
        self.policy = policy
        self.env = self._create_env(env, evaluate=False)
        self.gamma = gamma
        self.eval_env = self._create_env(eval_env, evaluate=True)

        # TODO: hack
        super(DDPG, self).__init__(
            policy=policy, env=self.env, replay_buffer=None, verbose=verbose,
            policy_base=DDPGPolicy, requires_vec_env=False,
            policy_kwargs=policy_kwargs)

        self.nb_train_steps = nb_train_steps
        self.nb_rollout_steps = nb_rollout_steps
        self.nb_eval_episodes = nb_eval_episodes
        self.normalize_observations = normalize_observations
        self.tau = tau
        self.batch_size = batch_size
        self.normalize_returns = normalize_returns
        self.observation_range = observation_range
        self.critic_l2_reg = critic_l2_reg
        self.return_range = return_range
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.reward_scale = reward_scale
        self.render = render
        self.render_eval = render_eval
        self.memory_limit = memory_limit
        self.buffer_size = buffer_size
        self.random_exploration = random_exploration
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log
        self.policy_kwargs = policy_kwargs

        # init
        self.graph = None
        self.policy_tf = None
        self.sess = None
        self.params = None
        self.summary = None
        self.target_params = None
        self.obs_rms_params = None
        self.ret_rms_params = None
        self.obs = None
        self.episode_step = None
        self.episodes = None
        self.total_steps = None
        self.epoch_episode_rewards = None
        self.epoch_episode_steps = None
        self.epoch_actor_losses = None
        self.epoch_critic_losses = None
        self.epoch_actions = None
        self.epoch_qs = None
        self.epoch_episodes = None
        self.epoch = None
        self.episode_rewards_history = None
        self.episode_reward = None

        if _init_setup_model:
            self.setup_model()

    @staticmethod
    def _create_env(env, evaluate=False):
        """Return, and potentially create, the environment.

        Parameters
        ----------
        env : str or gym.Env
            the environment, or the name of a registered environment.
        evaluate : bool, optional
            specifies whether this is a training or evaluation environment

        Returns
        -------
        gym.Env
            a gym-compatible environment
        """
        if env == "AntMaze":
            if evaluate:
                env = AntMaze(use_contexts=True, context_range=[16, 0])
                # env = AntMaze(use_contexts=True, context_range=[16, 16])
                # env = AntMaze(use_contexts=True, context_range=[0, 16])
            else:
                env = AntMaze(use_contexts=True,
                              random_contexts=True,
                              context_range=[(-4, 20), (-4, 20)])
        elif env == "AntPush":
            if evaluate:
                env = AntPush(use_contexts=True, context_range=[0, 19])
            else:
                env = AntPush(use_contexts=True, context_range=[0, 19])
                # env = AntPush(use_contexts=True,
                #               random_contexts=True,
                #               context_range=[(-16, 16), (-4, 20)])
        elif env == "AntFall":
            if evaluate:
                env = AntFall(use_contexts=True, context_range=[0, 27, 4.5])
            else:
                env = AntFall(use_contexts=True, context_range=[0, 27, 4.5])
                # env = AntFall(use_contexts=True,
                #               random_contexts=True,
                #               context_range=[(-4, 12), (-4, 28), (0, 5)])

        # TODO: remove once I've eliminated the parent class.
        if env is not None:
            env.reset()

        return env

    def setup_model(self):
        """Create the graph, session, policy, and summary objects."""
        with SetVerbosity(self.verbose):
            # determine whether the action space is continuous
            assert isinstance(self.action_space, Box), \
                "Error: DDPG cannot output a {} action space, only " \
                "spaces.Box is supported.".format(self.action_space)

            self.graph = tf.Graph()
            with self.graph.as_default():
                # Create the tensorflow session.
                self.sess = tf_util.make_session(num_cpu=1, graph=self.graph)

                # Create the policy.
                self.policy_tf = self.policy(
                    self.sess,
                    self.observation_space,
                    self.action_space,
                    return_range=self.return_range,
                    buffer_size=self.buffer_size,
                    batch_size=self.batch_size,
                    actor_lr=self.actor_lr,
                    critic_lr=self.critic_lr,
                    clip_norm=self.clip_norm,
                    critic_l2_reg=self.critic_l2_reg,
                    verbose=self.verbose,
                    tau=self.tau,
                    gamma=self.gamma,
                    normalize_observations=self.normalize_observations,
                    normalize_returns=self.normalize_returns,
                    observation_range=self.observation_range
                )

                self.params = \
                    get_trainable_vars("model") + \
                    get_trainable_vars('noise/') + \
                    get_trainable_vars('noise_adapt/')

                self.target_params = get_trainable_vars("target")
                self.obs_rms_params = [var for var in tf.global_variables()
                                       if "obs_rms" in var.name]
                self.ret_rms_params = [var for var in tf.global_variables()
                                       if "ret_rms" in var.name]

                # Initialize the model parameters and optimizers.
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.policy_tf.initialize()

                self.summary = tf.summary.merge_all()

    # TODO: delete
    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        deterministic_action = policy.actor_tf * np.abs(self.action_space.low)

        return policy.obs_ph, self.actions, deterministic_action

    def _policy(self, obs, apply_noise=True, compute_q=True):
        """Get the actions and critic output, from a given observation.

        Parameters
        ----------
        obs : list of float or list of int
            the observation
        apply_noise : bool
            enable the noise
        compute_q : bool
            compute the critic output

        Returns
        -------
        list of float
            the action value
        float
            the critic value
        """
        obs = np.array(obs).reshape((-1,) + self.observation_space.shape)

        # TODO: add noise
        action = self.policy_tf.get_action(obs, time=self.episode_step)
        action = action.flatten()
        action = np.clip(action, -1, 1)

        q_value = self.policy_tf.value(obs) if compute_q else None

        return action, q_value

    def _store_transition(self, obs0, action, reward, obs1, terminal1):
        """Store a transition in the replay buffer.

        Parameters
        ----------
        obs0 : list of float or list of int
            the last observation
        action : list of float or np.ndarray
            the action
        reward : float
            the reward
        obs1 : list fo float or list of int
            the current observation
        terminal1 : bool
            is the episode done
        """
        reward *= self.reward_scale
        self.policy_tf.store_transition(obs0, action, reward, obs1, terminal1)

    def _initialize(self):
        """Initialize the model parameters and optimizers."""
        self.sess.run(tf.global_variables_initializer())

    def learn(self,
              total_timesteps,
              log_dir=None,
              callback=None,  # TODO: delete
              seed=None,
              log_interval=100,
              eval_interval=5e4,
              tb_log_name="DDPG",  # TODO: delete
              reset_num_timesteps=True,  # TODO: delete
              exp_num=None):
        """Return a trained model.

        Parameters
        ----------
        total_timesteps : int
            the total number of samples to train on
        log_dir : str
            the directory where the training and evaluation statistics, as well
            as the tensorboard log, should be stored
        seed : int or None
            the initial seed for training, if None: keep current seed
        callback : function (dict, dict) -> boolean
            function called at every steps with state of the algorithm. It
            takes the local and global variables. If it returns False, training
            is aborted.
        log_interval : int
            the number of training steps before logging training results
        eval_interval : int
            number of simulation steps in the training environment before an
            evaluation is performed
        tb_log_name : str
            the name of the run for tensorboard log
        reset_num_timesteps : bool
            whether or not to reset the current timestep number (used in
            logging)
        exp_num : int, optional
            an additional experiment number term used by the runner scripts
            when running multiple experiments simultaneously. If set to None,
            the train, evaluate, and tensorboard results are stored in log_dir
            immediately

        Returns
        -------
        TODO
            the trained model
        """
        if exp_num is not None:
            log_dir = os.path.join(log_dir, "trial_{}".format(exp_num))

        # Make sure that the log directory exists, and if not, make it.
        ensure_dir(log_dir)

        # Create a tensorboard object for logging.
        # save_path = os.path.join(log_dir, tb_log_name)
        # writer = tf.summary.FileWriter(save_path, graph=self.graph)
        writer = None

        # file path for training and evaluation results
        train_filepath = os.path.join(log_dir, "train.csv")
        eval_filepath = os.path.join(log_dir, "eval.csv")

        with SetVerbosity(self.verbose):
            # Setup the seed value.
            random.seed(seed)
            np.random.seed(seed)
            tf.set_random_seed(seed)

            # we assume symmetric actions.  # FIXME
            assert np.all(np.abs(self.env.action_space.low)
                          == self.env.action_space.high)
            if self.verbose >= 2:
                logger.log('Using agent with the following configuration:')
                logger.log(str(self.__dict__.items()))

            # Initialize class variables.
            steps_incr = 0
            self.episode_reward = 0.
            self.episode_step = 0
            self.episodes = 0
            self.total_steps = 0
            self.epoch = 0
            self.episode_rewards_history = deque(maxlen=100)
            self.episode_reward = np.zeros((1,))

            with self.sess.as_default(), self.graph.as_default():
                # Prepare everything.
                self.obs = self.env.reset()
                start_time = time.time()

                while True:
                    # Reset epoch-specific variables.
                    self.epoch_episodes = 0
                    self.epoch_actions = []
                    self.epoch_qs = []
                    self.epoch_actor_losses = []
                    self.epoch_critic_losses = []
                    self.epoch_episode_rewards = []
                    self.epoch_episode_steps = []

                    for _ in range(log_interval):
                        # If the requirement number of time steps has been met,
                        # terminate training.
                        if self.total_steps > total_timesteps:
                            return self

                        # Perform rollouts.
                        self._collect_samples(writer)

                        # Train.
                        self._train(writer)

                    # Log statistics.
                    self._log_training(
                        train_filepath,
                        start_time,
                    )

                    # Evaluate.
                    if self.eval_env is not None and \
                            (self.total_steps - steps_incr) >= eval_interval:
                        steps_incr += eval_interval
                        eval_rewards, eval_successes = self._evaluate()
                        self._log_eval(
                            eval_filepath,
                            start_time,
                            eval_rewards,
                            eval_successes
                        )

                    # Update the epoch count.
                    self.epoch += 1

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(
            observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions, _, = self._policy(
            observation, apply_noise=not deterministic, compute_q=False)

        # reshape to the correct action shape
        actions = actions.reshape((-1,) + self.action_space.shape)
        # scale the output for the prediction
        actions = actions * np.abs(self.action_space.low)

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    # TODO: delete
    def action_probability(self, observation, state=None, mask=None,
                           actions=None):
        if actions is not None:
            raise ValueError("Error: DDPG does not have action probabilities.")

        # here there are no action probabilities, as DDPG does not use a
        # probability distribution
        warnings.warn("Warning: action probability is meaningless for DDPG. "
                      "Returning None")

        return None

    def get_parameter_list(self):
        return (self.params +
                self.target_params +
                self.obs_rms_params +
                self.ret_rms_params)

    # TODO: modify to match the way I want it
    def save(self, save_path):
        data = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "nb_eval_episodes": self.nb_eval_episodes,
            "nb_train_steps": self.nb_train_steps,
            "nb_rollout_steps": self.nb_rollout_steps,
            "verbose": self.verbose,
            "gamma": self.gamma,
            "tau": self.tau,
            "normalize_returns": self.normalize_returns,
            "normalize_observations": self.normalize_observations,
            "batch_size": self.batch_size,
            "observation_range": self.observation_range,
            "return_range": self.return_range,
            "critic_l2_reg": self.critic_l2_reg,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "clip_norm": self.clip_norm,
            "reward_scale": self.reward_scale,
            "memory_limit": self.memory_limit,
            "buffer_size": self.buffer_size,
            "random_exploration": self.random_exploration,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path,
                           data=data,
                           params=params_to_save)

    # TODO: modify to match the way I want it
    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, params = cls._load_from_file(load_path)

        if 'policy_kwargs' in kwargs \
                and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError(
                "The specified policy kwargs do not equal the stored policy "
                "kwargs. Stored kwargs: {}, specified kwargs: {}".
                format(data['policy_kwargs'], kwargs['policy_kwargs']))

        model = cls(None, env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()
        # Patch for version < v2.6.0, duplicated keys where saved
        if len(params) > len(model.get_parameter_list()):
            n_params = len(model.params)
            n_target_params = len(model.target_params)
            n_normalisation_params = len(model.obs_rms_params) + len(
                model.ret_rms_params)
            # Check that the issue is the one from
            # https://github.com/hill-a/stable-baselines/issues/363
            assert len(params) == 2 * (n_params + n_target_params) \
                + n_normalisation_params,\
                "The number of parameter saved differs from the number of " \
                "parameters that should be loaded: {}!={}".format(
                    len(params), len(model.get_parameter_list()))

            # Remove duplicates
            params_ = params[:n_params + n_target_params]
            if n_normalisation_params > 0:
                params_ += params[-n_normalisation_params:]
            params = params_
        model.load_parameters(params)

        return model

    def _collect_samples(self, writer):
        """Perform the sample collection operation.

        TODO

        :return:
        """
        rank = MPI.COMM_WORLD.Get_rank()

        for _ in range(self.nb_rollout_steps):
            # Predict next action.
            action, q_value = self._policy(
                self.obs, apply_noise=True, compute_q=True)
            assert action.shape == self.env.action_space.shape

            # Randomly sample actions from a uniform distribution with a
            # probability self.random_exploration (used in HER + DDPG)
            if np.random.rand() < self.random_exploration:
                rescaled_action = action = self.action_space.sample()
            else:
                rescaled_action = action * np.abs(self.action_space.low)

            # Execute next action.
            new_obs, reward, done, info = self.env.step(rescaled_action)

            # Visualize the current step.
            if rank == 0 and self.render:
                self.env.render()

            # Book-keeping.
            self.total_steps += 1
            self.num_timesteps += 1
            if rank == 0 and self.render:
                self.env.render()
            self.episode_reward += reward
            self.episode_step += 1
            self.epoch_actions.append(action)
            self.epoch_qs.append(q_value)

            # Store a transition in the replay buffer.
            self._store_transition(self.obs, action, reward, new_obs, done)

            # Update the current observation.
            self.obs = new_obs

            if done:
                # Episode done.
                self.epoch_episode_rewards.append(self.episode_reward)
                self.episode_rewards_history.append(self.episode_reward)
                self.epoch_episode_steps.append(self.episode_step)
                self.episode_reward = 0.
                self.episode_step = 0
                self.epoch_episodes += 1
                self.episodes += 1

                # Reset the environment.
                self.obs = self.env.reset()

    def _train(self, writer):
        """Perform the training operation.

        Through this method, the actor and critic networks are updated within
        the policy, and the summary information is logged to tensorboard.

        Parameters
        ----------
        writer : TODO
            TODO
        """
        for t_train in range(self.nb_train_steps):
            # Run a step of training from batch.
            critic_loss, actor_loss, td_map = self.policy_tf.update()

            # Run summary.
            if t_train == 0 and writer is not None:
                summary = self.sess.run(self.summary, td_map)
                writer.add_summary(summary, self.total_steps)

            # Add actor and critic loss information for logging purposes.
            self.epoch_critic_losses.append(critic_loss)
            self.epoch_actor_losses.append(actor_loss)

    def _evaluate(self):
        """Perform the evaluation operation.

        TODO

        Returns
        -------
        array_like
            TODO
        list of bool
            a list of boolean terms representing if each episode ended in
            success or not. If the list is empty, then the environment did not
            output successes or failures, and the success rate will be set to
            zero.
        """
        eval_episode_rewards = []
        eval_episode_successes = []

        if self.verbose >= 1:
            for _ in range(3):
                print("-------------------")
            print("Running evaluation for {} episodes:".format(
                self.nb_eval_episodes))

        for i in range(self.nb_eval_episodes):
            # Reset the environment and episode reward.
            eval_obs = self.eval_env.reset()
            eval_episode_reward = 0.

            while True:
                eval_action, _ = self._policy(
                    eval_obs,
                    apply_noise=False,
                    compute_q=False)

                eval_obs, eval_r, eval_done, eval_info = self.eval_env.step(
                    eval_action * np.abs(self.action_space.low))

                if self.render_eval:
                    self.eval_env.render()

                eval_episode_reward += eval_r

                if eval_done:
                    eval_episode_rewards.append(eval_episode_reward)
                    maybe_is_success = eval_info.get('is_success')
                    if maybe_is_success is not None:
                        eval_episode_successes.append(float(maybe_is_success))

                    # Exit the loop.
                    break

            if self.verbose >= 1:
                print("{}/{}...".format(i+1, self.nb_eval_episodes))

        if self.verbose >= 1:
            print("Done.")
            print("Average return: {}".format(np.mean(eval_episode_rewards)))
            if len(eval_episode_successes) > 0:
                print("Success rate: {}".format(
                    np.mean(eval_episode_successes)))
            for _ in range(3):
                print("-------------------")
            print("")

        return eval_episode_rewards, eval_episode_successes

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
        mpi_size = MPI.COMM_WORLD.Get_size()

        # Log statistics.
        duration = time.time() - start_time
        stats = self.policy_tf.get_stats()
        combined_stats = stats.copy()
        combined_stats['rollout/return'] = np.mean(self.epoch_episode_rewards)
        combined_stats['rollout/return_history'] = np.mean(
            self.episode_rewards_history)
        combined_stats['rollout/episode_steps'] = np.mean(
            self.epoch_episode_steps)
        combined_stats['rollout/actions_mean'] = np.mean(self.epoch_actions)
        combined_stats['rollout/Q_mean'] = np.mean(self.epoch_qs)
        combined_stats['train/loss_actor'] = np.mean(self.epoch_actor_losses)
        combined_stats['train/loss_critic'] = np.mean(self.epoch_critic_losses)
        combined_stats['total/duration'] = duration
        combined_stats['total/steps_per_second'] = float(self.total_steps) \
            / float(duration)
        combined_stats['total/episodes'] = self.episodes
        combined_stats['rollout/episodes'] = self.epoch_episodes
        combined_stats['rollout/actions_std'] = np.std(self.epoch_actions)

        combined_stats_sums = MPI.COMM_WORLD.allreduce(
            np.array([as_scalar(x)
                      for x in combined_stats.values()]))
        combined_stats = {
            k: v / mpi_size for (k, v) in
            zip(combined_stats.keys(), combined_stats_sums)
        }

        # Total statistics.
        combined_stats['total/epochs'] = self.epoch + 1
        combined_stats['total/steps'] = self.total_steps

        # Save combined_stats in a csv file.
        if file_path is not None:
            exists = os.path.exists(file_path)
            with open(file_path, 'a') as f:
                w = csv.DictWriter(f, fieldnames=combined_stats.keys())
                if not exists:
                    w.writeheader()
                w.writerow(combined_stats)

        # Print statistics.
        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])
        logger.dump_tabular()
        logger.info('')

    def _log_eval(self,
                  file_path,
                  start_time,
                  eval_episode_rewards,
                  eval_episode_successes):
        """Log evaluation statistics.

        Parameters
        ----------
        file_path : str
            path to the evaluation csv file
        start_time : float
            the time when training began. This is used to print the total
            training time.
        eval_episode_rewards : array_like
            the list of cumulative rewards from every episode in the evaluation
            phase
        eval_episode_successes : list of bool
            a list of boolean terms representing if each episode ended in
            success or not. If the list is empty, then the environment did not
            output successes or failures, and the success rate will be set to
            zero.
        """
        duration = time.time() - start_time

        if len(eval_episode_successes) > 0:
            success_rate = np.mean(eval_episode_successes)
        else:
            success_rate = 0  # no success rate to log

        evaluation_stats = {
            "duration": duration,
            "total_step": self.total_steps,
            "success_rate": success_rate,
            "average_return": np.mean(eval_episode_rewards)
        }

        # Save evaluation statistics in a csv file.
        if file_path is not None:
            exists = os.path.exists(file_path)
            with open(file_path, "a") as f:
                w = csv.DictWriter(f, fieldnames=evaluation_stats.keys())
                if not exists:
                    w.writeheader()
                w.writerow(evaluation_stats)
