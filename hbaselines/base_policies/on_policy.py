"""Script containing the abstract on-policy policy class."""
import numpy as np

from hbaselines.base_policies.policy import Policy


class OnPolicyPolicy(Policy):
    """Base On-Policy Policy.

    This class extends the base policy class by including methods to both
    process minibatches of data into array of correct shapes, and compute the
    Generalized Advantage Estimates.

    Attributes
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
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
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
        clipping parameter for the value function, it can be a function. This
        is a parameter specific to the OpenAI implementation. If None is passed
        (default), then `cliprange` (that is used for the policy) will be used.
        IMPORTANT: this clipping depends on the reward scaling. To deactivate
        value function clipping (and recover the original PPO implementation),
        you have to pass a negative value (e.g. -1).
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 co_space,
                 verbose,
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
                 l2_penalty,
                 model_params,
                 num_envs=1):
        """Instantiate the base policy object.

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
        """
        super(OnPolicyPolicy, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            co_space=co_space,
            verbose=verbose,
            l2_penalty=l2_penalty,
            model_params=model_params,
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

    def initialize(self):
        """See parent class."""
        raise NotImplementedError

    def update(self, update_actor=True, **kwargs):
        """See parent class."""
        raise NotImplementedError

    def get_action(self, obs, context, apply_noise, random_actions, env_num=0):
        """See parent class."""
        raise NotImplementedError

    def store_transition(self, obs0, context0, action, reward, obs1, context1,
                         done, is_final_step, env_num=0, evaluate=False):
        """See parent class."""
        raise NotImplementedError

    def get_td_map(self):
        """See parent class."""
        raise NotImplementedError

    def _gae_returns(self, mb_rewards, mb_values, mb_dones, last_values):
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
        last_values : array_like
            the value associated with the current observation within the
            environment

        Returns
        -------
        array_like
            GAE-style expected discounted returns.
        """
        n_steps = mb_rewards.shape[0]

        # Discount/bootstrap off value fn.
        mb_advs = np.zeros_like(mb_rewards)
        mb_vactual = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
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

    def process_minibatch(self,
                          mb_obs,
                          mb_contexts,
                          mb_actions,
                          mb_values,
                          mb_neglogpacs,
                          mb_all_obs,
                          mb_rewards,
                          mb_returns,
                          mb_dones,
                          last_values):
        """Process a minibatch of samples.

        This method re-formats the data to numpy arrays that can be passed to
        the tensorflow placeholders, and computes the GAE terms

        Parameters
        ----------
        mb_obs : array_like
            a minibatch of observations
        mb_contexts : array_like
            a minibatch of contextual terms
        mb_actions : array_like
            a minibatch of actions
        mb_values : array_like
            a minibatch of estimated values by the policy
        mb_neglogpacs : array_like
            a minibatch of the negative log-likelihood of performed actions
        mb_all_obs : array_like
            a minibatch of full state observations (for multiagent envs)
        mb_rewards : array_like
            a minibatch of environment rewards
        mb_returns : array_like
            a minibatch of expected discounted returns
        mb_dones : array_like
            a minibatch of done masks
        last_values : array_like
            the value associated with the current observation within the
            environment

        Returns
        -------
        array_like
            the reformatted minibatch of observations
        array_like
            the reformatted minibatch of contextual terms
        array_like
            the reformatted minibatch of actions
        array_like
            the reformatted minibatch of estimated values by the policy
        array_like
            the reformatted minibatch of the negative log-likelihood of
            performed actions
        array_like
            the reformatted minibatch of full state observations (for
            multiagent envs)
        array_like
            the reformatted minibatch of environment rewards
        array_like
            the reformatted minibatch of expected discounted returns
        array_like
            the reformatted minibatch of done masks
        array_like
            a minibatch of estimated advantages
        int
            the number of sampled steps in the minibatch
        """
        n_steps = 0

        for env_num in range(self.num_envs):
            # Convert the data to numpy arrays.
            mb_obs[env_num] = np.concatenate(mb_obs[env_num], axis=0)
            mb_rewards[env_num] = np.asarray(mb_rewards[env_num])
            mb_actions[env_num] = np.concatenate(mb_actions[env_num], axis=0)
            mb_values[env_num] = np.concatenate(mb_values[env_num], axis=0)
            mb_neglogpacs[env_num] = np.concatenate(
                mb_neglogpacs[env_num], axis=0)
            mb_dones[env_num] = np.asarray(mb_dones[env_num])
            n_steps += mb_obs[env_num].shape[0]

            # Compute the bootstrapped/discounted returns.
            mb_returns[env_num] = self._gae_returns(
                mb_rewards=mb_rewards[env_num],
                mb_values=mb_values[env_num],
                mb_dones=mb_dones[env_num],
                last_values=last_values[env_num],
            )

        # Concatenate the stored data.
        if self.num_envs > 1:
            mb_obs = np.concatenate(mb_obs, axis=0)
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

        # Compute the advantages.
        advs = mb_returns - mb_values
        mb_advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        return mb_obs, mb_contexts, mb_actions, mb_values, mb_neglogpacs, \
            mb_all_obs, mb_rewards, mb_returns, mb_dones, mb_advs, n_steps
