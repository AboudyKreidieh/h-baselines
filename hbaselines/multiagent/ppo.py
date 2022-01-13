"""PPO-compatible multi-agent feedforward neural network policy."""
from hbaselines.multiagent.base import MultiAgentPolicy as BasePolicy
from hbaselines.fcnet.ppo import FeedForwardPolicy


class MultiFeedForwardPolicy(BasePolicy):
    """PPO-compatible multi-agent feedforward neural network policy."""

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
                 shared,
                 maddpg,
                 n_agents,
                 all_ob_space=None,
                 num_envs=1,
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
        model_params : dict
            dictionary of model-specific parameters. See parent class.
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
        super(MultiFeedForwardPolicy, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            co_space=co_space,
            verbose=verbose,
            l2_penalty=l2_penalty,
            model_params=model_params,
            shared=shared,
            maddpg=maddpg,
            all_ob_space=all_ob_space,
            n_agents=n_agents,
            base_policy=FeedForwardPolicy,
            num_envs=num_envs,
            scope=scope,
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
