"""Script containing the abstract actor-critic policy class."""
from hbaselines.base_policies.policy import Policy
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.utils.tf_util import get_target_updates


class ActorCriticPolicy(Policy):
    """Base Actor Critic Policy.

    This class extends the abstract policy class by including a method for
    computing soft target updated for the value functions.

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
    buffer_size : int
        the max number of transitions to store
    batch_size : int
        SGD batch size
    actor_lr : float
        actor learning rate
    critic_lr : float
        critic learning rate
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    tau : float
        target update rate
    gamma : float
        discount factor
    use_huber : bool
        specifies whether to use the huber distance function as the loss for
        the critic. If set to False, the mean-squared error metric is used
        instead
    l2_penalty : float
        L2 regularization penalty. This is applied to the policy network.
    model_params : dict
        dictionary of model-specific parameters. See parent class.
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
                 use_huber,
                 l2_penalty,
                 model_params):
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
        """
        super(ActorCriticPolicy, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            co_space=co_space,
            verbose=verbose,
            l2_penalty=l2_penalty,
            model_params=model_params,
        )

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = gamma
        self.use_huber = use_huber

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

    @staticmethod
    def _setup_target_updates(model_scope, target_scope, scope, tau, verbose):
        """Create the soft and initial target updates.

        The initial model parameters are assumed to be stored under the scope
        name "model", while the target policy parameters are assumed to be
        under the scope name "target".

        If an additional outer scope was provided when creating the policies,
        they can be passed under the `scope` parameter.

        Parameters
        ----------
        model_scope : str
            the scope of the model parameters
        target_scope : str
            the scope of the target parameters
        scope : str or None
            the outer scope, set to None if not available
        tau : float
            target update rate
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug

        Returns
        -------
        tf.Operation
            initial target updates, to match the target with the model
        tf.Operation
            soft target update operations
        """
        if scope is not None:
            model_scope = scope + '/' + model_scope
            target_scope = scope + '/' + target_scope

        return get_target_updates(
            get_trainable_vars(model_scope),
            get_trainable_vars(target_scope),
            tau, verbose)
