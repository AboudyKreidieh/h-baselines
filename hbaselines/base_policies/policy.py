"""Script containing the abstract policy class."""
import numpy as np
import tensorflow as tf

from hbaselines.utils.tf_util import get_trainable_vars


class Policy(object):
    """Base Policy.

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
    l2_penalty : float
        L2 regularization penalty. This is applied to the policy network.
    model_params : dict
        dictionary of model-specific parameters. The following must be
        specified:

        * model_type (str): the type of model to use. Must be one of {"fcnet",
          "conv"}.
        * layers (list of int or None): the size of the Neural network for the
          policy
        * layer_norm (bool): enable layer normalisation
        * batch_norm (bool): enable batch normalisation
        * dropout (bool): enable dropout
        * act_fun (tf.nn.*): the activation function to use in the neural
          network

        In addition, the following parameters may be required dependent on
        your choice of model type:

        * ignore_image (bool): observation includes an image but should it be
          ignored. Required if "model_type" is set to "conv".
        * image_height (int): the height of the image in the observation.
          Required if "model_type" is set to "conv".
        * image_width (int): the width of the image in the observation.
          Required if "model_type" is set to "conv".
        * image_channels (int): the number of channels of the image in the
          observation. Required if "model_type" is set to "conv".
        * kernel_sizes (list of int): the kernel size of the neural network
          conv layers for the policy. Required if "model_type" is set to
          "conv".
        * strides (list of int): the kernel size of the neural network conv
          layers for the policy. Required if "model_type" is set to "conv".
        * filters (list of int): the channels of the neural network conv
          layers for the policy. Required if "model_type" is set to "conv".
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 co_space,
                 verbose,
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
            dictionary of model-specific parameters. The following must be
            specified:

            * model_type (str): the type of model to use. Must be one of
              {"fcnet", "conv"}.
            * layers (list of int or None): the size of the Neural network for
              the policy
            * layer_norm (bool): enable layer normalisation
            * act_fun (tf.nn.*): the activation function to use in the neural
              network

            In addition, the following parameters may be required dependent on
            your choice of model type:

            * ignore_image (bool): observation includes an image but should it
              be ignored. Required if "model_type" is set to "conv".
            * image_height (int): the height of the image in the observation.
              Required if "model_type" is set to "conv".
            * image_width (int): the width of the image in the observation.
              Required if "model_type" is set to "conv".
            * image_channels (int): the number of channels of the image in the
              observation. Required if "model_type" is set to "conv".
            * kernel_sizes (list of int): the kernel size of the neural network
              conv layers for the policy. Required if "model_type" is set to
              "conv".
            * strides (list of int): the kernel size of the neural network conv
              layers for the policy. Required if "model_type" is set to "conv".
            * filters (list of int): the channels of the neural network conv
              layers for the policy. Required if "model_type" is set to "conv".
        """
        self.sess = sess
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.co_space = co_space
        self.verbose = verbose
        self.l2_penalty = l2_penalty
        self.model_params = model_params
        self.num_envs = num_envs

        # Run assertions.
        required = ["model_type", "layers", "layer_norm", "batch_norm",
                    "dropout", "act_fun"]
        not_specified = [s not in model_params.keys() for s in required]
        if any(not_specified):
            raise AssertionError("{} missing from model_params".format(
                ", ".join([param for i, param in enumerate(required)
                           if not_specified[i]])))

        if model_params["model_type"] == "conv":
            required = ["ignore_image", "image_height", "image_width",
                        "image_channels", "kernel_sizes", "strides", "filters"]
            not_specified = [s not in model_params.keys() for s in required]
            if any(not_specified):
                raise AssertionError(
                    "{} missing from model_params. Required if \"model_type\" "
                    "in model_params is set to \"conv\"".format(
                        ", ".join([
                            param for i, param in enumerate(required)
                            if not_specified[i]
                        ])))

        assert model_params["model_type"] in ["conv", "fcnet"], (
            "\"model_type\" in model_params must be one of {\"conv\", "
            "\"fcnet\"}.")

    def initialize(self):
        """Initialize the policy.

        This is used at the beginning of training by the algorithm, after the
        model parameters have been initialized.
        """
        raise NotImplementedError

    def update(self, update_actor=True, **kwargs):
        """Perform a gradient update step.

        Parameters
        ----------
        update_actor : bool
            specifies whether to update the actor policy. The critic policy is
            still updated if this value is set to False.
        """
        raise NotImplementedError

    def get_action(self, obs, context, apply_noise, random_actions, env_num=0):
        """Call the actor methods to compute policy actions.

        Parameters
        ----------
        obs : array_like
            the observation
        context : array_like or None
            the contextual term. Set to None if no context is provided by the
            environment.
        apply_noise : bool
            whether to add Gaussian noise to the output of the actor. Defaults
            to False
        random_actions : bool
            if set to True, actions are sampled randomly from the action space
            instead of being computed by the policy. This is used for
            exploration purposes.
        env_num : int
            the environment number. Used to handle situations when multiple
            parallel environments are being used.

        Returns
        -------
        array_like
            computed action by the policy
        """
        raise NotImplementedError

    def store_transition(self, obs0, context0, action, reward, obs1, context1,
                         done, is_final_step, env_num=0, evaluate=False):
        """Store a transition in the replay buffer.

        Parameters
        ----------
        obs0 : array_like
            the last observation
        context0 : array_like or None
            the last contextual term. Set to None if no context is provided by
            the environment.
        action : array_like
            the action
        reward : float
            the reward
        obs1 : array_like
            the current observation
        context1 : array_like or None
            the current contextual term. Set to None if no context is provided
            by the environment.
        done : float
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
        """
        raise NotImplementedError

    def get_td_map(self):
        """Return dict map for the summary (to be run in the algorithm)."""
        raise NotImplementedError

    def clear_memory(self, env_num):
        """Clear internal memory that is used by the replay buffer."""
        pass

    @staticmethod
    def _get_obs(obs, context, axis=0):
        """Return the processed observation.

        If the contextual term is not None, this will look as follows:

                                    -----------------
                    processed_obs = | obs | context |
                                    -----------------

        Otherwise, this method simply returns the observation.

        Parameters
        ----------
        obs : array_like
            the original observation
        context : array_like or None
            the contextual term. Set to None if no context is provided by the
            environment.
        axis : int
            the axis to concatenate the observations and contextual terms by

        Returns
        -------
        array_like
            the processed observation
        """
        if context is not None:
            context = context.flatten() if axis == 0 else context
            obs = np.concatenate((obs, context), axis=axis)
        return obs

    @staticmethod
    def _get_ob_dim(ob_space, co_space):
        """Return the processed observation dimension.

        If the context space is not None, it is included in the computation of
        this term.

        Parameters
        ----------
        ob_space : gym.spaces.*
            the observation space of the environment
        co_space : gym.spaces.*
            the context space of the environment

        Returns
        -------
        tuple
            the true observation dimension
        """
        ob_dim = ob_space.shape
        if co_space is not None:
            ob_dim = tuple(map(sum, zip(ob_dim, co_space.shape)))
        return ob_dim

    @staticmethod
    def _l2_loss(l2_penalty, scope_name):
        """Compute the L2 regularization penalty.

        Parameters
        ----------
        l2_penalty : float
            L2 regularization penalty
        scope_name : str
            the scope of the trainable variables to regularize

        Returns
        -------
        float
            the overall regularization penalty
        """
        if l2_penalty > 0:
            print("regularizing policy network: L2 = {}".format(l2_penalty))
            regularizer = tf.contrib.layers.l2_regularizer(
                scale=l2_penalty, scope="{}/l2_regularize".format(scope_name))
            l2_loss = tf.contrib.layers.apply_regularization(
                regularizer,
                weights_list=get_trainable_vars(scope_name))
        else:
            # no regularization
            l2_loss = 0

        return l2_loss
