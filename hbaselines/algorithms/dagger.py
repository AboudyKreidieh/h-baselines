"""Script contain the DAgger training algorithm.

See: https://arxiv.org/pdf/1011.0686.pdf
"""
import tensorflow as tf


# =========================================================================== #
#                   Policy parameters for FeedForwardPolicy                   #
# =========================================================================== #

FEEDFORWARD_PARAMS = dict(
    # the max number of transitions to store
    buffer_size=200000,
    # the size of the batch for learning the policy
    batch_size=128,
    # the learning rate of the policy
    learning_rate=3e-4,
    # enable layer normalization
    layer_norm=False,
    # the size of the neural network for the policy
    layers=[256, 256],
    # the activation function to use in the neural network
    act_fun=tf.nn.relu,
    # specifies whether to use the huber distance function as the loss
    # function. If set to False, the mean-squared error metric is used instead.
    # Only applies to stochastic policies.
    use_huber=False,
    # specifies whether the policies are stochastic or deterministic
    stochastic=False,
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
    # the value that the intrinsic reward should be scaled by
    intrinsic_reward_scale=1,
    # specifies whether the goal issued by the higher-level policies is meant
    # to be a relative or absolute goal, i.e. specific state or change in state
    relative_goals=False,
))


class DAggerAlgorithm(object):
    """DAgger training algorithm."""

    pass
