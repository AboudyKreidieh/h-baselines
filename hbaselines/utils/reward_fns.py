"""A compilation of contextual rewards functions.

These reward functions are used either to augment the environment rewards with
a specific goal-directed reward, or to assign rewards to lower-level policies.

All reward functions here return new rewards and discounts.
"""
import numpy as np
import tensorflow as tf


def summarize_stats(stats):
    """Summarize a dictionary of variables.

  Args:
    stats: a dictionary of {name: tensor} to compute stats over.
  """
    for name, stat in stats.items():
        mean = tf.reduce_mean(stat)
        tf.summary.scalar('mean_%s' % name, mean)
        tf.summary.scalar('max_%s' % name, tf.reduce_max(stat))
        tf.summary.scalar('min_%s' % name, tf.reduce_min(stat))
        std = tf.sqrt(
            tf.reduce_mean(tf.square(stat)) - tf.square(mean) + 1e-10)
        tf.summary.scalar('std_%s' % name, std)
        tf.summary.histogram(name, stat)


def index_states(states, indices):
    """Return indexed states.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    indices: (a list of Numpy integer array) Indices of states dimensions
      to be mapped.
  Returns:
    A [batch_size, num_indices] Tensor representing the batch of indexed
    states.
  """
    if indices is None:
        return states
    indices = tf.constant(indices, dtype=tf.int32)
    return tf.gather(states, indices=indices, axis=1)


def record_tensor(tensor, indices, stats, name='states'):
    """Record specified tensor dimensions into stats.

  Args:
    tensor: A [batch_size, num_dims] Tensor.
    indices: (a list of integers) Indices of dimensions to record.
    stats: A dictionary holding stats.
    name: (string) Name of tensor.
  """
    if indices is None:
        indices = range(tensor.shape.as_list()[1])
    for index in indices:
        stats['%s_%02d' % (name, index)] = tensor[:, index]


def potential_rewards(states,
                      actions,
                      rewards,
                      next_states,
                      contexts,
                      gamma=1.0,
                      reward_fn=None):
    """Return the potential-based rewards.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    gamma: Reward discount.
    reward_fn: A reward function.
  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
    del actions  # unused args
    gamma = tf.to_float(gamma)
    rewards_tp1, discounts = reward_fn(None, None, rewards, next_states,
                                       contexts)
    rewards, _ = reward_fn(None, None, rewards, states, contexts)
    return -rewards + gamma * rewards_tp1, discounts


def timed_rewards(states,
                  actions,
                  rewards,
                  next_states,
                  contexts,
                  reward_fn=None,
                  dense=False,
                  timer_index=-1):
    """Return the timed rewards.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    reward_fn: A reward function.
    dense: (boolean) Provide dense rewards or sparse rewards at time = 0.
    timer_index: (integer) The context list index that specifies timer.
  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
    assert contexts[timer_index].get_shape().as_list()[1] == 1
    timers = contexts[timer_index][:, 0]
    rewards, discounts = reward_fn(states, actions, rewards, next_states,
                                   contexts)
    terminates = tf.to_float(timers <= 0)  # if terminate set 1, else set 0
    for _ in range(rewards.shape.ndims - 1):
        terminates = tf.expand_dims(terminates, axis=-1)
    if not dense:
        rewards *= terminates  # if terminate, return rewards, else return 0
    discounts *= (tf.to_float(1.0) - terminates)
    return rewards, discounts


def reset_rewards(states,
                  actions,
                  rewards,
                  next_states,
                  contexts,
                  reset_index=0,
                  reset_state=None,
                  reset_reward_function=None,
                  include_forward_rewards=True,
                  include_reset_rewards=True):
    """Returns the rewards for a forward/reset agent.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    reset_index: (integer) The context list index that specifies reset.
    reset_state: Reset state.
    reset_reward_function: Reward function for reset step.
    include_forward_rewards: Include the rewards from the forward pass.
    include_reset_rewards: Include the rewards from the reset pass.

  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
    reset_state = tf.constant(
        reset_state, dtype=next_states.dtype, shape=next_states.shape)
    reset_states = tf.expand_dims(reset_state, 0)

    def true_fn():
        if include_reset_rewards:
            return reset_reward_function(states, actions, rewards, next_states,
                                         [reset_states] + contexts[1:])
        else:
            return tf.zeros_like(rewards), tf.ones_like(rewards)

    def false_fn():
        if include_forward_rewards:
            return plain_rewards(states, actions, rewards, next_states,
                                 contexts)
        else:
            return tf.zeros_like(rewards), tf.ones_like(rewards)

    rewards, discounts = tf.cond(
        tf.cast(contexts[reset_index][0, 0], dtype=tf.bool), true_fn, false_fn)
    return rewards, discounts


def tanh_similarity(states,
                    actions,
                    rewards,
                    next_states,
                    contexts,
                    mse_scale=1.0,
                    state_scales=1.0,
                    goal_scales=1.0,
                    summarize=False):
    """Returns the similarity between next_states and contexts using tanh and
    mse.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    mse_scale: A float, to scale mse before tanh.
    state_scales: multiplicative scale for (next) states. A scalar or 1D
    tensor,
      must be broadcastable to number of state dimensions.
    goal_scales: multiplicative scale for contexts. A scalar or 1D tensor,
      must be broadcastable to number of goal dimensions.
    summarize: (boolean) enable summary ops.


  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
    del states, actions, rewards  # Unused
    mse = tf.reduce_mean(tf.squared_difference(next_states * state_scales,
                                               contexts[0] * goal_scales), -1)
    tanh = tf.tanh(mse_scale * mse)
    if summarize:
        with tf.name_scope('RewardFn/'):
            tf.summary.scalar('mean_mse', tf.reduce_mean(mse))
            tf.summary.histogram('mse', mse)
            tf.summary.scalar('mean_tanh', tf.reduce_mean(tanh))
            tf.summary.histogram('tanh', tanh)
    rewards = tf.to_float(1 - tanh)
    return rewards, tf.ones_like(rewards)


def negative_mse(states,
                 actions,
                 rewards,
                 next_states,
                 contexts,
                 state_scales=1.0,
                 goal_scales=1.0,
                 summarize=False):
    """Returns the negative mean square error between next_states and contexts.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    state_scales: multiplicative scale for (next) states. A scalar or 1D
    tensor,
      must be broadcastable to number of state dimensions.
    goal_scales: multiplicative scale for contexts. A scalar or 1D tensor,
      must be broadcastable to number of goal dimensions.
    summarize: (boolean) enable summary ops.

  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
    del states, actions, rewards  # Unused
    mse = tf.reduce_mean(tf.squared_difference(next_states * state_scales,
                                               contexts[0] * goal_scales), -1)
    if summarize:
        with tf.name_scope('RewardFn/'):
            tf.summary.scalar('mean_mse', tf.reduce_mean(mse))
            tf.summary.histogram('mse', mse)
    rewards = tf.to_float(-mse)
    return rewards, tf.ones_like(rewards)


# ==========================================================================+ #
# ==========================================================================+ #
# ==========================================================================+ #


def negative_distance(states,
                      next_states,
                      goals,
                      state_scales=1.0,
                      goal_scales=1.0,
                      reward_scales=1.0,
                      weight_vector=None,
                      termination_epsilon=1e-4,
                      state_indices=None,
                      goal_indices=None,
                      relative_context=False,
                      diff=False,
                      norm='L2',
                      epsilon=1e-10,
                      bonus_epsilon=0.,
                      offset=0.0):
    """Return the negative euclidean distance between next_states and goals.

    Parameters
    ----------
    states : array_like
        A (num_state_dims,) array representing a batch of states.
    next_states : array_like
        A (num_state_dims,) array representing a batch of next states.
    goals : array_like
        A (num_context_dims,) array representing a batch of contexts.
    state_scales : float
        multiplicative scale for (next) states
    goal_scales : float
        multiplicative scale for goals
    reward_scales : float
        multiplicative scale for rewards
    weight_vector : float or array_like
        The weighting vector, broadcastable to `next_states`.
    termination_epsilon : float
        terminate if dist is less than this quantity.
    state_indices : list of int
        list of state indices to select.
    goal_indices : list of int
        list of goal indices to select.
    relative_context : bool
        if True, then the goal is a relative goal, i.e. the requested position
        is the current position plus the requested goal.
    diff : bool
        specifies whether to reward the current distance or the difference in
        consecutive differences
    norm : {"L1", "L2"}
        specifies whether to use L1 or L2 normalization.
    epsilon : float
        small offset to ensure non-negative/zero distance.
    bonus_epsilon : float
        if the distance is below this value, an additional bonus is added to
        the output reward
    offset : float
        an offset value that is added to the returned reward

    Returns
    -------
    array_like
        the rewards for each element in the batch
    array_like
        the discounts for each element in the batch
    """
    # Get the indexed versions of the states and goals.
    if state_indices is not None:
        states = states[state_indices]
        next_states = next_states[state_indices]
    if goal_indices is not None:
        goals = goals[goal_indices]

    # Check for relative context.
    if relative_context:
        goals = states + goals

    sq_dists = np.square(next_states * state_scales - goals * goal_scales)
    old_sq_dists = np.square(states * state_scales - goals * goal_scales)

    if weight_vector is not None:
        sq_dists *= weight_vector
        old_sq_dists *= weight_vector

    # Apply either the L1 or L2 norm.
    if norm == 'L1':
        dist = np.sqrt(sq_dists + epsilon)
        old_dist = np.sqrt(old_sq_dists + epsilon)
        dist = np.sum(dist, -1)
        old_dist = np.sum(old_dist, -1)
    elif norm == 'L2':
        dist = np.sum(sq_dists, -1)
        old_dist = np.sum(old_sq_dists, -1)
        dist = np.sqrt(dist + epsilon)
        old_dist = np.sqrt(old_dist + epsilon)
    else:
        raise NotImplementedError(norm)

    discounts = float(dist > termination_epsilon)
    bonus = float(dist < bonus_epsilon)
    dist *= reward_scales
    old_dist *= reward_scales

    if diff:
        return bonus + offset + old_dist - dist, discounts
    else:
        return bonus + offset - dist, discounts


# ==========================================================================+ #
# ==========================================================================+ #
# ==========================================================================+ #


def cosine_similarity(states,
                      actions,
                      rewards,
                      next_states,
                      contexts,
                      state_scales=1.0,
                      goal_scales=1.0,
                      reward_scales=1.0,
                      normalize_states=True,
                      normalize_goals=True,
                      weight_index=None,
                      weight_vector=None,
                      summarize=False,
                      state_indices=None,
                      goal_indices=None,
                      offset=0.0):
    """Returns the cosine similarity between next_states - states and contexts.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    state_scales: multiplicative scale for (next) states. A scalar or 1D
    tensor,
      must be broadcastable to number of state dimensions.
    goal_scales: multiplicative scale for goals. A scalar or 1D tensor,
      must be broadcastable to number of goal dimensions.
    reward_scales: multiplicative scale for rewards. A scalar or 1D tensor,
      must be broadcastable to number of reward dimensions.
    weight_index: (integer) The context list index that specifies weight.
    weight_vector: (a number or a list or Numpy array) The weighting vector,
      broadcastable to `next_states`.
    summarize: (boolean) enable summary ops.
    termination_epsilon: terminate if dist is less than this quantity.
    state_indices: (a list of integers) list of state indices to select.
    goal_indices: (a list of integers) list of goal indices to select.
    vectorize: Return a vectorized form.
    norm: L1 or L2.
    epsilon: small offset to ensure non-negative/zero distance.

  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
    del actions, rewards  # Unused
    stats = {}
    record_tensor(next_states, state_indices, stats, 'next_states')
    states = index_states(states, state_indices)
    next_states = index_states(next_states, state_indices)
    goals = index_states(contexts[0], goal_indices)

    if weight_vector is not None:
        goals *= tf.convert_to_tensor(weight_vector, dtype=next_states.dtype)
    if weight_index is not None:
        weights = tf.abs(index_states(contexts[0], weight_index))
        goals *= weights

    direction_vec = next_states - states
    if normalize_states:
        direction_vec = tf.nn.l2_normalize(direction_vec, -1)
    goal_vec = goals
    if normalize_goals:
        goal_vec = tf.nn.l2_normalize(goal_vec, -1)

    similarity = tf.reduce_sum(goal_vec * direction_vec, -1)
    discounts = tf.ones_like(similarity)
    return offset + tf.to_float(similarity), tf.to_float(discounts)


def diff_distance(states,
                  actions,
                  rewards,
                  next_states,
                  contexts,
                  state_scales=1.0,
                  goal_scales=1.0,
                  reward_scales=1.0,
                  weight_index=None,
                  weight_vector=None,
                  summarize=False,
                  termination_epsilon=1e-4,
                  state_indices=None,
                  goal_indices=None,
                  norm='L2',
                  epsilon=1e-10):
    """Return the difference in euclidean distance between states/next_states
    and contexts.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    state_scales: multiplicative scale for (next) states. A scalar or 1D
    tensor,
      must be broadcastable to number of state dimensions.
    goal_scales: multiplicative scale for goals. A scalar or 1D tensor,
      must be broadcastable to number of goal dimensions.
    reward_scales: multiplicative scale for rewards. A scalar or 1D tensor,
      must be broadcastable to number of reward dimensions.
    weight_index: (integer) The context list index that specifies weight.
    weight_vector: (a number or a list or Numpy array) The weighting vector,
      broadcastable to `next_states`.
    summarize: (boolean) enable summary ops.
    termination_epsilon: terminate if dist is less than this quantity.
    state_indices: (a list of integers) list of state indices to select.
    goal_indices: (a list of integers) list of goal indices to select.
    vectorize: Return a vectorized form.
    norm: L1 or L2.
    epsilon: small offset to ensure non-negative/zero distance.

  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
    del actions, rewards  # Unused
    stats = {}
    record_tensor(next_states, state_indices, stats, 'next_states')
    next_states = index_states(next_states, state_indices)
    states = index_states(states, state_indices)
    goals = index_states(contexts[0], goal_indices)
    next_sq_dists = tf.squared_difference(next_states * state_scales,
                                          goals * goal_scales)
    sq_dists = tf.squared_difference(states * state_scales,
                                     goals * goal_scales)
    record_tensor(sq_dists, None, stats, 'sq_dists')
    if weight_vector is not None:
        next_sq_dists *= tf.convert_to_tensor(weight_vector,
                                              dtype=next_states.dtype)
        sq_dists *= tf.convert_to_tensor(weight_vector,
                                         dtype=next_states.dtype)
    if weight_index is not None:
        next_sq_dists *= contexts[weight_index]
        sq_dists *= contexts[weight_index]
    if norm == 'L1':
        next_dist = tf.sqrt(next_sq_dists + epsilon)
        dist = tf.sqrt(sq_dists + epsilon)
        next_dist = tf.reduce_sum(next_dist, -1)
        dist = tf.reduce_sum(dist, -1)
    elif norm == 'L2':
        next_dist = tf.reduce_sum(next_sq_dists, -1)
        next_dist = tf.sqrt(
            next_dist + epsilon)  # tf.gradients fails when tf.sqrt(-0.0)
        dist = tf.reduce_sum(sq_dists, -1)
        dist = tf.sqrt(dist + epsilon)  # tf.gradients fails when tf.sqrt(-0.0)
    else:
        raise NotImplementedError(norm)
    discounts = next_dist > termination_epsilon
    if summarize:
        with tf.name_scope('RewardFn/'):
            tf.summary.scalar('mean_dist', tf.reduce_mean(dist))
            tf.summary.histogram('dist', dist)
            summarize_stats(stats)
    diff = dist - next_dist
    diff *= reward_scales
    return tf.to_float(diff), tf.to_float(discounts)


def binary_indicator(states,
                     actions,
                     rewards,
                     next_states,
                     contexts,
                     termination_epsilon=1e-4,
                     offset=0,
                     epsilon=1e-10,
                     state_indices=None,
                     summarize=False):
    """Returns 0/1 by checking if next_states and contexts overlap.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    termination_epsilon: terminate if dist is less than this quantity.
    offset: Offset the rewards.
    epsilon: small offset to ensure non-negative/zero distance.

  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
    del states, actions  # unused args
    next_states = index_states(next_states, state_indices)
    dist = tf.reduce_sum(tf.squared_difference(next_states, contexts[0]), -1)
    dist = tf.sqrt(dist + epsilon)
    discounts = dist > termination_epsilon
    rewards = tf.logical_not(discounts)
    rewards = tf.to_float(rewards) + offset
    return tf.to_float(rewards), tf.ones_like(
        tf.to_float(discounts))  # tf.to_float(discounts)


def plain_rewards(states, actions, rewards, next_states, contexts):
    """Returns the given rewards.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.

  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
    del states, actions, next_states, contexts  # Unused
    return rewards, tf.ones_like(rewards)


def ctrl_rewards(states,
                 actions,
                 rewards,
                 next_states,
                 contexts,
                 reward_scales=1.0):
    """Returns the negative control cost.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    reward_scales: multiplicative scale for rewards. A scalar or 1D tensor,
      must be broadcastable to number of reward dimensions.

  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
    del states, rewards, contexts  # Unused
    if actions is None:
        rewards = tf.to_float(tf.zeros(shape=next_states.shape[:1]))
    else:
        rewards = -tf.reduce_sum(tf.square(actions), axis=1)
        rewards *= reward_scales
        rewards = tf.to_float(rewards)
    return rewards, tf.ones_like(rewards)


def diff_rewards(states,
                 actions,
                 rewards,
                 next_states,
                 contexts,
                 state_indices=None,
                 goal_index=0):
    """Return (next_states - goals) as a batched vector reward."""
    del states, rewards, actions  # Unused
    if state_indices is not None:
        next_states = index_states(next_states, state_indices)
    rewards = tf.to_float(next_states - contexts[goal_index])
    return rewards, tf.ones_like(rewards)


def state_rewards(states,
                  actions,
                  rewards,
                  next_states,
                  contexts,
                  weight_index=None,
                  state_indices=None,
                  weight_vector=1.0,
                  offset_vector=0.0,
                  summarize=False):
    """Return the rewards that are linear mapping of next_states.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    weight_index: (integer) Index of contexts lists that specify weighting.
    state_indices: (a list of Numpy integer array) Indices of states dimensions
      to be mapped.
    weight_vector: (a number or a list or Numpy array) The weighting vector,
      broadcastable to `next_states`.
    offset_vector: (a number or a list of Numpy array) The off vector.
    summarize: (boolean) enable summary ops.

  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
    del states, actions, rewards  # unused args
    stats = {}
    record_tensor(next_states, state_indices, stats)
    next_states = index_states(next_states, state_indices)
    weight = tf.constant(
        weight_vector, dtype=next_states.dtype, shape=next_states[0].shape)
    weights = tf.expand_dims(weight, 0)
    offset = tf.constant(
        offset_vector, dtype=next_states.dtype, shape=next_states[0].shape)
    offsets = tf.expand_dims(offset, 0)
    if weight_index is not None:
        weights *= contexts[weight_index]
    rewards = tf.to_float(
        tf.reduce_sum(weights * (next_states + offsets), axis=1))
    if summarize:
        with tf.name_scope('RewardFn/'):
            summarize_stats(stats)
    return rewards, tf.ones_like(rewards)
