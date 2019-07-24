"""A compilation of contextual rewards functions.

These reward functions are used either to augment the environment rewards with
a specific goal-directed reward, or to assign rewards to lower-level policies.

All reward functions here return new rewards and discounts.
"""
import numpy as np


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
