"""A compilation of contextual rewards functions.

These reward functions are used either to augment the environment rewards with
a specific goal-conditioned reward, or to assign rewards to lower-level
policies.

All reward functions here return new rewards and discounts.
"""
import numpy as np


def negative_distance(states,
                      next_states,
                      goals,
                      state_scales=1.0,
                      goal_scales=1.0,
                      reward_scales=1.0,
                      state_indices=None,
                      goal_indices=None,
                      relative_context=False,
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
    state_indices : list of int
        list of state indices to select.
    goal_indices : list of int
        list of goal indices to select.
    relative_context : bool
        if True, then the goal is a relative goal, i.e. the requested position
        is the current position plus the requested goal.
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

    # Apply the L2 norm.
    dist = np.sum(sq_dists, -1)
    dist = np.sqrt(dist + epsilon)

    bonus = float(dist < bonus_epsilon)
    dist *= reward_scales

    return bonus + offset - dist
