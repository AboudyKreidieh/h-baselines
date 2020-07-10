"""Utility methods for HAC environments."""


def check_validity(model_name,
                   initial_state_space,
                   max_actions,
                   timesteps_per_action):
    """Ensure environment configurations were properly entered.

    This is done via a sequence of assertions.

    Parameters
    ----------
    model_name : str
        name of the Mujoco model file
    initial_state_space : list of (float, float)
        bounds for the initial values for all elements in the state space.
        This is achieved during the reset procedure.
    max_actions : int
        maximum number of atomic actions. This will typically be
        flags.time_scale**(flags.layers).
    timesteps_per_action : int
        number of time steps per atomic action
    """
    # Ensure model file is an ".xml" file
    assert model_name[-4:] == ".xml", "Mujoco model must be an \".xml\" file"

    for i in range(len(initial_state_space)):
        assert initial_state_space[i][1] >= initial_state_space[i][0], \
            "In initial state space, upper bound must be >= lower bound"

    # Ensure max action and timesteps_per_action are positive integers
    assert max_actions > 0, "Max actions should be a positive integer"

    assert timesteps_per_action > 0, \
        "Timesteps per action should be a positive integer"
