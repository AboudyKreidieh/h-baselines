"""Script containing utility methods shared amount the environments."""
import numpy as np

# These edges have an extra lane that RL vehicles do not traverse (since they
# do not change lanes). We as a result ignore their first lane computing
# per-lane states.
EXTRA_LANE_EDGES = [
    "119257908#1-AddedOnRampEdge",
    "119257908#1-AddedOffRampEdge",
    ":119257908#1-AddedOnRampNode_0",
    ":119257908#1-AddedOffRampNode_0",
    "119257908#3",
]
# a normalizing term for the vehicle headways
MAX_HEADWAY = 20.0
# a normalizing term for the vehicle speeds
MAX_SPEED = 1.0


def get_relative_obs(env, veh_id):
    """Return the relative observation of a vehicle.

    The observation consists of (by index):

    1. the ego speed
    2. the headway
    3. the speed of the leader

    This also adds the leaders and followers to the vehicle class for
    visualization purposes.

    Parameters
    ----------
    env : flow.Env
        the environment
    veh_id : str
        the ID of the vehicle whose observation is meant to be returned

    Returns
    -------
    array_like
        the observation
    str
        the ID of the leader
    """
    obs = [None for _ in range(3)]

    # Add the speed of the ego vehicle.
    obs[0] = env.k.vehicle.get_speed(veh_id, 0) / MAX_SPEED

    # Add the speed and bumper-to-bumper headway of leading vehicles.
    leader = env.k.vehicle.get_leader(veh_id)
    if leader in ["", None]:
        # in case leader is not visible
        lead_speed = 10.0
        lead_head = 5.0
    else:
        lead_speed = env.k.vehicle.get_speed(leader, 0) / MAX_SPEED
        lead_head = min(env.k.vehicle.get_headway(veh_id, 0) / MAX_HEADWAY, 5.)

    obs[1] = lead_speed
    obs[2] = lead_head

    return obs, leader


def update_rl_veh(env,
                  rl_queue,
                  rl_veh,
                  removed_veh,
                  control_range,
                  num_rl,
                  rl_ids):
    """Update the RL lists of controllable, entering, and exiting vehicles.

    Used by the open environments.

    Parameters
    ----------
    env : flow.Env
        the environment class
    rl_queue : collections.dequeue
        the queue of vehicles that are not controllable yet
    rl_veh : list of str
        the list of current controllable vehicles, sorted by their positions
    removed_veh : list of str
        the list of RL vehicles that passed the control range
    control_range : (float, float)
        the control range (min_pos, max_pos)
    num_rl : int
        the maximum number of vehicles to control at any given time
    rl_ids : list of str or iterator
        the RL IDs to add to the different attributes

    Returns
    -------
    collections.dequeue
        the updated rl_queue term
    list of str
        the updated rl_veh term
    list of str
        the updated removed_veh term
    """
    # Add rl vehicles that just entered the network into the rl queue.
    for veh_id in rl_ids:
        if veh_id not in list(rl_queue) + rl_veh + removed_veh:
            rl_queue.append(veh_id)

    # Remove rl vehicles that exited the controllable range of the network.
    for veh_id in rl_veh:
        if env.k.vehicle.get_x_by_id(veh_id) > control_range[1] \
                or veh_id not in env.k.vehicle.get_rl_ids():
            removed_veh.append(veh_id)
            rl_veh.remove(veh_id)

    # Fill up rl_veh until they are enough controlled vehicles.
    while len(rl_queue) > 0 and len(rl_veh) < num_rl:
        # Ignore vehicles that are in the ghost edges.
        if env.k.vehicle.get_x_by_id(rl_queue[0]) < control_range[0]:
            break

        rl_id = rl_queue.popleft()
        veh_pos = env.k.vehicle.get_x_by_id(rl_id)

        # Add the vehicle if it is within the control range.
        if veh_pos < control_range[1]:
            rl_veh.append(rl_id)

    return rl_queue, rl_veh, removed_veh


def get_lane(env, veh_id):
    """Return a processed lane number."""
    lane = env.k.vehicle.get_lane(veh_id)
    edge = env.k.vehicle.get_edge(veh_id)
    return lane if edge not in EXTRA_LANE_EDGES else lane - 1


def v_eq_function(v, *args):
    """Return the error between the desired and actual equivalent gap."""
    num_vehicles, length = args

    # maximum gap in the presence of one rl vehicle
    s_eq_max = (length - num_vehicles * 5) / num_vehicles

    v0 = 30
    s0 = 2
    tau = 1
    gamma = 4

    error = s_eq_max - (s0 + v * tau) * (1 - (v / v0) ** gamma) ** -0.5

    return error


def get_rl_accel(accel, vel, max_accel, dt):
    """Compute the RL acceleration from the desired acceleration.

    We reduce the decelerations at smaller speeds to smoothen the effects.

    Parameters
    ----------
    accel : array_like or dict
        the RL actions
    vel : array_like
        the speed of the RL vehicles
    max_accel : float
        scaling factor for the AV accelerations, in m/s^2
    dt : float
        seconds per simulation step

    Returns
    -------
    array_like
        the updated acceleration values
    """
    # for multi-agent environments
    if isinstance(accel, dict):
        accel = [accel[key][0] for key in accel.keys()]

    # Scale to the range of accelerations.
    accel = max_accel * np.array(accel)

    # Redefine if below a speed threshold so that all actions result in
    # non-negative desired speeds.
    for i in range(len(vel)):
        ac_range = 2. * max_accel
        if vel[i] < 0.5 * ac_range * dt:
            accel[i] += 0.5 * ac_range - vel[i] / dt

    return accel
