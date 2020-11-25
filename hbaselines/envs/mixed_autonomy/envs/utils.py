"""Script containing utility methods shared amount the environments."""

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
MAX_HEADWAY = 20
# a normalizing term for the vehicle speeds
MAX_SPEED = 5


def get_relative_obs(env, veh_id):
    """Return the relative observation of a vehicle.

    The observation consists of (by index):

    1. the ego speed
    2. the headway
    3. the speed of the leader
    4. the tailway
    5. the speed of the follower

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
    str
        the ID of the follower
    """
    obs = [None for _ in range(5)]

    # used to handle missing observations of adjacent vehicles
    max_speed = env.k.network.max_speed()
    max_length = env.k.network.length()

    # Add the speed of the ego vehicle.
    obs[0] = env.k.vehicle.get_speed(veh_id, 0) / MAX_SPEED

    # Add the speed and bumper-to-bumper headway of leading vehicles.
    leader = env.k.vehicle.get_leader(veh_id)
    if leader in ["", None]:
        # in case leader is not visible
        lead_speed = max_speed / MAX_SPEED
        lead_head = max_length / MAX_HEADWAY
    else:
        lead_speed = env.k.vehicle.get_speed(leader, 0) / MAX_SPEED
        lead_head = env.k.vehicle.get_headway(veh_id, 0) / MAX_HEADWAY
        env.leader.append(leader)

    obs[1] = lead_speed
    obs[2] = lead_head

    # Add the speed and bumper-to-bumper headway of following vehicles.
    follower = env.k.vehicle.get_follower(veh_id)
    if follower in ["", None]:
        # in case follower is not visible
        follow_speed = max_speed / MAX_SPEED
        follow_head = max_length / MAX_HEADWAY
    else:
        follow_speed = env.k.vehicle.get_speed(follower, 0) / MAX_SPEED
        follow_head = env.k.vehicle.get_headway(follower, 0) / MAX_HEADWAY
        env.follower.append(follower)

    obs[3] = follow_speed
    obs[4] = follow_head

    return obs, leader, follower


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
