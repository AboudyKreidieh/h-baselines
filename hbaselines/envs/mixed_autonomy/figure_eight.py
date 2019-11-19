"""Figure eight example."""
from flow.envs import AccelEnv
from flow.envs.multiagent import MultiAgentAccelEnv
from flow.networks import FigureEightNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import IDMController, ContinuousRouter, RLController
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS


# desired velocity for all vehicles in the network, in m/s
TARGET_VELOCITY = 20
# maximum acceleration for autonomous vehicles, in m/s^2
MAX_ACCEL = 3
# maximum deceleration for autonomous vehicles, in m/s^2
MAX_DECEL = 3


def get_flow_params(num_automated=1,
                    horizon=1500,
                    simulator="traci",
                    multiagent=False):
    """Return the flow-specific parameters of the figure eight network.

    This network consists of two rings, placed at opposite ends of the network,
    and connected by an intersection with road segments of length equal to the
    diameter of the rings. If two vehicles attempt to cross the intersection
    from opposing directions, the dynamics of these vehicles are constrained by
    right-of-way rules provided by the simulator.

    We consider a figure eight with a ring radius of 30 m and total length of
    402m. The network contains a total of 14 vehicles. We study various levels
    of mixed-autonomy.

    This benchmark is adapted from the following article:

    Vinitsky, Eugene, et al. "Benchmarks for reinforcement learning in
    mixed-autonomy traffic." Conference on Robot Learning. 2018.

    Parameters
    ----------
    num_automated : int
        number of automated (RL) vehicles
    horizon : int
        time horizon of a single rollout
    simulator : str
        the simulator used, one of {'traci', 'aimsun'}
    multiagent : bool
        whether the automated vehicles are via a single-agent policy or a
        shared multi-agent policy with the actions of individual vehicles
        assigned by a separate policy call

    Returns
    -------
    dict
        flow-related parameters, consisting of the following keys:

        * exp_tag: name of the experiment
        * env_name: environment class of the flow environment the experiment
          is running on. (note: must be in an importable module.)
        * network: network class the experiment uses.
        * simulator: simulator that is used by the experiment (e.g. aimsun)
        * sim: simulation-related parameters (see flow.core.params.SimParams)
        * env: environment related parameters (see flow.core.params.EnvParams)
        * net: network-related parameters (see flow.core.params.NetParams and
          the network's documentation or ADDITIONAL_NET_PARAMS component)
        * veh: vehicles to be placed in the network at the start of a rollout
          (see flow.core.params.VehicleParams)
        * initial (optional): parameters affecting the positioning of vehicles
          upon initialization/reset (see flow.core.params.InitialConfig)
        * tls (optional): traffic lights to be introduced to specific nodes
          (see flow.core.params.TrafficLightParams)

    Raises
    ------
    AssertionError
        if the `num_automated` parameter is a value other than 1, 2, 7, or 14
    """
    assert num_automated in [1, 2, 7, 14], \
        "num_automated must be one of [1, 2, 7 14]"

    # We evenly distribute the autonomous vehicles in between the human-driven
    # vehicles in the network.
    num_human = 14 - num_automated
    human_per_automated = int(num_human/num_automated)

    vehicles = VehicleParams()
    for i in range(num_automated):
        vehicles.add(
            veh_id='human_{}'.format(i),
            acceleration_controller=(IDMController, {
                'noise': 0.2
            }),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="obey_safe_speed",
                decel=1.5,
            ),
            num_vehicles=human_per_automated)
        vehicles.add(
            veh_id='rl_{}'.format(i),
            acceleration_controller=(RLController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="obey_safe_speed",
                accel=MAX_ACCEL,
                decel=MAX_DECEL,
            ),
            num_vehicles=1)

    return dict(
        # name of the experiment
        exp_tag='figure_eight',

        # name of the flow environment the experiment is running on
        env_name=MultiAgentAccelEnv if multiagent else AccelEnv,

        # name of the network class the experiment is running on
        network=FigureEightNetwork,

        # simulator that is used by the experiment
        simulator=simulator,

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.1,
            render=False,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=horizon,
            additional_params={
                'target_velocity': TARGET_VELOCITY,
                'max_accel': MAX_ACCEL,
                'max_decel': MAX_DECEL,
                'sort_vehicles': False
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            additional_params=ADDITIONAL_NET_PARAMS.copy(),
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon init/reset
        # (see flow.core.params.InitialConfig)
        initial=InitialConfig(),
    )
