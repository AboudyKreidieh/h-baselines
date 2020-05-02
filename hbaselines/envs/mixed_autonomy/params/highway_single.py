"""Multi-agent highway with ramps example.

Trains a non-constant number of agents, all sharing the same policy, on the
highway with ramps network.
"""
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
from flow.core.params import VehicleParams
from flow.core.params import SumoParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import SumoLaneChangeParams
from flow.networks.highway import HighwayNetwork
from flow.networks.highway import ADDITIONAL_NET_PARAMS

from hbaselines.envs.mixed_autonomy.envs import AVOpenEnv
from hbaselines.envs.mixed_autonomy.envs import AVOpenMultiAgentEnv


def get_flow_params(evaluate=False, multiagent=False):
    """Return the flow-specific parameters of the single lane highway network.

    Parameters
    ----------
    evaluate : bool
        whether to compute the evaluation reward
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
    """
    # SET UP PARAMETERS FOR THE SIMULATION

    # number of steps per rollout
    HORIZON = 1800
    # inflow rate on the highway in vehicles per hour
    HIGHWAY_INFLOW_RATE = 2000
    # percentage of autonomous vehicles compared to human vehicles on highway
    PENETRATION_RATE = 0.1

    # SET UP PARAMETERS FOR THE NETWORK

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params.update({
        # length of the highway
        "length": 2500,
        # number of lanes
        "lanes": 1,
        # speed limit for all edges
        "speed_limit": 30,
        # number of edges to divide the highway into
        "num_edges": 2
    })

    # CREATE VEHICLE TYPES AND INFLOWS

    vehicles = VehicleParams()
    inflows = InFlows()

    # human vehicles
    vehicles.add(
        "human",
        num_vehicles=0,
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode="strategic",
        ),
        car_following_params=SumoCarFollowingParams(
            min_gap=0,
        ),
        acceleration_controller=(IDMController, {
            "a": 0.3,
            "b": 2.0,
            "noise": 0.5,
        }),
    )

    inflows.add(
        veh_type="human",
        edge="highway_0",
        vehs_per_hour=int(HIGHWAY_INFLOW_RATE * (1 - PENETRATION_RATE)),
        depart_lane="free",
        depart_speed=15,
        name="idm_highway_inflow"
    )

    # automated vehicles
    if PENETRATION_RATE > 0.0:
        vehicles.add(
            "rl",
            num_vehicles=0,
            acceleration_controller=(RLController, {}),
        )

        inflows.add(
            veh_type="rl",
            edge="highway_0",
            vehs_per_hour=int(HIGHWAY_INFLOW_RATE * PENETRATION_RATE),
            depart_lane="free",
            depart_speed=15,
            name="rl_highway_inflow"
        )

    # SET UP THE FLOW PARAMETERS

    return dict(
        # name of the experiment
        exp_tag='highway-single',

        # name of the flow environment the experiment is running on
        env_name=AVOpenMultiAgentEnv if multiagent else AVOpenEnv,

        # name of the network class the experiment is running on
        network=HighwayNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            evaluate=evaluate,
            horizon=HORIZON,
            warmup_steps=50,
            sims_per_step=2,
            additional_params={
                "max_accel": 1,
                "max_decel": 1,
                "target_velocity": 30,
                "penalty_type": "time_headway",
                "penalty": 1,
                "inflows": None,
                "rl_penetration": PENETRATION_RATE,
                "num_rl": 10,
                "ghost_length": 500,
            }
        ),

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.5,
            render=False,
            restart_instance=True
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflows,
            additional_params=additional_net_params
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon init/reset
        # (see flow.core.params.InitialConfig)
        initial=InitialConfig(),
    )
