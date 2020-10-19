"""Single lane highway example."""
import os

from flow.controllers import IDMController
from flow.controllers import RLController
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
from flow.core.params import VehicleParams
from flow.core.params import SumoParams
from flow.core.params import SumoCarFollowingParams
from flow.networks.highway import HighwayNetwork
from flow.networks.highway import ADDITIONAL_NET_PARAMS

from hbaselines.envs.mixed_autonomy.envs import AVOpenEnv
from hbaselines.envs.mixed_autonomy.envs import AVOpenMultiAgentEnv
from hbaselines.envs.mixed_autonomy.envs.imitation import AVOpenImitationEnv
import hbaselines.config as hbaselines_config

# the speed of entering vehicles
TRAFFIC_SPEED = 24.1
# the speed limit in the ghost edge
END_SPEED = 6.0
# inflow rate on the highway in vehicles per hour
TRAFFIC_FLOW = 2215
# number of steps per rollout
HORIZON = 1500
# percentage of autonomous vehicles compared to human vehicles on highway
PENETRATION_RATE = 1/12
# whether to include noise in the environment
INCLUDE_NOISE = True
# range for the inflows allowed in the network. If set to None, the inflows are
# not modified from their initial value.
INFLOWS = [1000, 2000]
# the path to the warmup files to initialize a network
WARMUP_PATH = os.path.join(
    hbaselines_config.PROJECT_PATH, "experiments/warmup/highway")


def get_flow_params(fixed_boundary,
                    stopping_penalty,
                    acceleration_penalty,
                    use_follower_stopper,
                    evaluate=False,
                    multiagent=False,
                    imitation=False):
    """Return the flow-specific parameters of the single lane highway network.

    Parameters
    ----------
    fixed_boundary : bool
        specifies whether the boundary conditions update in between resets
    stopping_penalty : bool
        whether to include a stopping penalty
    acceleration_penalty : bool
        whether to include a regularizing penalty for accelerations by the AVs
    use_follower_stopper : bool
        whether to use the follower-stopper controller for the AVs
    evaluate : bool
        whether to compute the evaluation reward
    multiagent : bool
        whether the automated vehicles are via a single-agent policy or a
        shared multi-agent policy with the actions of individual vehicles
        assigned by a separate policy call
    imitation : bool
        whether to use the imitation environment

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
    # steps to run before the agent is allowed to take control (set to lower
    # value during testing)
    if WARMUP_PATH is not None:
        warmup_steps = 0
    else:
        warmup_steps = 50 if os.environ.get("TEST_FLAG") else 500

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params.update({
        # length of the highway
        "length": 2500,
        # number of lanes
        "lanes": 1,
        # speed limit for all edges
        "speed_limit": 30,
        # number of edges to divide the highway into
        "num_edges": 2,
        # whether to include a ghost edge of length 500m. This edge is provided
        # a different speed limit.
        "use_ghost_edge": True,
        # speed limit for the ghost edge
        "ghost_speed_limit": END_SPEED,
        # length of the cell imposing a boundary
        "boundary_cell_length": 300,
    })

    vehicles = VehicleParams()
    inflows = InFlows()

    # human vehicles
    vehicles.add(
        "human",
        num_vehicles=0,
        acceleration_controller=(IDMController, {
            "a": 1.3,
            "b": 2.0,
            "noise": 0.3 if INCLUDE_NOISE else 0.0,
            "display_warnings": False,
            "fail_safe": [
                'obey_speed_limit', 'safe_velocity', 'feasible_accel'],
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap=0.5,
            # right of way at intersections + obey limits on deceleration
            speed_mode=12
        ),
    )

    inflows.add(
        veh_type="human",
        edge="highway_0",
        vehs_per_hour=int(TRAFFIC_FLOW * (1 - PENETRATION_RATE)),
        depart_lane="free",
        depart_speed=TRAFFIC_SPEED,
        name="idm_highway_inflow"
    )

    # automated vehicles
    vehicles.add(
        "rl",
        num_vehicles=0,
        acceleration_controller=(RLController, {
            "fail_safe": [
                'obey_speed_limit', 'safe_velocity', 'feasible_accel'],
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap=0.5,
            # right of way at intersections + obey limits on deceleration
            speed_mode=12,
        ),
    )

    inflows.add(
        veh_type="rl",
        edge="highway_0",
        vehs_per_hour=int(TRAFFIC_FLOW * PENETRATION_RATE),
        depart_lane="free",
        depart_speed=TRAFFIC_SPEED,
        name="rl_highway_inflow"
    )

    # SET UP THE FLOW PARAMETERS

    if multiagent:
        if imitation:
            env_name = None  # to be added later
        else:
            env_name = AVOpenMultiAgentEnv
    else:
        if imitation:
            env_name = AVOpenImitationEnv
        else:
            env_name = AVOpenEnv

    return dict(
        # name of the experiment
        exp_tag="highway",

        # name of the flow environment the experiment is running on
        env_name=env_name,

        # name of the network class the experiment is running on
        network=HighwayNetwork,

        # simulator that is used by the experiment
        simulator="traci",

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            evaluate=evaluate,
            horizon=HORIZON,
            warmup_steps=warmup_steps,
            sims_per_step=3,
            done_at_exit=False,
            additional_params={
                "max_accel": 0.5,
                "max_decel": 0.5,
                "target_velocity": 10,
                "stopping_penalty": stopping_penalty,
                "acceleration_penalty": acceleration_penalty,
                "use_follower_stopper": use_follower_stopper,
                "inflows": None if fixed_boundary else INFLOWS,
                "rl_penetration": PENETRATION_RATE,
                "num_rl": float("inf") if multiagent else 10,
                "control_range": [500, 2300],
                "expert_model": (IDMController, {
                    "a": 1.3,
                    "b": 2.0,
                }),
                "warmup_path": WARMUP_PATH,
            }
        ),

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.4,
            render=False,
            restart_instance=True,
            use_ballistic=True,
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
