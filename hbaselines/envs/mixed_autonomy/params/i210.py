"""I-210 subnetwork example."""
import os

from flow.controllers import RLController
from flow.controllers import IDMController
from flow.controllers import SimLaneChangeController
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
from flow.core.params import VehicleParams
from flow.core.params import SumoParams
from flow.core.params import SumoLaneChangeParams
from flow.core.params import SumoCarFollowingParams
from flow.networks.i210_subnetwork import I210SubNetwork, EDGES_DISTRIBUTION
import flow.config as flow_config

from hbaselines.envs.mixed_autonomy.envs import AVOpenEnv
from hbaselines.envs.mixed_autonomy.envs import LaneOpenMultiAgentEnv
from hbaselines.envs.mixed_autonomy.envs.imitation import AVOpenImitationEnv

# the inflow rate of vehicles (in veh/hr)
INFLOW_RATE = 2050
# the speed of inflowing vehicles from the main edge (in m/s)
INFLOW_SPEED = 25.5
# fraction of vehicles that are RL vehicles. 0.10 corresponds to 10%
PENETRATION_RATE = 1/12
# horizon over which to run the env
HORIZON = 1500
# range for the inflows allowed in the network. If set to None, the inflows are
# not modified from their initial value.
INFLOWS = [1000, 2000]


def get_flow_params(fixed_boundary,
                    stopping_penalty,
                    acceleration_penalty,
                    evaluate=False,
                    multiagent=False,
                    imitation=False):
    """Return the flow-specific parameters of the I-210 subnetwork.

    Parameters
    ----------
    fixed_boundary : bool
        specifies whether the boundary conditions update in between resets
    stopping_penalty : bool
        whether to include a stopping penalty
    acceleration_penalty : bool
        whether to include a regularizing penalty for accelerations by the AVs
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
    warmup_steps = 50 if os.environ.get("TEST_FLAG") else 500

    # Create the base vehicle types that will be used for inflows.
    vehicles = VehicleParams()
    vehicles.add(
        "human",
        num_vehicles=0,
        acceleration_controller=(IDMController, {
            'a': 1.3,
            'b': 2.0,
            'noise': 0.3,
            "display_warnings": False,
            "fail_safe": [
                'obey_speed_limit', 'safe_velocity', 'feasible_accel'],
        }),
        lane_change_controller=(SimLaneChangeController, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=0.5,
            # right of way at intersections + obey limits on deceleration
            speed_mode=12
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=1621,
        ),
    )
    vehicles.add(
        "rl",
        num_vehicles=0,
        acceleration_controller=(RLController, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=0.5,
            # right of way at intersections + obey limits on deceleration
            speed_mode=12,
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=0,  # no lane changes
        ),
    )

    # Add the inflows from the main highway.
    inflow = InFlows()
    for lane in [0, 1, 2, 3, 4]:
        inflow.add(
            veh_type="human",
            edge="ghost0",
            vehs_per_hour=int(INFLOW_RATE * (1 - PENETRATION_RATE)),
            depart_lane=lane,
            depart_speed=INFLOW_SPEED
        )
        inflow.add(
            veh_type="rl",
            edge="ghost0",
            vehs_per_hour=int(INFLOW_RATE * PENETRATION_RATE),
            depart_lane=lane,
            depart_speed=INFLOW_SPEED
        )

    # Choose the appropriate environment.
    if multiagent:
        if imitation:
            env_name = None  # to be added later
        else:
            env_name = LaneOpenMultiAgentEnv
    else:
        if imitation:
            env_name = AVOpenImitationEnv
        else:
            env_name = AVOpenEnv

    return dict(
        # name of the experiment
        exp_tag='I-210_subnetwork',

        # name of the flow environment the experiment is running on
        env_name=env_name,

        # name of the network class the experiment is running on
        network=I210SubNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # simulation-related parameters
        sim=SumoParams(
            sim_step=0.4,
            render=False,
            restart_instance=True,
            use_ballistic=True,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            evaluate=evaluate,
            horizon=HORIZON,
            warmup_steps=warmup_steps,
            done_at_exit=False,
            sims_per_step=3,
            additional_params={
                "max_accel": 0.5,
                "max_decel": 0.5,
                "target_velocity": 10,
                "stopping_penalty": stopping_penalty,
                "acceleration_penalty": acceleration_penalty,
                "inflows": None if fixed_boundary else INFLOWS,
                "rl_penetration": PENETRATION_RATE,
                "num_rl": 10 if multiagent else 50,
                "control_range": [500, 2300],
                "expert_model": (IDMController, {
                    "a": 1.3,
                    "b": 2.0,
                }),
            }
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            template=os.path.join(
                flow_config.PROJECT_PATH,
                "examples/exp_configs/templates/sumo/i210_with_ghost_cell_"
                "with_downstream.xml"
            ),
            additional_params={
                "on_ramp": False,
                "ghost_edge": True,
            }
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon init / reset
        # (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            edges_distribution=EDGES_DISTRIBUTION.copy(),
        ),
    )
