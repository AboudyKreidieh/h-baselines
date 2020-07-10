"""Flow-specific parameters for the multi-lane ring scenario."""
from flow.controllers import IDMController
from flow.controllers import ContinuousRouter
from flow.controllers import RLController
from flow.core.params import SumoParams
from flow.core.params import EnvParams
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.core.params import VehicleParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import SumoLaneChangeParams
from flow.networks.ring import RingNetwork
from flow.networks.ring import ADDITIONAL_NET_PARAMS

from hbaselines.envs.mixed_autonomy.envs import AVClosedEnv
from hbaselines.envs.mixed_autonomy.envs import AVClosedMultiAgentEnv
from hbaselines.envs.mixed_autonomy.envs.imitation import AVClosedImitationEnv

# Number of vehicles in the network
NUM_VEHICLES = [50, 75]
# number of automated (RL) vehicles
NUM_AUTOMATED = 5
# Length of the ring (in meters)
RING_LENGTH = 1500
# Number of lanes in the ring
NUM_LANES = 1
# whether to include noise in the environment
INCLUDE_NOISE = True


def get_flow_params(fixed_density,
                    stopping_penalty,
                    acceleration_penalty,
                    evaluate=False,
                    multiagent=False,
                    imitation=False):
    """Return the flow-specific parameters of the ring road network.

    This scenario consists of 50 (if density is fixed) or 50-75 vehicles (5 of
    which are automated) are placed on a sing-lane circular track of length
    1500m. In the absence of the automated vehicle, the human-driven vehicles
    exhibit stop-and-go instabilities brought about by the string-unstable
    characteristic of human car-following dynamics. Within this setting, the
    RL vehicles are tasked with dissipating the formation and propagation of
    stop-and-go waves via an objective function that rewards maximizing
    system-level speeds.

    Parameters
    ----------
    fixed_density : bool
        specifies whether the number of human-driven vehicles updates in
        between resets
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
    vehicles = VehicleParams()
    for i in range(NUM_AUTOMATED):
        vehicles.add(
            veh_id="human_{}".format(i),
            acceleration_controller=(IDMController, {
                "a": 1.3,
                "b": 2.0,
                "noise": 0.3 if INCLUDE_NOISE else 0.0
            }),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                min_gap=0.5,
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=1621,
            ),
            num_vehicles=NUM_VEHICLES[0] - NUM_AUTOMATED if i == 0 else 0)
        vehicles.add(
            veh_id="rl_{}".format(i),
            acceleration_controller=(RLController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                min_gap=0.5,
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=0,  # no lane changes by automated vehicles
            ),
            num_vehicles=NUM_AUTOMATED if i == 0 else 0)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params["length"] = RING_LENGTH
    additional_net_params["lanes"] = NUM_LANES

    if multiagent:
        if imitation:
            env_name = None  # to be added later
        else:
            env_name = AVClosedMultiAgentEnv
    else:
        if imitation:
            env_name = AVClosedEnv
        else:
            env_name = AVClosedImitationEnv

    return dict(
        # name of the experiment
        exp_tag='multilane-ring',

        # name of the flow environment the experiment is running on
        env_name=env_name,

        # name of the network class the experiment is running on
        network=RingNetwork,

        # simulator that is used by the experiment
        simulator="traci",

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            use_ballistic=True,
            render=False,
            sim_step=0.5,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=1800,
            warmup_steps=50,
            sims_per_step=2,
            evaluate=evaluate,
            additional_params={
                "max_accel": 1,
                "max_decel": 1,
                "target_velocity": 30,
                "stopping_penalty": stopping_penalty,
                "acceleration_penalty": acceleration_penalty,
                "num_vehicles": None if fixed_density else NUM_VEHICLES,
                "even_distribution": False,
                "sort_vehicles": True,
                "expert_model": (IDMController, {
                    "a": 1.3,
                    "b": 2.0,
                }),
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            additional_params=additional_net_params,
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon init/reset
        # (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            spacing="random",
            min_gap=0.5,
            shuffle=True,
        ),
    )
