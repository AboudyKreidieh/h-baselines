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

# Number of vehicles in the network
NUM_VEHICLES = 50
# Length of the ring (in meters)
RING_LENGTH = 1500
# Number of lanes in the ring
NUM_LANES = 1


def get_flow_params(num_automated=5,
                    simulator="traci",
                    evaluate=False,
                    multiagent=False):
    """Return the flow-specific parameters of the ring road network.

    This scenario consists of 50-75 vehicles (50 of which are automated) are
    placed on a sing-lane circular track of length 1500 m. In the absence of
    the automated vehicle, the 22 human-driven vehicles exhibit stop-and-go
    instabilities brought about by the string-unstable characteristic of human
    car-following dynamics.

    Parameters
    ----------
    num_automated : int
        number of automated (RL) vehicles
    simulator : str
        the simulator used, one of {'traci', 'aimsun'}
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
    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "a": 0.3,
            "b": 2.0,
            "noise": 0.5,
        }),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=0.5,
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode="strategic",
        ),
        num_vehicles=NUM_VEHICLES - num_automated)
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=0.5,
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=0,  # no lane changes by automated vehicles
        ),
        num_vehicles=num_automated)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params["length"] = RING_LENGTH
    additional_net_params["lanes"] = NUM_LANES

    return dict(
        # name of the experiment
        exp_tag='multilane-ring',

        # name of the flow environment the experiment is running on
        env_name=AVClosedMultiAgentEnv if multiagent else AVClosedEnv,

        # name of the network class the experiment is running on
        network=RingNetwork,

        # simulator that is used by the experiment
        simulator=simulator,

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            use_ballistic=True,
            render=False,
            sim_step=0.5,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=3600,
            warmup_steps=50,
            sims_per_step=2,
            evaluate=evaluate,
            additional_params={
                "max_accel": 1,
                "max_decel": 1,
                "penalty": 1,
                "num_vehicles": [50, 75],
                "sort_vehicles": True,
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
