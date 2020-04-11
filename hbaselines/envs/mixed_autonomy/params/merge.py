"""Open merge example."""
from flow.envs import MergePOEnv
try:
    from flow.envs.multiagent import MultiMergePOEnv
except (ImportError, ModuleNotFoundError):
    MultiMergePOEnv = object  # TODO: remove once I have an environment
from flow.networks import MergeNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import InFlows, SumoCarFollowingParams
from flow.networks.merge import ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams
from flow.controllers import IDMController, RLController


def get_flow_params(exp_num=1,
                    horizon=6000,
                    simulator="traci",
                    multiagent=False):
    """Return the flow-specific parameters of the merge network.

    This scenario consists of a single-lane highway network with an on-ramp
    used to generate periodic perturbations to sustain congested behavior.

    In order to model the effect of p% CAV penetration on the network, every
    100/pth vehicle is replaced with an automated vehicle whose actions are
    sampled from an RL policy.

    This benchmark is adapted from the following article:

    Kreidieh, Abdul Rahman, Cathy Wu, and Alexandre M. Bayen. "Dissipating
    stop-and-go waves in closed and open networks via deep reinforcement
    learning." 2018 21st International Conference on Intelligent Transportation
    Systems (ITSC). IEEE, 2018.

    Parameters
    ----------
    exp_num : int
        experiment number

        * 0: 10% RL penetration,  5 max controllable vehicles
        * 1: 25% RL penetration, 13 max controllable vehicles
        * 2: 33% RL penetration, 17 max controllable vehicles

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
        if the `exp_num` parameter is a value other than 0, 1, or 2
    """
    assert exp_num in [0, 1, 2], "exp_num must be 0, 1, or 2"

    # inflow rate at the highway
    flow_rate = 2000
    # percent of autonomous vehicles
    rl_penetration = [0.1, 0.25, 0.33][exp_num]
    # num_rl term (see ADDITIONAL_ENV_PARAMs)
    num_rl = [5, 13, 17][exp_num]

    # We consider a highway network with an upstream merging lane producing
    # shockwaves
    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params["merge_lanes"] = 1
    additional_net_params["highway_lanes"] = 1
    additional_net_params["pre_merge_length"] = 500

    # RL vehicles constitute 5% of the total number of vehicles
    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "noise": 0.2
        }),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
        ),
        num_vehicles=5)
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
        ),
        num_vehicles=0)

    # Vehicles are introduced from both sides of merge, with RL vehicles
    # entering from the highway portion as well
    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="inflow_highway",
        vehs_per_hour=(1 - rl_penetration) * flow_rate,
        depart_lane="free",
        depart_speed=10)
    inflow.add(
        veh_type="rl",
        edge="inflow_highway",
        vehs_per_hour=rl_penetration * flow_rate,
        depart_lane="free",
        depart_speed=10)
    inflow.add(
        veh_type="human",
        edge="inflow_merge",
        vehs_per_hour=100,
        depart_lane="free",
        depart_speed=7.5)

    return dict(
        # name of the experiment
        exp_tag="merge",

        # name of the flow environment the experiment is running on
        env_name=MultiMergePOEnv if multiagent else MergePOEnv,

        # name of the network class the experiment is running on
        network=MergeNetwork,

        # simulator that is used by the experiment
        simulator=simulator,

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.2,
            render=False,
            restart_instance=True,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=horizon,
            sims_per_step=5,
            warmup_steps=0,
            additional_params={
                "max_accel": 1.5,
                "max_decel": 1.5,
                "target_velocity": 20,
                "num_rl": num_rl,
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            additional_params=additional_net_params,
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon init/reset
        # (see flow.core.params.InitialConfig)
        initial=InitialConfig(),
    )
