"""Flow-specific parameters for the ring scenario."""
import os
import numpy as np

from flow.controllers import IDMController
from flow.controllers import ContinuousRouter
from flow.controllers import RLController
from flow.core.params import SumoParams
from flow.core.params import EnvParams
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.core.params import VehicleParams
from flow.core.params import SumoCarFollowingParams
from flow.networks.ring import RingNetwork
from flow.networks.ring import ADDITIONAL_NET_PARAMS

from hbaselines.envs.mixed_autonomy.envs import AVClosedEnv
from hbaselines.envs.mixed_autonomy.envs import AVClosedMultiAgentEnv
from hbaselines.envs.mixed_autonomy.envs.imitation import AVClosedImitationEnv
from hbaselines.envs.mixed_autonomy.envs.utils import get_relative_obs

# Number of vehicles in the network
RING_LENGTH = [220, 270]
# number of automated (RL) vehicles
NUM_AUTOMATED = 1


def full_observation_fn(env):
    """Compute the full state observation.

    This observation consists of the speeds and bumper-to-bumper headways of
    all automated vehicles in the network.
    """
    # Initialize a set on empty observations
    obs = [0 for _ in range(env.observation_space.shape[0])]

    # Add relative observation of each vehicle.
    for i, v_id in enumerate(env.rl_ids()):
        obs[5 * i: 5 * (i + 1)], leader, follower = get_relative_obs(env, v_id)

    return np.asarray(obs)


def get_flow_params(stopping_penalty,
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

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "a": 1.3,
            "b": 2.0,
            "noise": 0.2
        }),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode=25,
            min_gap=0.5,
        ),
        num_vehicles=21)
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=0.5,
            speed_mode=25,
        ),
        num_vehicles=1)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()

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
        exp_tag='ring',

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
            sim_step=0.2,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=1500,
            warmup_steps=warmup_steps,
            sims_per_step=1,
            evaluate=evaluate,
            additional_params={
                "max_accel": 0.5,
                "stopping_penalty": stopping_penalty,
                "acceleration_penalty": acceleration_penalty,
                "use_follower_stopper": False,
                "ring_length": RING_LENGTH,
                "expert_model": (IDMController, {
                    "a": 1.3,
                    "b": 2.0,
                }),
                "full_observation_fn": full_observation_fn,
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
