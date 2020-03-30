"""Environment for training automated vehicles in a mixed-autonomy setting."""
import collections
import numpy as np
from gym.spaces import Box
from copy import deepcopy
import random

from flow.envs import Env
from flow.core.params import VehicleParams


BASE_ENV_PARAMS = dict(
    # maximum acceleration for autonomous vehicles, in m/s^2
    max_accel=1,
    # maximum deceleration for autonomous vehicles, in m/s^2
    max_decel=1,
    # scaling term for the action penalty by the AVs
    penalty=1,
)

CLOSED_ENV_PARAMS = BASE_ENV_PARAMS.copy()
CLOSED_ENV_PARAMS.update(dict(
    # range for the number of vehicles allowed in the network. If set to None,
    # the number of vehicles are is modified from its initial value.
    num_vehicles=[50, 75],
    # whether to sort RL vehicles by their initial position. Used to account
    # for noise brought about by shuffling.
    sort_vehicles=True,
))

OPEN_ENV_PARAMS = BASE_ENV_PARAMS.copy()
OPEN_ENV_PARAMS.update(dict(
    # range for the inflows allowed in the network. If set to None, the inflows
    # are not modified from their initial value.
    inflows=[1000, 2000],
    # the AV penetration rate, defining the portion of inflow vehicles that
    # will be automated. If "inflows" is set to None, this is irrelevant.
    rl_penetration=0.1,
    # maximum number of controllable vehicles in the network
    num_rl=5,
))


class AVEnv(Env):
    """Environment for training automated vehicles in a mixed-autonomy setting.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * penalty: scaling term for the action penalty by the AVs

    States
        The observation consists of the speeds and bumper-to-bumper headways of
        the vehicles immediately preceding and following autonomous vehicle, as
        well as the ego speed of the autonomous vehicles. In multi-lane
        settings, the observations will also include the speeds and
        bumper-to-bumper headways of preceding and following vehicles across
        all lanes.

    Actions
        The action space consists of a vector of bounded accelerations for each
        autonomous vehicle $i$. In order to ensure safety, these actions are
        bounded by failsafes provided by the simulator at every time step.

    Rewards
        The reward provided by the system is equal to the average speed of all
        vehicles in the network minus a scaled penalty for the sum of squares
        of the accelerations.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.

    Attributes
    ----------
    leader : list of str
        the names of the vehicles leading the RL vehicles at any given step.
        Used for visualization.
    follower : list of str
        the names of the vehicles following the RL vehicles at any given step.
        Used for visualization.
    num_rl : int
        a fixed term to represent the number of RL vehicles in the network. In
        closed networks, this is the original number of RL vehicles. Otherwise,
        this value is passed via env_params.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        for p in BASE_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Env parameter "{}" not supplied'.format(p))

        super(AVEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )

        self.leader = []
        self.follower = []
        self.num_rl = deepcopy(self.initial_vehicles.num_rl_vehicles)

    @property
    def rl_ids(self):
        """Return the IDs of the currently observed and controlled RL vehicles.

        This is static in closed networks and dynamic in open networks.
        """
        return self.k.vehicle.get_rl_ids()

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(self.num_rl, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        # maximum number of lanes in any section
        max_lanes = max(self.k.network.num_lanes(edge)
                        for edge in self.k.network.get_edge_list())

        return Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.num_rl * (1 + 4*max_lanes), ),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        self.k.vehicle.apply_acceleration(self.rl_ids, rl_actions)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate or rl_actions is None:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            # reward high system-level average speeds
            reward = np.mean(
                self.k.vehicle.get_speed(self.k.vehicle.get_ids()))

            # penalize the sum of squares of the accelerations
            penalty_scale = self.env_params.additional_params["penalty"]
            reward -= penalty_scale * sum(np.square(rl_actions[:self.num_rl]))

            return reward

    def get_state(self):
        """See class definition."""
        # maximum number of lanes in any section
        max_lanes = max(self.k.network.num_lanes(edge)
                        for edge in self.k.network.get_edge_list())

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        # Initialize a set on empty observations
        obs = [0 for _ in range(self.observation_space.shape[0])]

        for i, veh_id in enumerate(self.rl_ids):
            # Add the speed of the ego vehicle.
            obs[5 * max_lanes * i] = self.k.vehicle.get_speed(veh_id)

            # Add the speed and bumper-to-bumper headway of leading vehicles.
            if max_lanes == 1:
                lead_id = self.k.vehicle.get_leader(veh_id)
                if lead_id in ["", None]:
                    # in case leader is not visible
                    lead_speed = max_speed
                    lead_head = max_length
                else:
                    lead_speed = self.k.vehicle.get_speed(lead_id)
                    lead_head = self.k.vehicle.get_headway(veh_id)
                    self.leader.append(lead_id)

                obs[5 * i + 1] = lead_speed
                obs[5 * i + 2] = lead_head
            else:
                pass  # TODO

            # Add the speed and bumper-to-bumper headway of following vehicles.
            if max_lanes == 1:
                follow_id = self.k.vehicle.get_follower(veh_id)
                if follow_id in ["", None]:
                    # in case follower is not visible
                    follow_speed = max_speed
                    follow_head = max_length
                else:
                    follow_speed = self.k.vehicle.get_speed(follow_id)
                    follow_head = self.k.vehicle.get_headway(follow_id)
                    self.follower.append(follow_id)

                obs[5 * i + 3] = follow_speed
                obs[5 * i + 4] = follow_head
            else:
                pass  # TODO

        return obs

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes.
        """
        # specify observed vehicles
        for veh_id in self.leader + self.follower:
            self.k.vehicle.set_observed(veh_id)

    def reset(self):
        """See parent class.

        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        self.leader = []
        self.follower = []
        return super().reset()


class AVClosedEnv(AVEnv):
    """Closed network variant of AVEnv.

    This environment is suitable for training policies on a ring road.

    We attempt to train a control policy in this setting that is robust to
    changes in density by altering the number of human-driven vehicles within
    the network. The number of automated vehicles, however, are kept constant
    in order to maintain a fixed state/action space. It it worth noting that
    this leads to varying AV penetration rates across simulations.

    Moreover, we ensure that vehicles in the observation/action are sorted by
    their initial position in the network to account for any noise brought
    about by positioning of vehicles after shuffling.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * penalty: scaling term for the action penalty by the AVs
    * num_vehicles: range for the number of vehicles allowed in the network. If
      set to None, the number of vehicles are is modified from its initial
      value.
    * sort_vehicles: whether to sort RL vehicles by their initial position.
      Used to account for noise brought about by shuffling.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        for p in CLOSED_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Env parameter "{}" not supplied'.format(p))

        # this is stored to be reused during the reset procedure
        self._network_cls = network.__class__
        self._network_name = deepcopy(network.orig_name)
        self._network_net_params = deepcopy(network.net_params)
        self._network_initial_config = deepcopy(network.initial_config)
        self._network_traffic_lights = deepcopy(network.traffic_lights)
        self._network_vehicles = deepcopy(network.vehicles)

        # attributes for sorting RL IDs by their initial position.
        self._sorted_rl_ids = []

        super(AVClosedEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )

    @property
    def rl_ids(self):
        """See parent class."""
        if self.env_params.additional_params["sort_vehicles"]:
            return self._sorted_rl_ids
        else:
            return self.k.vehicle.get_rl_ids()

    def reset(self):
        """See class definition."""
        # Skip if ring length is None.
        if self.env_params.additional_params["num_vehicles"] is None:
            return super(AVClosedEnv, self).reset()

        self.step_counter = 1
        self.time_counter = 1

        # Make sure restart instance is set to True when resetting.
        self.sim_params.restart_instance = True

        # Create a new VehicleParams object with a new number of human-
        # driven vehicles.
        n_vehicles = self.env_params.additional_params["num_vehicles"]
        n_rl = self._network_vehicles.num_rl_vehicles
        n_vehicles_low = n_vehicles[0] - n_rl
        n_vehicles_high = n_vehicles[1] - n_rl
        new_n_vehicles = random.randint(n_vehicles_low, n_vehicles_high)
        params = self._network_vehicles.type_parameters

        print("humans: {}, automated: {}".format(new_n_vehicles, n_rl))

        new_vehicles = VehicleParams()
        new_vehicles.add(
            "human",
            acceleration_controller=params["human"]["acceleration_controller"],
            lane_change_controller=params["human"]["lane_change_controller"],
            routing_controller=params["human"]["routing_controller"],
            initial_speed=params["human"]["initial_speed"],
            car_following_params=params["human"]["car_following_params"],
            lane_change_params=params["human"]["lane_change_params"],
            num_vehicles=new_n_vehicles)
        new_vehicles.add(
            "rl",
            acceleration_controller=params["rl"]["acceleration_controller"],
            lane_change_controller=params["rl"]["lane_change_controller"],
            routing_controller=params["rl"]["routing_controller"],
            initial_speed=params["rl"]["initial_speed"],
            car_following_params=params["rl"]["car_following_params"],
            lane_change_params=params["rl"]["lane_change_params"],
            num_vehicles=n_rl)

        # Update the network.
        self.network = self._network_cls(
            self._network_name,
            net_params=self._network_net_params,
            vehicles=new_vehicles,
            initial_config=self._network_initial_config,
            traffic_lights=self._network_traffic_lights,
        )

        # Perform the reset operation.
        obs = super(AVClosedEnv, self).reset()

        # Get the initial positions of the RL vehicles to allow us to sort the
        # vehicles by this term.
        def init_pos(veh_id):
            return self.k.vehicle.get_x_by_id(veh_id)

        # Create a list of the RL IDs sorted by the above term.
        self._sorted_rl_ids = sorted(self.k.vehicle.get_rl_ids(), key=init_pos)

        return obs


class AVOpenEnv(AVEnv):
    """Open network variant of AVEnv.

    This environment is suitable for training policies on a merge or highway
    network.

    We attempt to train a control policy in this setting that is robust to
    changes in density by altering the inflow rate of vehicles within the
    network. This is made to proportionally increase the inflow rate of both
    the human-driven and automated (or RL) vehicles in the network to maintain
    a fixed RL penetration rate.

    Moreover, in order to account for variability in the number of automated
    vehicles during training, we include a "num_rl" term and perform the
    following operations to the states and actions:

    * States: In order to maintain a fixed observation size in open networks,
      when the number of AVs in the network is less than "num_rl", the extra
      entries are filled in with zeros. Conversely, if the number of autonomous
      vehicles is greater than "num_rl", the observations from the additional
      vehicles are not included in the state space.
    * Actions: In order to account for variability in the number of autonomous
      vehicles in open networks, if n_AV < "num_rl" the additional actions
      provided by the agent are not assigned to any vehicle. Moreover, if
      n_AV > "num_rl", the additional vehicles are not provided with actions
      from the learning agent, and instead act as human-driven vehicles as
      well.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * penalty: scaling term for the action penalty by the AVs
    * inflows: range for the inflows allowed in the network. If set to None,
      the inflows are not modified from their initial value.
    * rl_penetration: the AV penetration rate, defining the portion of inflow
      vehicles that will be automated. If "inflows" is set to None, this is
      irrelevant.
    * num_rl: maximum number of controllable vehicles in the network
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        for p in OPEN_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Env parameter "{}" not supplied'.format(p))

        # this is stored to be reused during the reset procedure
        pass  # TODO

        # queue of rl vehicles waiting to be controlled
        self._rl_queue = collections.deque()

        # additional attributes
        self._current_rl_ids = []

        super(AVOpenEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )

    @property
    def rl_ids(self):
        """See parent class."""
        return self._current_rl_ids

    def reset(self):
        """See class definition."""
        if self.env_params.additional_params["inflows"] is not None:
            pass  # TODO

        return super(AVOpenEnv, self).reset()
