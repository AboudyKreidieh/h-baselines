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
    # desired velocity for all vehicles in the network, in m/s
    target_velocity=30,
    # the penalty type, one of: {"acceleration", "time_headway", "both"}
    penalty_type="acceleration",
    # scaling term for the action penalty by the AVs
    penalty=1,
)

CLOSED_ENV_PARAMS = BASE_ENV_PARAMS.copy()
CLOSED_ENV_PARAMS.update(dict(
    # range for the number of vehicles allowed in the network. If set to None,
    # the number of vehicles are is modified from its initial value.
    num_vehicles=[50, 75],
    # whether to distribute the automated vehicles evenly among the human
    # driven vehicles. Otherwise, they are randomly distributed.
    even_distribution=False,
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
    # the interval (in meters) in which automated vehicles are controlled
    control_range=[500, 2500],
))

# Scale the normalized speeds and headways are scaled by. Headways are squashed
# to very small values, since the length is 1500, so it is multiplied by ten.
SPEED_SCALE = 1
HEADWAY_SCALE = 10


class AVEnv(Env):
    """Environment for training automated vehicles in a mixed-autonomy setting.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * penalty_type: the penalty type, one of: {"acceleration", "time_headway",
      "both"}
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
            shape=(self.num_rl,),
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
            shape=(self.num_rl * (1 + 4 * max_lanes),),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        self.k.vehicle.apply_acceleration(self.rl_ids(), rl_actions)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        return self._compute_reward_util(
            rl_actions,
            self.k.vehicle.get_ids(),
            **kwargs
        )

    def _compute_reward_util(self, rl_actions, veh_ids, **kwargs):
        """Compute the reward over a specific list of vehicles.

        Parameters
        ----------
        rl_actions : array_like
            the actions performed by the automated vehicles
        veh_ids : list of str
            the vehicle IDs to compute the network-level rewards over

        Returns
        -------
        float
            the computed reward
        """
        if self.env_params.evaluate or rl_actions is None:
            return np.mean(self.k.vehicle.get_speed(veh_ids))
        else:
            num_vehicles = len(veh_ids)
            vel = np.array(self.k.vehicle.get_speed(veh_ids))
            if any(vel < -100) or kwargs["fail"] or num_vehicles == 0:
                # in case of collisions or an empty network
                reward = 0
            else:
                reward = 0

                # =========================================================== #
                # Reward high system-level average speeds.                    #
                # =========================================================== #

                reward_scale = 0.1

                # Compute a positive form of the two-norm from a desired target
                # velocity.
                target = self.env_params.additional_params['target_velocity']
                max_cost = np.array([target] * num_vehicles)
                max_cost = np.linalg.norm(max_cost)
                cost = np.linalg.norm(vel - target)
                reward += reward_scale * max(max_cost - cost, 0)

                # =========================================================== #
                # Penalize stopped RL vehicles.                               #
                # =========================================================== #

                for veh_id in self.rl_ids():
                    speed = self.k.vehicle.get_speed(veh_id)
                    reward -= 5 * max(1 - speed, 0) ** 2

                # =========================================================== #
                # Penalize congestion style behaviors.                        #
                # =========================================================== #

                params = self.env_params.additional_params
                penalty_type = params["penalty_type"]
                penalty_scale = params["penalty"]

                # Penalize the sum of squares of the accelerations.
                if penalty_type in ["acceleration", "both"]:
                    reward -= penalty_scale * \
                        sum(np.square(rl_actions[:self.num_rl]))

                # Penalize small time headways.
                elif penalty_type in ["time_headway", "both"]:
                    cost2 = 0
                    t_min = 2  # smallest acceptable time headway
                    for rl_id in self.rl_ids():
                        lead_id = self.k.vehicle.get_leader(rl_id)
                        if lead_id not in ["", None] \
                                and self.k.vehicle.get_speed(rl_id) > 0:
                            t_headway = max(
                                self.k.vehicle.get_headway(rl_id) /
                                self.k.vehicle.get_speed(rl_id), 0)
                            cost2 += min((t_headway - t_min) / t_min, 0)
                    reward += penalty_scale * cost2

            return reward

    def get_state(self):
        """See class definition."""
        self.leader = []
        self.follower = []

        # maximum number of lanes in any section
        max_lanes = max(self.k.network.num_lanes(edge)
                        for edge in self.k.network.get_edge_list())

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        # Initialize a set on empty observations
        obs = [0 for _ in range(self.observation_space.shape[0])]

        for i, veh_id in enumerate(self.rl_ids()):
            # Add the speed of the ego vehicle.
            obs[5 * max_lanes * i] = self.k.vehicle.get_speed(veh_id, error=0)

            # Add the speed and bumper-to-bumper headway of leading vehicles.
            if max_lanes == 1:
                leader = self.k.vehicle.get_leader(veh_id)
                if leader in ["", None]:
                    # in case leader is not visible
                    lead_speed = max_speed
                    lead_head = max_length
                else:
                    lead_speed = self.k.vehicle.get_speed(leader, error=0)
                    lead_head = self.k.vehicle.get_headway(veh_id, error=0)
                    self.leader.append(leader)

                obs[5 * max_lanes * i + 1] = lead_speed
                obs[5 * max_lanes * i + 2] = lead_head
            else:
                pass  # TODO

            # Add the speed and bumper-to-bumper headway of following vehicles.
            if max_lanes == 1:
                follower = self.k.vehicle.get_follower(veh_id)
                if follower in ["", None]:
                    # in case follower is not visible
                    follow_speed = max_speed
                    follow_head = max_length
                else:
                    follow_speed = self.k.vehicle.get_speed(follower, error=0)
                    follow_head = self.k.vehicle.get_headway(follower, error=0)
                    self.follower.append(follower)

                obs[5 * max_lanes * i + 3] = follow_speed
                obs[5 * max_lanes * i + 4] = follow_head
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
    * penalty_type: the penalty type, one of: {"acceleration", "time_headway",
      "both"}
    * penalty: scaling term for the action penalty by the AVs
    * num_vehicles: range for the number of vehicles allowed in the network. If
      set to None, the number of vehicles are is modified from its initial
      value.
    * even_distribution: whether to distribute the automated vehicles evenly
      among the human driven vehicles. Otherwise, they are randomly distributed
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

        if self.env_params.additional_params["even_distribution"]:
            assert not self.initial_config.shuffle, \
                "InitialConfig.shuffle must be set to False when using even " \
                "distributions."

    def rl_ids(self):
        """See parent class."""
        if self.env_params.additional_params["sort_vehicles"]:
            return self._sorted_rl_ids
        else:
            return self.k.vehicle.get_rl_ids()

    def reset(self):
        """See class definition."""
        if self.env_params.additional_params["num_vehicles"] is None:
            # Skip if ring length is None.
            _ = super(AVClosedEnv, self).reset()
        else:
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

            if self.env_params.additional_params["even_distribution"]:
                num_human = new_n_vehicles - n_rl
                humans_remaining = num_human

                new_vehicles = VehicleParams()
                for i in range(n_rl):
                    # Add one automated vehicle.
                    new_vehicles.add(
                        veh_id="rl_{}".format(i),
                        acceleration_controller=params["rl_{}".format(i)][
                            "acceleration_controller"],
                        lane_change_controller=params["rl_{}".format(i)][
                            "lane_change_controller"],
                        routing_controller=params["rl_{}".format(i)][
                            "routing_controller"],
                        initial_speed=params["rl_{}".format(i)][
                            "initial_speed"],
                        car_following_params=params["rl_{}".format(i)][
                            "car_following_params"],
                        lane_change_params=params["rl_{}".format(i)][
                            "lane_change_params"],
                        num_vehicles=1)

                    # Add a fraction of the remaining human vehicles.
                    vehicles_to_add = round(humans_remaining / (n_rl - i))
                    humans_remaining -= vehicles_to_add
                    new_vehicles.add(
                        veh_id="human_{}".format(i),
                        acceleration_controller=params["human_{}".format(i)][
                            "acceleration_controller"],
                        lane_change_controller=params["human_{}".format(i)][
                            "lane_change_controller"],
                        routing_controller=params["human_{}".format(i)][
                            "routing_controller"],
                        initial_speed=params["human_{}".format(i)][
                            "initial_speed"],
                        car_following_params=params["human_{}".format(i)][
                            "car_following_params"],
                        lane_change_params=params["human_{}".format(i)][
                            "lane_change_params"],
                        num_vehicles=vehicles_to_add)
            else:
                new_vehicles = VehicleParams()
                new_vehicles.add(
                    "human_0",
                    acceleration_controller=params["human_0"][
                        "acceleration_controller"],
                    lane_change_controller=params["human_0"][
                        "lane_change_controller"],
                    routing_controller=params["human_0"]["routing_controller"],
                    initial_speed=params["human_0"]["initial_speed"],
                    car_following_params=params["human_0"][
                        "car_following_params"],
                    lane_change_params=params["human_0"]["lane_change_params"],
                    num_vehicles=new_n_vehicles)
                new_vehicles.add(
                    "rl_0",
                    acceleration_controller=params["rl_0"][
                        "acceleration_controller"],
                    lane_change_controller=params["rl_0"][
                        "lane_change_controller"],
                    routing_controller=params["rl_0"]["routing_controller"],
                    initial_speed=params["rl_0"]["initial_speed"],
                    car_following_params=params["rl_0"][
                        "car_following_params"],
                    lane_change_params=params["rl_0"]["lane_change_params"],
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
            _ = super(AVClosedEnv, self).reset()

        # Get the initial positions of the RL vehicles to allow us to sort the
        # vehicles by this term.
        def init_pos(veh_id):
            return self.k.vehicle.get_x_by_id(veh_id)

        # Create a list of the RL IDs sorted by the above term.
        self._sorted_rl_ids = sorted(self.k.vehicle.get_rl_ids(), key=init_pos)

        # Perform the reset operation again because the vehicle IDs weren't
        # caught the first time.
        obs = super(AVClosedEnv, self).reset()

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

    Finally, in order to ignore the effects of the boundaries when performing
    control, autonomous vehicles are only performed and acted on within a
    certain range specified under the "control_range" parameter.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * penalty_type: the penalty type, one of: {"acceleration", "time_headway",
      "both"}
    * penalty: scaling term for the action penalty by the AVs
    * inflows: range for the inflows allowed in the network. If set to None,
      the inflows are not modified from their initial value.
    * rl_penetration: the AV penetration rate, defining the portion of inflow
      vehicles that will be automated. If "inflows" is set to None, this is
      irrelevant.
    * num_rl: maximum number of controllable vehicles in the network
    * control_range: the interval (in meters) in which automated vehicles are
      controlled
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        for p in OPEN_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Env parameter "{}" not supplied'.format(p))

        super(AVOpenEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )

        # maximum number of controlled vehicles
        self.num_rl = env_params.additional_params["num_rl"]

        # queue of rl vehicles waiting to be controlled
        self.rl_queue = collections.deque()

        # names of the rl vehicles controlled at any step
        self.rl_veh = []

        # names of the rl vehicles past the control range
        self.removed_veh = []

        # used for visualization: the vehicles behind and after RL vehicles
        # (ie the observed vehicles) will have a different color
        self.leader = []
        self.follower = []

    def rl_ids(self):
        """See parent class."""
        return self.rl_veh

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # Collect the names of the vehicles within the control range.
        control_range = self.env_params.additional_params["control_range"]
        control_min = control_range[0]
        control_max = control_range[1]
        veh_ids = [
            veh_id for veh_id in self.k.vehicle.get_ids() if
            control_min <= self.k.vehicle.get_x_by_id(veh_id) <= control_max
        ]

        return self._compute_reward_util(rl_actions, veh_ids, **kwargs)

    def additional_command(self):
        """See parent class.

        This method performs to auxiliary tasks:

        * Define which vehicles are observed for visualization purposes.
        * Maintains the "rl_veh" and "rl_queue" variables to ensure the RL
          vehicles that are represented in the state space does not change
          until one of the vehicles in the state space leaves the network.
          Then, the next vehicle in the queue is added to the state space and
          provided with actions from the policy.
        """
        control_range = self.env_params.additional_params["control_range"]

        # add rl vehicles that just entered the network into the rl queue
        for veh_id in self.k.vehicle.get_rl_ids():
            if veh_id not in \
                    list(self.rl_queue) + self.rl_veh + self.removed_veh:
                self.rl_queue.append(veh_id)

        # remove rl vehicles that exited the controllable range of the network
        for veh_id in self.rl_veh:
            if self.k.vehicle.get_x_by_id(veh_id) > control_range[1]:
                self.removed_veh.append(veh_id)
                self.rl_veh.remove(veh_id)

        # fill up rl_veh until they are enough controlled vehicles
        while len(self.rl_queue) > 0 and len(self.rl_veh) < self.num_rl:
            # ignore vehicles that are in the ghost edges
            if self.k.vehicle.get_x_by_id(self.rl_queue[0]) < \
                    control_range[0]:
                break

            rl_id = self.rl_queue.popleft()
            veh_pos = self.k.vehicle.get_x_by_id(rl_id)

            # add the vehicle if it is within the control range
            if veh_pos < control_range[1]:
                self.rl_veh.append(rl_id)

        # specify observed vehicles
        for veh_id in self.leader + self.follower:
            self.k.vehicle.set_observed(veh_id)

    def reset(self):
        """See class definition."""
        if self.env_params.additional_params["inflows"] is not None:
            pass  # TODO

        self.leader = []
        self.follower = []
        self.rl_veh = []
        self.removed_veh = []
        self.rl_queue = collections.deque()
        return super(AVOpenEnv, self).reset()
