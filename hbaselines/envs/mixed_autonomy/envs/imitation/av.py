"""Imitation variant of the environments in mixed_autonomy/envs/av.py"""
import random

from flow.core.params import VehicleParams

from hbaselines.envs.mixed_autonomy.envs.av import AVEnv
from hbaselines.envs.mixed_autonomy.envs.av import AVClosedEnv
from hbaselines.envs.mixed_autonomy.envs.av import AVOpenEnv


class AVImitationEnv(AVEnv):
    """Imitation variant of AVEnv.

    States
        See parent class.

    Actions
        No actions are needed. They are automatically provided by the expert
        policy.

    Rewards
        The reward is set to 0. It is not needed here.

    Termination
        See parent class.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        super(AVImitationEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator
        )

        # Update this term to contain the number of vehicles starting with "rl"
        # in their name.
        self.num_rl = len(
            [veh_id for veh_id in self.initial_vehicles.get_ids()
             if veh_id.startswith("rl")]
        )

    def rl_ids(self):
        """Return the IDs of the currently observed and controlled RL vehicles.

        This is static in closed networks and dynamic in open networks.
        """
        return [veh_id for veh_id in self.k.vehicle.get_ids()
                if veh_id.startswith("rl")]

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        pass

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        return 0


class AVClosedImitationEnv(AVClosedEnv):
    """Imitation variant of AVClosedEnv.

    States
        See parent class.

    Actions
        No actions are needed. They are automatically provided by the expert
        policy.

    Rewards
        The reward is set to 0. It is not needed here.

    Termination
        See parent class.

    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        super(AVClosedImitationEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator
        )

        # Update this term to contain the number of vehicles starting with "rl"
        # in their name.
        self.num_rl = len(
            [veh_id for veh_id in self.initial_vehicles.get_ids()
             if veh_id.startswith("rl")]
        )

    def rl_ids(self):
        """See parent class."""
        if self.env_params.additional_params["sort_vehicles"]:
            return self._sorted_rl_ids
        else:
            return [veh_id for veh_id in self.k.vehicle.get_ids()
                    if veh_id.startswith("rl")]

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        pass

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        return 0

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
        self._sorted_rl_ids = sorted(
            [
                veh_id for veh_id in self.k.vehicle.get_ids()
                if veh_id.startswith("rl")
            ],
            key=init_pos
        )

        # Perform the reset operation again because the vehicle IDs weren't
        # caught the first time.
        obs = super(AVClosedEnv, self).reset()

        return obs


class AVOpenImitationEnv(AVOpenEnv):
    """Imitation variant of AVOpenEnv.

    States
        See parent class.

    Actions
        No actions are needed. They are automatically provided by the expert
        policy.

    Rewards
        The reward is set to 0. It is not needed here.

    Termination
        See parent class.
    """

    def rl_ids(self):
        """See parent class."""
        return self.rl_veh

    def additional_command(self):
        """See parent class.

        The name of the rl_ids is modified to be the vehicles starting with rl
        in their name.
        """
        rl_ids = [veh_id for veh_id in self.k.vehicle.get_ids()
                  if veh_id.startswith("rl")]

        # add rl vehicles that just entered the network into the rl queue
        for veh_id in rl_ids:
            if veh_id not in list(self.rl_queue) + self.rl_veh:
                self.rl_queue.append(veh_id)

        # remove rl vehicles that exited the network
        for veh_id in list(self.rl_queue):
            if veh_id not in rl_ids:
                self.rl_queue.remove(veh_id)
        for veh_id in self.rl_veh:
            if veh_id not in rl_ids:
                self.rl_veh.remove(veh_id)

        # fill up rl_veh until they are enough controlled vehicles
        while len(self.rl_queue) > 0 and len(self.rl_veh) < self.num_rl:
            # ignore vehicles that are in the ghost edges
            if self.k.vehicle.get_x_by_id(self.rl_queue[0]) < \
                    self.env_params.additional_params["ghost_length"]:
                break

            rl_id = self.rl_queue.popleft()
            self.rl_veh.append(rl_id)

        # specify observed vehicles
        for veh_id in self.leader + self.follower:
            self.k.vehicle.set_observed(veh_id)
