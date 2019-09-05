"""Environment used to train vehicles to improve traffic on a highway."""
import numpy as np
from gym.spaces.box import Box
from flow.core.rewards import desired_velocity
from flow.multiagent_envs.multiagent_env import MultiEnv


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 1,
    # maximum deceleration of autonomous vehicles
    'max_decel': 1,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25,
    # whether to include lane changes to the action space
    "include_lane_changes": False,
}


class MultiAgentHighwayPOEnv(MultiEnv):
    """Partially observable multi-agent environment for an highway with ramps.

    This environment is used to train autonomous vehicles to attenuate the
    formation and propagation of waves in an open highway network. The highway
    can contain an arbitrary number of entrance and exit ramps, and is intended
    to be used with the HighwayRampsScenario scenario.

    This environment supports multi-agent training, where each automated
    vehicle in the network is an agent. The policy is shared among the agents,
    so there can be a non-constant number of RL vehicles throughout the
    simulation.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s
    * include_lane_changes: whether to include lane changes to the action space

    The following states, actions and rewards are considered for one autonomous
    vehicle only, as they will be computed in the same way for each of them.

    States
        The observation consists of the speeds and bumper-to-bumper headways of
        the vehicles immediately preceding and following autonomous vehicle, as
        well as the speed of the autonomous vehicle.

    Actions
        The action consists of an acceleration, bound according to the
        environment parameters. In order to ensure safety, the acceleration is
        bound again by failsafes provided by the simulator at every time step.

    Rewards
        The reward function encourages proximity of the system-level velocity
        to a desired velocity specified in the environment parameters, while
        slightly penalizing small time headways among autonomous vehicles.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.include_lane_changes = env_params.additional_params[
            "include_lane_changes"]

        super().__init__(env_params, sim_params, scenario, simulator)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=0, high=1, shape=(5, ), dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        if self.include_lane_changes:
            low = [-abs(self.env_params.additional_params['max_decel']),
                   -1, -1, -1]
            high = [abs(self.env_params.additional_params['max_accel']),
                    1, 1, 1]
            return Box(low=low, high=high, dtype=np.float32)
        else:
            return Box(
                low=-np.abs(self.env_params.additional_params['max_decel']),
                high=self.env_params.additional_params['max_accel'],
                shape=(1,),
                dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # in the warmup steps, rl_actions is None
        if rl_actions:
            rl_ids = list(rl_actions.keys())
            accel = list(rl_actions.values())
            self.k.vehicle.apply_acceleration(rl_ids, accel)

    def get_state(self):
        """See class definition."""
        obs = {}

        # normalizing constants TODO: probably too large
        max_speed = self.k.scenario.max_speed()
        max_length = self.k.scenario.length()

        for rl_id in self.k.vehicle.get_rl_ids():
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)

            if lead_id in ["", None]:
                # in case leader is not visible
                lead_speed = max_speed
                lead_head = max_length
            else:
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_headway(lead_id)

            if follower in ["", None]:
                # in case follower is not visible
                follow_speed = 0
                follow_head = max_length
            else:
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)

            observation = np.array([
                this_speed / max_speed,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_length,
                (this_speed - follow_speed) / max_speed,
                follow_head / max_length
            ])

            obs.update({rl_id: observation})

        return obs

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rewards = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            if self.env_params.evaluate:
                # reward is speed of vehicle if we are in evaluation mode
                reward = np.mean([self.k.vehicle.get_speed(veh_id)
                                  for veh_id in self.k.vehicle.get_ids()])
            elif kwargs['fail']:
                # reward is 0 if a collision occurred
                reward = 0
            else:
                # reward high system-level velocities
                cost1 = desired_velocity(self, fail=kwargs['fail'])

                # penalize small time headways
                cost2 = 0
                t_min = 1  # smallest acceptable time headway

                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id not in ["", None] \
                        and self.k.vehicle.get_speed(rl_id) > 0:
                    t_headway = max(
                        self.k.vehicle.get_headway(rl_id) /
                        self.k.vehicle.get_speed(rl_id), 0)
                    cost2 += min((t_headway - t_min) / t_min, 0)

                # weights for cost1 and cost2, respectively
                eta1, eta2 = 1.00, 0.10

                reward = max(eta1 * cost1 + eta2 * cost2, 0)

            rewards[rl_id] = reward
        return rewards

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes.
        """
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            # leader
            lead_id = self.k.vehicle.get_leader(rl_id)
            if lead_id:
                self.k.vehicle.set_observed(lead_id)
            # follower
            follow_id = self.k.vehicle.get_follower(rl_id)
            if follow_id:
                self.k.vehicle.set_observed(follow_id)
