"""Environment for training automated vehicles in a mixed-autonomy setting."""
import numpy as np
from gym.spaces import Box
from copy import deepcopy

from hbaselines.envs.mixed_autonomy.envs import AVEnv
from hbaselines.envs.mixed_autonomy.envs.av import CLOSED_ENV_PARAMS
from hbaselines.envs.mixed_autonomy.envs.av import OPEN_ENV_PARAMS
from hbaselines.envs.mixed_autonomy.envs.utils import get_rl_accel
from hbaselines.envs.mixed_autonomy.envs.utils import update_rl_veh


class AVMultiAgentEnv(AVEnv):
    """Multi-agent variants of AVEnv."""

    @property
    def action_space(self):
        """See class definition."""
        if self.env_params.additional_params["use_follower_stopper"]:
            return Box(
                low=0,
                high=15,
                shape=(1,),
                dtype=np.float32)
        else:
            return Box(
                low=-1,
                high=1,
                shape=(1,),
                dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(3 * self._obs_frames,),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        if self.env_params.additional_params["use_follower_stopper"]:
            for veh_id in rl_actions.keys():
                self._av_controller.veh_id = veh_id
                self._av_controller.v_des = rl_actions[veh_id][0]
                acceleration = self._av_controller.get_action(self)

                # Apply the action via the simulator.
                self.k.vehicle.apply_acceleration(veh_id, acceleration)
        else:
            rl_ids = list(rl_actions.keys())

            acceleration = get_rl_accel(
                accel=[deepcopy(rl_actions[veh_id][0]) for veh_id in rl_ids],
                vel=self.k.vehicle.get_speed(rl_ids),
                max_accel=self.env_params.additional_params["max_accel"],
                dt=self.sim_step,
            )

            # Run the action through the controller, to include failsafe
            # actions.
            for i, veh_id in enumerate(rl_ids):
                acceleration[i] = self.k.vehicle.get_acc_controller(
                    veh_id).get_action(self, acceleration=acceleration[i])

            # Apply the action via the simulator.
            self.k.vehicle.apply_acceleration(
                acc=acceleration, veh_ids=list(rl_actions.keys()))

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # In case no vehicles were available in the current step, pass an empty
        # reward dict.
        if rl_actions is None:
            return {}

        rl_ids = list(rl_actions.keys())
        num_vehicles = self.k.vehicle.num_vehicles
        vel = np.array(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))

        if any(vel < -100) or kwargs["fail"] or num_vehicles == 0:
            # Return a reward of 0 case of collisions or an empty network.
            reward = {key: 0 for key in rl_ids}
        else:
            c1 = 0.005  # reward scale for the speeds
            c2 = 0.100  # reward scale for the accelerations

            reward = {
                key: (- c1 * (self.k.vehicle.get_speed(key) - self._v_eq) ** 2
                      - c2 * self.k.vehicle.get_accel(key) ** 2)
                for key in rl_ids
            }

        return reward

    def get_state(self):
        """See class definition."""
        # Update the storage of observations for individual vehicles.
        self._update_obs_history()

        # Initialize a set on empty observations
        obs = {key: None for key in self.rl_ids()}

        for i, veh_id in enumerate(self.rl_ids()):
            # Concatenate the past n samples for a given time delta in the
            # output observations.
            obs_t = np.concatenate(self._obs_history[veh_id][::-self._skip])
            obs_vehicle = np.array([0. for _ in range(3 * self._obs_frames)])
            obs_vehicle[:len(obs_t)] = obs_t

            obs[veh_id] = obs_vehicle

        return obs


class AVClosedMultiAgentEnv(AVMultiAgentEnv):
    """Closed network variant of AVMultiAgentEnv."""

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        for p in CLOSED_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Env parameter "{}" not supplied'.format(p))

        super(AVClosedMultiAgentEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )


class AVOpenMultiAgentEnv(AVMultiAgentEnv):
    """Open network variant of AVMultiAgentEnv."""

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        for p in OPEN_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Env parameter "{}" not supplied'.format(p))

        assert not (env_params.additional_params["warmup_path"] is not None
                    and env_params.additional_params["inflows"] is not None), \
            "Cannot assign a value to both \"warmup_path\" and \"inflows\""

        super(AVOpenMultiAgentEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )

    def rl_ids(self):
        """See parent class."""
        return self.rl_veh

    def additional_command(self):
        """See definition in AVOpenEnv."""
        super(AVOpenMultiAgentEnv, self).additional_command()

        # Update the RL lists.
        self.rl_queue, self.rl_veh, self.removed_veh = update_rl_veh(
            self,
            rl_queue=self.rl_queue,
            rl_veh=self.rl_veh,
            removed_veh=self.removed_veh,
            control_range=self._control_range,
            num_rl=self.num_rl,
            rl_ids=reversed(sorted(
                self.k.vehicle.get_rl_ids(), key=self.k.vehicle.get_x_by_id)),
        )

        # Specify actions for the uncontrolled RL vehicles based on human-
        # driven dynamics.
        for veh_id in list(
                set(self.k.vehicle.get_rl_ids()) - set(self.rl_veh)):
            self._rl_controller.veh_id = veh_id
            acceleration = self._rl_controller.get_action(self)
            self.k.vehicle.apply_acceleration(veh_id, acceleration)
