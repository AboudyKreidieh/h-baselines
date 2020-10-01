import numpy as np
import gym
import csv
import time
import random
from collections import defaultdict
from scipy.optimize import fsolve

from hbaselines.envs.mixed_autonomy.envs.utils import v_eq_function

VEHICLE_LENGTH = 5
MAX_HEADWAY = 20


class RingEnv(gym.Env):

    def __init__(self,
                 length,
                 num_vehicles,
                 dt,
                 horizon,
                 sims_per_step,
                 min_gap=1.0,
                 gen_emission=False,
                 rl_ids=None,
                 warmup_steps=0,
                 initial_state=None):
        """

        :param length:
        :param num_vehicles:
        :param dt:
        :param rl_ids:
        :param warmup_steps:
        :param initial_state:
        """
        self.length = length
        self.num_vehicles = num_vehicles
        self.dt = dt
        self.horizon = horizon
        self.sims_per_step = sims_per_step
        self.min_gap = min_gap
        self.gen_emission = gen_emission
        self.rl_ids = np.asarray(rl_ids)
        self.warmup_steps = warmup_steps
        self.initial_state = initial_state
        self.time_log = None
        self._v_eq = None
        self._mean_speeds = None

        # simulation parameters
        self.t = 0
        self.positions, self.speeds = self._set_initial_state(
            length=self.length,
            num_vehicles=self.num_vehicles,
            initial_state=self.initial_state,
            min_gap=self.min_gap,
        )
        self.headways = self._compute_headway()
        self._emission_data = []

        # human-driver model parameters
        self.v0 = 30
        self.T = 1
        self.a = 1.3
        self.b = 2.0
        self.delta = 4
        self.s0 = 2
        self.noise = 0.2

        # failsafe parameters
        self.decel = 4.5
        self.delay = 0

    @staticmethod
    def _set_initial_state(length, num_vehicles, initial_state, min_gap):
        """

        :param length:
        :param num_vehicles:
        :param initial_state:
        :return:
        """
        if initial_state is None:
            # uniformly distributed vehicles
            pos = np.arange(0, length, length / num_vehicles)
            # no initial speed (0 m/s)
            vel = np.array([0. for _ in range(num_vehicles)])
        elif initial_state == "random":
            # Choose random number not including a minimum gap.
            pos = sorted(np.random.uniform(
                low=0,
                high=length - num_vehicles * (VEHICLE_LENGTH + min_gap),
                size=(num_vehicles,)))
            # Append to each position the min_gap value.
            pos += (VEHICLE_LENGTH + min_gap) * np.arange(num_vehicles)
            # no initial speed (0 m/s)
            vel = np.array([0. for _ in range(num_vehicles)])
        else:
            pos = np.array([])
            vel = np.array([])

        return pos, vel

    def _update_state(self, pos, vel, accel):
        """

        :param pos:
        :param vel:
        :param accel:
        :return:
        """
        new_vel = vel + accel * self.dt
        new_pos = np.mod(
            pos + vel * self.dt + 0.5 * accel * self.dt ** 2, self.length)

        return new_pos, new_vel

    def _compute_headway(self):
        """

        :return:
        """
        pos = self.positions
        headway = np.append(
            pos[1:] - pos[:-1] - VEHICLE_LENGTH,
            pos[0] - pos[-1] - VEHICLE_LENGTH)
        headway[np.argmax(pos)] += self.length
        return headway

    def _get_accel(self, pos, vel, h):
        """

        :return:
        """
        lead_vel = np.append(vel[1:], vel[0])
        s_star = self.s0 + np.clip(
            vel * self.T + np.multiply(vel, vel - lead_vel) / (2 * np.sqrt(self.a * self.b)),
            a_min=0,
            a_max=np.inf,
        )

        accel = self.a * (
            1 - np.power(vel/self.v0, self.delta) - np.power(s_star/h, 2))
        noise = np.random.normal(0, self.noise, self.num_vehicles)

        accel_max = self._failsafe(np.arange(self.num_vehicles))
        accel_min = - vel / self.dt

        accel = np.clip(accel + noise, a_max=accel_max, a_min=accel_min)

        return accel

    def _get_rl_accel(self, accel, vel, h):
        """

        :param accel:
        :param vel:
        :param h:
        :return:
        """
        # for multi-agent environments
        if isinstance(accel, dict):
            accel = [accel[key] for key in self.rl_ids]

        # Redefine if below a speed threshold so that all actions result in
        # non-negative desired speeds.
        for i, veh_id in enumerate(self.rl_ids):
            ac_range = self.action_space.high - self.action_space.low
            speed = self.speeds[veh_id]
            if speed < 0.5 * ac_range * self.dt:
                accel[i] += 0.5 * ac_range - speed / self.dt

        accel_min = - vel / self.dt
        accel_max = self._failsafe(self.rl_ids)
        # accel_max = np.clip(
        #     2 * (h - self.min_gap - vel * self.dt) / self.dt ** 2,
        #     a_min=accel_min, a_max=np.inf)

        return np.clip(accel, a_max=accel_max, a_min=accel_min)

    def _failsafe(self, veh_ids):
        lead_vel = self.speeds[(veh_ids + 1) % self.num_vehicles]
        h = self.headways[veh_ids]

        # how much we can reduce the speed in each timestep
        speed_reduction = self.decel * self.dt
        # how many steps to get the speed to zero
        steps_to_zero = np.round(lead_vel / speed_reduction)
        brake_distance = self.dt * (
            np.multiply(steps_to_zero, lead_vel) -
            0.5 * speed_reduction * np.multiply(steps_to_zero, steps_to_zero+1)
        )
        brake_distance = h + brake_distance - self.min_gap

        indx_nonzero = brake_distance > 0
        brake_distance = brake_distance[indx_nonzero]

        v_safe = np.zeros(len(veh_ids))

        s = self.dt
        t = self.delay

        # h = the distance that would be covered if it were possible to
        # stop exactly after gap and decelerate with max_deaccel every
        # simulation step
        sqrt_quantity = np.sqrt(
            ((s * s)
             + (4.0 * ((s * (2.0 * brake_distance / speed_reduction - t))
                       + (t * t))))) * -0.5
        n = np.floor(.5 - ((t + sqrt_quantity) / s))
        h = 0.5 * n * (n-1) * speed_reduction * s + n * speed_reduction * t
        assert all(h <= brake_distance + 1e-6)
        # compute the additional speed that must be used during
        # deceleration to fix the discrepancy between g and h
        r = (brake_distance - h) / (n * s + t)
        x = n * speed_reduction + r
        assert all(x >= 0)

        v_safe[indx_nonzero] = x

        max_accel = (v_safe - self.speeds[veh_ids]) / self.dt

        return max_accel

    def get_state(self):
        """

        :return:
        """
        return []

    def compute_reward(self, action):
        """

        :return:
        """
        return 0

    def step(self, action):
        """

        :param action:
        :return:
        """
        collision = False
        done = False
        for _ in range(self.sims_per_step):
            self.t += 1

            # Compute the accelerations.
            accelerations = self._get_accel(
                pos=self.positions,
                vel=self.speeds,
                h=self.headways,
            )

            # Compute the accelerations for RL vehicles.
            if self.rl_ids is not None and action is not None:
                accelerations[self.rl_ids] = self._get_rl_accel(
                    accel=action,
                    vel=self.speeds[self.rl_ids],
                    h=self.headways[self.rl_ids],
                )

            # Update the speeds, positions, and headways.
            self.positions, self.speeds = self._update_state(
                pos=self.positions,
                vel=self.speeds,
                accel=accelerations,
            )
            self.headways = self._compute_headway()

            if self.gen_emission:
                data = {"t": self.t}
                data.update({
                    "pos_{}".format(i): self.positions[i]
                    for i in range(self.num_vehicles)
                })
                data.update({
                    "vel_{}".format(i): self.speeds[i]
                    for i in range(self.num_vehicles)
                })
                self._emission_data.append(data)

            # Determine whether the rollout is done.
            collision = any(self.headways < 0)
            done = (self.t >= (self.warmup_steps + self.horizon)
                    * self.sims_per_step) or collision

            if done:
                break

        if collision:
            print("Collision")

        info = {}
        if self.t > self.warmup_steps * self.sims_per_step:
            speed = np.mean(self.speeds)
            self._mean_speeds.append(speed)

            info.update({"v_eq": self._v_eq})
            info.update({"v_eq_frac": speed / self._v_eq})
            info.update({"speed": np.mean(self._mean_speeds)})

        return self.get_state(), self.compute_reward(action or [0]), done, info

    def reset(self, length=None):
        """

        :return:
        """
        length = random.randint(220, 270)
        self.length = length or self.length

        # solve for the velocity upper bound of the ring
        v_guess = 4
        self._v_eq = fsolve(v_eq_function, np.array(v_guess),
                            args=(self.num_vehicles, self.length))[0]
        self._mean_speeds = []

        print('\n-----------------------')
        print('ring length:', self.length)
        print('v_eq:', self._v_eq)
        print('-----------------------')

        if self.time_log is None:
            self.time_log = time.time()
        else:
            print("Runtime: {}".format(time.time() - self.time_log))
            self.time_log = time.time()

        if len(self._emission_data) > 0:
            # Save the data to a csv.
            with open('people.csv', 'w', newline='') as output_file:
                fc = csv.DictWriter(
                    output_file, fieldnames=self._emission_data[0].keys())
                fc.writeheader()
                fc.writerows(self._emission_data)

            # Empty the dictionary.
            self._emission_data = []

        self.t = 0
        self.positions, self.speeds = self._set_initial_state(
            length=self.length,
            num_vehicles=self.num_vehicles,
            initial_state=self.initial_state,
            min_gap=self.min_gap,
        )

        if self.gen_emission:
            data = {"t": self.t}
            data.update({
                "pos_{}".format(i): self.positions[i]
                for i in range(self.num_vehicles)
            })
            data.update({
                "vel_{}".format(i): self.speeds[i]
                for i in range(self.num_vehicles)
            })
            self._emission_data.append(data)

        for _ in range(self.warmup_steps):
            self.step(action=None)

        return self.get_state()


class RingSingleAgentEnv(RingEnv):

    def __init__(self,
                 length,
                 num_vehicles,
                 dt,
                 horizon,
                 sims_per_step,
                 min_gap=1.0,
                 gen_emission=False,
                 rl_ids=None,
                 warmup_steps=0,
                 initial_state=None):
        """

        :param length:
        :param num_vehicles:
        :param dt:
        :param horizon:
        :param sims_per_step:
        :param min_gap:
        :param gen_emission:
        :param rl_ids:
        :param warmup_steps:
        :param initial_state:
        """
        super(RingSingleAgentEnv, self).__init__(
            length=length,
            num_vehicles=num_vehicles,
            dt=dt,
            horizon=horizon,
            sims_per_step=sims_per_step,
            min_gap=min_gap,
            gen_emission=gen_emission,
            rl_ids=rl_ids,
            warmup_steps=warmup_steps,
            initial_state=initial_state,
        )

        # observations from previous time steps
        self._obs_history = []

    @property
    def action_space(self):
        """See class definition."""
        return gym.spaces.Box(
            low=-0.5,
            high=0.5,
            shape=(1,),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(25,),
            dtype=np.float32)

    def get_state(self):
        """

        :return:
        """
        self.num_rl = len(self.rl_ids)
        # Initialize a set on empty observations
        obs = [0 for _ in range(5 * self.num_rl)]

        for i, veh_id in enumerate(self.rl_ids):
            # Add relative observation of each vehicle.
            obs[5*i: 5*(i+1)] = [
                self.speeds[veh_id],  # ego speed
                self.speeds[(veh_id + 1) % self.num_vehicles],  # lead speed
                self.headways[veh_id] / MAX_HEADWAY,  # lead gap
                self.speeds[(veh_id - 1) % self.num_vehicles],  # follow speed
                self.headways[(veh_id - 1) % self.num_vehicles] / MAX_HEADWAY,  # follow gap
            ]

        # Add the observation to the observation history to the
        self._obs_history.append(obs)
        if len(self._obs_history) > 25:
            self._obs_history = self._obs_history[-25:]

        # Concatenate the past n samples for a given time delta and return as
        # the final observation.
        obs_t = np.concatenate(self._obs_history[::-5])
        obs = np.array([0. for _ in range(25 * self.num_rl)])
        obs[:len(obs_t)] = obs_t

        return obs

    def compute_reward(self, action):
        reward_scale = 0.1
        reward = reward_scale * np.mean(self.speeds) ** 2

        return reward

    def reset(self, length=None):
        obs = super(RingSingleAgentEnv, self).reset(length)

        # observations from previous time steps
        self._obs_history = []

        return obs


class RingMultiAgentEnv(RingEnv):

    def __init__(self,
                 length,
                 num_vehicles,
                 dt,
                 horizon,
                 sims_per_step,
                 min_gap=1.0,
                 gen_emission=False,
                 rl_ids=None,
                 warmup_steps=0,
                 initial_state=None):
        """

        :param length:
        :param num_vehicles:
        :param dt:
        :param horizon:
        :param sims_per_step:
        :param min_gap:
        :param gen_emission:
        :param rl_ids:
        :param warmup_steps:
        :param initial_state:
        """
        super(RingMultiAgentEnv, self).__init__(
            length=length,
            num_vehicles=num_vehicles,
            dt=dt,
            horizon=horizon,
            sims_per_step=sims_per_step,
            min_gap=min_gap,
            gen_emission=gen_emission,
            rl_ids=rl_ids,
            warmup_steps=warmup_steps,
            initial_state=initial_state,
        )

        # observations from previous time steps
        self._obs_history = defaultdict(list)

    @property
    def action_space(self):
        """See class definition."""
        return gym.spaces.Box(
            low=-0.5,
            high=0.5,
            shape=(1,),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(25,),
            dtype=np.float32)

    def step(self, action):
        """

        :param action:
        :return:
        """
        obs, rew, done, info = super(RingMultiAgentEnv, self).step(action)
        done = {"__all__": done}

        return obs, rew, done, info

    def get_state(self):
        """

        :return:
        """
        obs = {}
        for veh_id in self.rl_ids:
            obs_vehicle = [
                self.speeds[veh_id],  # ego speed
                self.speeds[(veh_id + 1) % self.num_vehicles],  # lead speed
                self.headways[veh_id] / MAX_HEADWAY,  # lead gap
                self.speeds[(veh_id - 1) % self.num_vehicles],  # follow speed
                self.headways[(veh_id - 1) % self.num_vehicles] / MAX_HEADWAY,  # follow gap
            ]

            # Add the observation to the observation history to the
            self._obs_history[veh_id].append(obs_vehicle)
            if len(self._obs_history[veh_id]) > 25:
                self._obs_history[veh_id] = self._obs_history[veh_id][-25:]

            # Concatenate the past n samples for a given time delta and return
            # as the final observation.
            obs_t = np.concatenate(self._obs_history[veh_id][::-5])
            obs_vehicle = np.array([0. for _ in range(25)])
            obs_vehicle[:len(obs_t)] = obs_t

            obs[veh_id] = obs_vehicle

        return obs

    def compute_reward(self):
        reward_scale = 0.1
        reward = {
            key: reward_scale * np.mean(self.speeds) ** 2
            for key in self.rl_ids
        }

        return reward

    def reset(self, length=None):
        obs = super(RingMultiAgentEnv, self).reset(length)

        # observations from previous time steps
        self._obs_history = defaultdict(list)

        return obs


if __name__ == "__main__":
    env = RingEnv(
        length=260,
        num_vehicles=22,
        dt=0.2,
        horizon=1500,
        gen_emission=True,
        rl_ids=None,
        warmup_steps=500,
        initial_state="random",
        sims_per_step=1,
    )

    obs = env.reset()

    done = False
    while not done:
        _, rew, done, info = env.step(None)
        if isinstance(done, dict):
            done = done["__all__"]
    _ = env.reset()
