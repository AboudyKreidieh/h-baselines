"""Script containing a non-flow variant of the ring road environment."""
import numpy as np
import csv
import time
import random
import json
import gym
from scipy.optimize import fsolve
from collections import defaultdict
from gym.spaces import Box

from hbaselines.envs.mixed_autonomy.envs.utils import get_rl_accel
from hbaselines.envs.mixed_autonomy.envs.utils import v_eq_function

# the length of the individual vehicles
VEHICLE_LENGTH = 5.0
# a normalizing term for the vehicle headways
MAX_HEADWAY = 20.0
# a normalizing term for the vehicle speeds
MAX_SPEED = 1.0


class RingEnv(gym.Env):
    """Non-flow variant of the ring road environment.

    Attributes
    ----------
    initial_state : str or None
        the initial state. Must be one of the following:
        * None: in this case, vehicles are evenly distributed
        * "random": in this case, vehicles are randomly placed with a minimum
          gap between vehicles specified by "min_gap"
        * str: A string that is not "random" is assumed to be a path to a json
          file specifying initial vehicle positions and speeds
    length : float
        the length of the ring at the current time step
    num_vehicles : int
        total number of vehicles in the network
    dt : float
        seconds per simulation step
    horizon : int
        the environment time horizon, in steps
    sims_per_step : int
        the number of simulation steps per environment step
    max_accel : float
        scaling factor for the AV accelerations, in m/s^2
    min_gap : float
        the minimum allowable gap by all vehicles. This is used during the
        failsafe computations.
    gen_emission : bool
        whether to generate the emission file
    rl_ids : array_like
        the indices of vehicles that are treated as automated, or RL, vehicles
    num_rl : int
        the number of automated, or RL, vehicles
    warmup_steps : int
        number of steps performed before the initialization of training during
        a rollout
    maddpg : bool
        whether to use a variant that is compatible with MADDPG
    t : int
        number of simulation steps since the start of the current rollout
    positions : array_like
        positions of all vehicles in the network
    speeds : array_like
        speeds of all vehicles in the network
    headways : array_like
        bumper-to-bumper gaps of all vehicles in the network
    accelerations : array_like
        previous step accelerations by the individual vehicles
    v0 : float
        desirable velocity, in m/s
    T : float
        safe time headway, in s
    a : float
        max acceleration, in m/s2
    b : float
        comfortable deceleration, in m/s2
    delta : float
        acceleration exponent
    s0 : float
        linear jam distance, in m
    noise : float
        std dev of normal perturbation to the acceleration
    decel : float
        maximum desired deceleration
    delay : float
        delay in applying the action, in seconds. This is used by the failsafe
        computation.
    """

    def __init__(self,
                 length,
                 num_vehicles,
                 dt,
                 horizon,
                 sims_per_step,
                 max_accel=0.5,
                 min_gap=1.0,
                 gen_emission=False,
                 rl_ids=None,
                 warmup_steps=0,
                 initial_state=None,
                 maddpg=False,
                 obs_frames=5):
        """Instantiate the environment class.

        Parameters
        ----------
        length : float or [float, float]
            the length of the ring if a float, and a range of [min, max] length
            values that are sampled from during the reset procedure
        num_vehicles : int
            total number of vehicles in the network
        dt : float
            seconds per simulation step
        horizon : int
            the environment time horizon, in steps
        sims_per_step : int
            the number of simulation steps per environment step
        max_accel : float
            scaling factor for the AV accelerations, in m/s^2
        min_gap : float
            the minimum allowable gap by all vehicles. This is used during the
            failsafe computations.
        gen_emission : bool
            whether to generate the emission file
        rl_ids : list of int or None
            the indices of vehicles that are treated as automated, or RL,
            vehicles
        warmup_steps : int
            number of steps performed before the initialization of training
            during a rollout
        initial_state : str or None
            the initial state. Must be one of the following:
            * None: in this case, vehicles are evenly distributed
            * "random": in this case, vehicles are randomly placed with a
              minimum gap between vehicles specified by "min_gap"
            * str: A string that is not "random" is assumed to be a path to a
              json file specifying initial vehicle positions and speeds
        maddpg : bool
            whether to use a variant that is compatible with MADDPG
        obs_frames : int
            number of observation frames to use. Additional frames are
            provided from previous time steps.
        """
        self._length = length

        # Load the initial state (if needed).
        if isinstance(initial_state, str) and initial_state != "random":
            with open(initial_state, "r") as fp:
                self.initial_state = json.load(fp)
            self._length = list(self.initial_state.keys())
        else:
            self.initial_state = initial_state

        self.length = self._set_length(self._length)
        self.num_vehicles = num_vehicles
        self.dt = dt
        self.horizon = horizon
        self.sims_per_step = sims_per_step
        self.max_accel = max_accel
        self.min_gap = min_gap
        self.gen_emission = gen_emission
        self.num_rl = len(rl_ids) if rl_ids is not None else 0
        self.rl_ids = np.asarray(rl_ids)
        self.warmup_steps = warmup_steps
        self.maddpg = maddpg
        self.obs_frames = obs_frames
        self._time_log = None
        self._v_eq = None
        self._mean_speeds = None
        self._mean_accels = None

        # observations from previous time steps
        self._obs_history = defaultdict(list)

        # simulation parameters
        self.t = 0
        self.positions, self.speeds = self._set_initial_state(
            length=self.length,
            num_vehicles=self.num_vehicles,
            initial_state=self.initial_state,
            min_gap=self.min_gap,
        )
        self.headways = self._compute_headway()
        self.accelerations = None
        self._emission_data = []

        # human-driver model parameters
        self.v0 = 30
        self.T = 1
        self.a = 1.3
        self.b = 2.0
        self.delta = 4
        self.s0 = 2
        self.noise = 0.0

        # failsafe parameters
        self.decel = 4.5
        self.delay = self.dt

    @staticmethod
    def _set_length(length):
        """Update the length of the ring road.

        Parameters
        ----------
        length : float or [float, float]
            the length of the ring if a float, and a range of [min, max] length
            values that are sampled from during the reset procedure

        Returns
        -------
        float
            the updated ring length
        """
        if isinstance(length, list):
            if len(length) == 2:
                # if the range for the length term was defined by the length
                # parameter
                length = random.randint(length[0], length[1])
            else:
                # if the lengths to choose from were defined the initial_states
                # parameter
                length = int(random.choice(length))

        return length

    @staticmethod
    def _set_initial_state(length, num_vehicles, initial_state, min_gap):
        """Choose an initial state for all vehicles in the network.

        Parameters
        ----------
        length : float
            the length of the ring road
        num_vehicles : int
            number of vehicles in the network
        initial_state : str or None or dict
            the initial state. See description in __init__.

        Returns
        -------
        array_like
            initial vehicle positions
        array_like
            initial vehicle speeds
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
            # Choose from the available initial states.
            pos_vel = random.choice(initial_state[str(length)])
            pos = np.array([pv[0] for pv in pos_vel])
            vel = np.array([pv[1] for pv in pos_vel])

        return pos, vel

    def _update_state(self, pos, vel, accel):
        """Update the positions and speeds of all vehicles.

        Parameters
        ----------
        pos : array_like
            positions of all vehicles in the network
        vel : array_like
            speeds of all vehicles in the network
        accel : array_like
            accelerations of all vehicles in the network

        Returns
        -------
        array_like
            the updated vehicle positions
        array_like
            the updated vehicle speeds
        """
        new_vel = vel + accel * self.dt
        new_pos = np.mod(
            pos + vel * self.dt + 0.5 * accel * self.dt ** 2, self.length)

        return new_pos, new_vel

    def _compute_headway(self):
        """Compute the current step headway for all vehicles."""
        # compute the individual headways
        headway = np.append(
            self.positions[1:] - self.positions[:-1] - VEHICLE_LENGTH,
            self.positions[0] - self.positions[-1] - VEHICLE_LENGTH)

        # dealing with wraparound
        headway[np.argmax(self.positions)] += self.length

        return headway

    def _get_accel(self, vel, h):
        """Compute the accelerations of individual vehicles.

        The acceleration values are dictated by the Intelligent Driver Model
        (IDM), which car-following parameters specified in __init__.

        Parameters
        ----------
        vel : array_like
            speeds of all vehicles in the network
        h : array_like
            bumper-to-bumper gaps of all vehicles in the network

        Returns
        -------
        array_like
            vehicle accelerations
        """
        lead_vel = np.append(vel[1:], vel[0])
        s_star = self.s0 + np.clip(
            vel * self.T + np.multiply(vel, vel - lead_vel) /
            (2 * np.sqrt(self.a * self.b)),
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

    def _failsafe(self, veh_ids):
        """Compute the failsafe maximum acceleration.

        Parameters
        ----------
        veh_ids : array_like
            the IDs of vehicles whose failsafe actions should be computed

        Returns
        -------
        array_like
            maximum accelerations
        """
        lead_vel = self.speeds[(veh_ids + 1) % self.num_vehicles]
        h = self.headways[veh_ids]

        # how much we can reduce the speed in each time step
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
        # compute the additional speed that must be used during deceleration to
        # fix the discrepancy between g and h
        r = (brake_distance - h) / (n * s + t)
        x = n * speed_reduction + r
        assert all(x >= 0)

        v_safe[indx_nonzero] = x

        max_accel = (v_safe - self.speeds[veh_ids]) / self.dt

        return max_accel

    def get_state(self):
        """Compute the environment reward.

        This is defined by the child classes.
        """
        return []

    def compute_reward(self, action):
        """Compute the environment reward.

        This is defined by the child classes.
        """
        return 0

    def step(self, action):
        """Advance the simulation by one step."""
        collision = False
        done = False
        for _ in range(self.sims_per_step):
            self.t += 1

            # Compute the accelerations.
            self.accelerations = self._get_accel(self.speeds, self.headways)

            if self.rl_ids is not None and action is not None:
                # Compute the accelerations for RL vehicles.
                self.accelerations[self.rl_ids] = get_rl_accel(
                    accel=action,
                    vel=self.speeds[self.rl_ids],
                    max_accel=self.max_accel,
                    dt=self.dt,
                )

                # Clip by safe, non-negative bounds.
                accel_min = - self.speeds[self.rl_ids] / self.dt
                accel_max = self._failsafe(self.rl_ids)
                self.accelerations[self.rl_ids] = np.clip(
                    self.accelerations[self.rl_ids],
                    a_max=accel_max,
                    a_min=accel_min)

            # Update the speeds, positions, and headways.
            self.positions, self.speeds = self._update_state(
                pos=self.positions,
                vel=self.speeds,
                accel=self.accelerations,
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
            self._mean_accels.append(np.mean(np.abs(self.accelerations)))

            info.update({"v_eq": self._v_eq})
            info.update({"v_eq_frac": np.mean(self._mean_speeds) / self._v_eq})
            info.update({"v_eq_frac_final": speed / self._v_eq})
            info.update({"speed": np.mean(self._mean_speeds)})
            info.update({"abs_accel": np.mean(self._mean_accels)})

        obs = self.get_state()
        if isinstance(obs, dict):
            obs = {key: np.asarray(obs[key]) for key in obs.keys()}
        else:
            obs = np.asarray(self.get_state())

        reward = self.compute_reward(action if action is not None else [0])

        return obs, reward, done, info

    def reset(self):
        """See parent class.

        We update the ring length to match a new value within a given range.
        """
        self.length = self._set_length(self._length)

        # solve for the velocity upper bound of the ring
        v_guess = 4
        self._v_eq = fsolve(v_eq_function, np.array(v_guess),
                            args=(self.num_vehicles, self.length))[0]
        self._mean_speeds = []
        self._mean_accels = []

        print('\n-----------------------')
        print('ring length:', self.length)
        print('v_eq:', self._v_eq)
        print('-----------------------')

        if self._time_log is None:
            self._time_log = time.time()
        else:
            print("Runtime: {}".format(time.time() - self._time_log))
            self._time_log = time.time()

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

        for _ in range(self.warmup_steps):
            self.step(action=None)

        # observations from previous time steps
        self._obs_history = defaultdict(list)

        return self.get_state()

    def render(self, mode='human'):
        """See parent class."""
        pass


class RingSingleAgentEnv(RingEnv):
    """Single agent variant of the ring environment."""

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_rl,),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(25 * self.num_rl,),
            dtype=np.float32)

    def get_state(self):
        """See parent class."""
        # Initialize a set on empty observations.
        obs = np.array([0. for _ in range(5 * self.obs_frames * self.num_rl)])

        for i, veh_id in enumerate(self.rl_ids):
            # Add relative observation of each vehicle.
            obs_vehicle = [
                # ego speed
                self.speeds[veh_id] / MAX_SPEED,
                # lead speed
                self.speeds[(veh_id + 1) % self.num_vehicles] / MAX_SPEED,
                # lead gap
                min(self.headways[veh_id] / MAX_HEADWAY, 5.0),
                # follower speed
                self.speeds[(veh_id - 1) % self.num_vehicles] / MAX_SPEED,
                # lead gap
                min(self.headways[(veh_id - 1) % self.num_vehicles]
                    / MAX_HEADWAY, 5.0),
            ]
            self._obs_history[veh_id].append(obs_vehicle)

            # Maintain queue length.
            if len(self._obs_history[veh_id]) > self.obs_frames:
                self._obs_history[veh_id] = \
                    self._obs_history[veh_id][-self.obs_frames:]

            # Concatenate the past n samples for a given time delta in the
            # output observations.
            obs_t = np.concatenate(self._obs_history[veh_id][::-1])
            obs[5*self.obs_frames*i:5*self.obs_frames*i+len(obs_t)] = obs_t

        return obs

    def compute_reward(self, action):
        """See parent class."""
        reward_scale = 0.1
        reward = reward_scale * np.mean(self.speeds) ** 2

        return reward


class RingMultiAgentEnv(RingEnv):
    """Multi-agent variant of the ring environment."""

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-1.,
            high=1.,
            shape=(1,),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(25,),
            dtype=np.float32)

    @property
    def all_observation_space(self):
        """Return the shape of the full observation space."""
        return Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(25 * self.num_rl,),
            dtype=np.float32,
        )

    def step(self, action):
        """See parent class.

        The done mask is replaced with a dictionary to match other multi-agent
        environments.
        """
        obs, rew, done, info = super(RingMultiAgentEnv, self).step(action)
        done = {"__all__": done}

        if self.maddpg:
            obs = {
                "obs": obs.copy(),
                "all_obs": self._full_obs(obs),
            }

        return obs, rew, done, info

    def get_state(self):
        """See parent class."""
        obs = {}
        for veh_id in self.rl_ids:
            obs_vehicle = [
                # ego speed
                self.speeds[veh_id] / MAX_SPEED,
                # lead speed
                self.speeds[(veh_id + 1) % self.num_vehicles] / MAX_SPEED,
                # lead gap
                min(self.headways[veh_id] / MAX_HEADWAY, 5.0),
                # follower speed
                self.speeds[(veh_id - 1) % self.num_vehicles] / MAX_SPEED,
                # lead gap
                min(self.headways[(veh_id - 1) % self.num_vehicles]
                    / MAX_HEADWAY, 5.0),
            ]
            self._obs_history[veh_id].append(obs_vehicle)

            # Maintain queue length.
            if len(self._obs_history[veh_id]) > self.obs_frames:
                self._obs_history[veh_id] = \
                    self._obs_history[veh_id][-self.obs_frames:]

            # Concatenate the past n samples for a given time delta and return
            # as the final observation.
            obs_t = np.concatenate(self._obs_history[veh_id][::-1])
            obs_vehicle = np.array([0. for _ in range(5 * self.obs_frames)])
            obs_vehicle[:len(obs_t)] = obs_t

            obs[veh_id] = obs_vehicle

        return obs

    def _full_obs(self, obs):
        """Return the full state observation."""
        return np.concatenate([obs[key] for key in self.rl_ids], axis=0)

    def compute_reward(self, action):
        """See parent class."""
        reward_scale = 0.1
        reward = {
            key: reward_scale * np.mean(self.speeds) ** 2
            for key in self.rl_ids
        }

        return reward

    def reset(self):
        """See parent class."""
        obs = super(RingMultiAgentEnv, self).reset()

        if self.maddpg:
            obs = {
                "obs": obs.copy(),
                "all_obs": self._full_obs(obs),
            }

        return obs


if __name__ == "__main__":
    for scale in range(1, 6):
        res = defaultdict(list)
        for ring_length in range(scale * 220, scale * 271, scale * 1):
            print(ring_length)
            for ix in range(10):
                print(ix)
                env = RingEnv(
                    length=ring_length,
                    num_vehicles=scale * 22,
                    dt=0.4,
                    horizon=1500,
                    gen_emission=False,
                    rl_ids=None,
                    warmup_steps=500,
                    initial_state="random",
                    sims_per_step=1,
                )

                _ = env.reset()
                xy = zip(env.positions, env.speeds)
                res[ring_length].append(sorted(xy))
            with open("ring-v{}.json".format(scale - 1), "w") as out_fp:
                json.dump(res, out_fp)
