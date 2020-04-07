import logging
import numpy as np
from gym import spaces
from pygame import Color
import matplotlib.pyplot as plt
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from multiworld.envs.pygame.pygame_viewer import PygameViewer


class Point2DEnv(MultitaskEnv, Serializable):
    """
    A little 2D point whose life goal is to reach a target.
    """

    def __init__(
            self,
            render_dt_msec=0,
            action_l2norm_penalty=0,  # disabled for now
            render_onscreen=False,
            render_size=32,
            reward_type="dense",
            action_scale=1.0,
            target_radius=0.60,
            boundary_dist=4,
            ball_radius=0.50,
            walls=None,
            fixed_goal=None,
            randomize_position_on_reset=True,
            images_are_rgb=True,  # else black and white
            show_goal=True,
            images_in_obs=True,
            **kwargs
    ):
        if walls is None:
            walls = []
        if walls is None:
            walls = []
        if fixed_goal is not None:
            fixed_goal = np.array(fixed_goal)
        if len(kwargs) > 0:
            LOGGER = logging.getLogger(__name__)
            LOGGER.log(logging.WARNING, "WARNING, ignoring kwargs:", kwargs)

        self.quick_init(locals())
        self.render_dt_msec = render_dt_msec
        self.action_l2norm_penalty = action_l2norm_penalty
        self.render_onscreen = render_onscreen
        self.render_size = render_size
        self.reward_type = reward_type
        self.action_scale = action_scale
        self.target_radius = target_radius
        self.boundary_dist = boundary_dist
        self.ball_radius = ball_radius
        self.walls = walls
        self.fixed_goal = fixed_goal
        self.randomize_position_on_reset = randomize_position_on_reset
        self.images_are_rgb = images_are_rgb
        self.show_goal = show_goal
        self.images_in_obs = images_in_obs
        self.image_size = 1024 * (3 if self.images_are_rgb else 1)
        if not self.images_in_obs:
            self.image_size = 0

        self.max_target_distance = self.boundary_dist - self.target_radius

        self._target_position = None
        self._position = np.zeros(2)

        u = np.ones(2)
        self.action_space = spaces.Box(-u, u, dtype=np.float32)

        o = self.boundary_dist * np.ones(2)
        self.obs_range = spaces.Box(
            -o, o, dtype='float32')
        self.context_space = spaces.Box(
            -o, o, dtype='float32')
        self.observation_space = spaces.Box(
            np.concatenate([np.zeros([self.image_size]), -o], 0),
            np.concatenate([np.ones([self.image_size]), o], 0),
            dtype='float32')

        self.drawer = None
        self.render_drawer = None
        self.horizon = 200
        self.t = 0

    @property
    def current_context(self):
        return self._target_position

    def step(self, velocities):
        assert self.action_scale <= 1.0
        velocities = np.clip(velocities, a_min=-1, a_max=1) * self.action_scale
        new_position = self._position + velocities
        orig_new_pos = new_position.copy()
        for wall in self.walls:
            new_position = wall.handle_collision(
                self._position, new_position
            )
        if sum(new_position != orig_new_pos) > 1:
            """
            Hack: sometimes you get caught on two walls at a time. If you
            process the input in the other direction, you might only get
            caught on one wall instead.
            """
            new_position = orig_new_pos.copy()
            for wall in self.walls[::-1]:
                new_position = wall.handle_collision(
                    self._position, new_position
                )

        self.t += 1

        self._position = new_position
        self._position = np.clip(
            self._position,
            a_min=-self.boundary_dist,
            a_max=self.boundary_dist,
        )
        distance_to_target = np.linalg.norm(
            self._position - self._target_position
        )
        is_success = distance_to_target < self.target_radius

        ob = self._get_obs()
        reward = self.compute_reward(velocities, {'ob': ob})
        info = {
            'radius': self.target_radius,
            'target_position': self._target_position,
            'distance_to_target': distance_to_target,
            'velocity': velocities,
            'speed': np.linalg.norm(velocities),
            'is_success': is_success,
        }
        done = self.t >= self.horizon
        return ob, reward, done, info

    def reset(self):
        self.t = 0
        self._target_position = self.sample_goal()['goals']
        if self.randomize_position_on_reset:
            self._position = self._sample_position(
                self.obs_range.low,
                self.obs_range.high,
            )

        return self._get_obs()

    def _position_inside_wall(self, pos):
        for wall in self.walls:
            if wall.contains_point(pos):
                return True
        return False

    def _sample_position(self, low, high):
        pos = np.random.uniform(low, high)
        while self._position_inside_wall(pos) is True:
            pos = np.random.uniform(low, high)
        return pos

    def _get_obs(self):
        if self.images_in_obs:
            img = self.get_image(32, 32).reshape([-1]).astype(np.float32) / 255.0
            return np.concatenate([img, self._position.copy()], 0)
        else:
            return self._position.copy()

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['ob'][:, self.image_size:]
        desired_goals = self._target_position[np.newaxis, :]
        d = np.linalg.norm(achieved_goals - desired_goals, axis=-1)
        if self.reward_type == "sparse":
            return -(d > self.target_radius).astype(np.float32)
        elif self.reward_type == "dense":
            return -d
        elif self.reward_type == 'vectorized_dense':
            return -np.abs(achieved_goals - desired_goals)
        else:
            raise NotImplementedError()

    def get_goal(self):
        return self._target_position.copy()

    def sample_goals(self, batch_size):
        if not self.fixed_goal is None:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.zeros((batch_size, self.obs_range.low.size))
            for b in range(batch_size):
                if batch_size > 1:
                    logging.warning("This is very slow!")
                goals[b, :] = self._sample_position(
                    self.obs_range.low,
                    self.obs_range.high,
                )
        return {'goals': goals}

    def set_position(self, pos):
        self._position[0] = pos[0]
        self._position[1] = pos[1]

    """Functions for ImageEnv wrapper"""

    def get_image(self, width=None, height=None):
        """Returns a black and white image"""
        if self.drawer is None:
            if width != height:
                raise NotImplementedError()
            self.drawer = PygameViewer(
                screen_width=width,
                screen_height=height,
                x_bounds=(-self.boundary_dist - self.ball_radius, self.boundary_dist + self.ball_radius),
                y_bounds=(-self.boundary_dist - self.ball_radius, self.boundary_dist + self.ball_radius),
                render_onscreen=self.render_onscreen,
            )
        self.draw(self.drawer)
        img = self.drawer.get_image()
        if self.images_are_rgb:
            return img.transpose((1, 0, 2))
        else:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            img = (-r + b).transpose().flatten()
            return img

    def draw(self, drawer):
        drawer.fill(Color('white'))
        if self.show_goal:
            drawer.draw_solid_circle(
                self._target_position,
                self.target_radius,
                Color('green'),
            )
        drawer.draw_solid_circle(
            self._position,
            self.ball_radius,
            Color('blue'),
        )

        for wall in self.walls:
            drawer.draw_segment(
                wall.endpoint1,
                wall.endpoint2,
                Color('black'),
            )
            drawer.draw_segment(
                wall.endpoint2,
                wall.endpoint3,
                Color('black'),
            )
            drawer.draw_segment(
                wall.endpoint3,
                wall.endpoint4,
                Color('black'),
            )
            drawer.draw_segment(
                wall.endpoint4,
                wall.endpoint1,
                Color('black'),
            )
        drawer.render()

    def render(self, mode='human', close=False):

        if mode == 'rgb_array':
            return self.get_image(self.render_size, self.render_size)

        if close:
            self.render_drawer = None
            return

        if self.render_drawer is None or self.render_drawer.terminated:
            self.render_drawer = PygameViewer(
                self.render_size,
                self.render_size,
                x_bounds=(-self.boundary_dist-self.ball_radius, self.boundary_dist+self.ball_radius),
                y_bounds=(-self.boundary_dist-self.ball_radius, self.boundary_dist+self.ball_radius),
                render_onscreen=True,
            )
        self.draw(self.render_drawer)
        self.render_drawer.tick(self.render_dt_msec)
        if mode != 'interactive':
            self.render_drawer.check_for_exit()

    """Static visualization/utility methods"""

    @staticmethod
    def true_model(state, action):
        velocities = np.clip(action, a_min=-1, a_max=1)
        position = state
        new_position = position + velocities
        return np.clip(
            new_position,
            a_min=-Point2DEnv.boundary_dist,
            a_max=Point2DEnv.boundary_dist,
        )

    @staticmethod
    def true_states(state, actions):
        real_states = [state]
        for action in actions:
            next_state = Point2DEnv.true_model(state, action)
            real_states.append(next_state)
            state = next_state
        return real_states

    def plot_trajectory(self, ax, states, actions, goal=None):
        assert len(states) == len(actions) + 1
        x = states[:, 0]
        y = -states[:, 1]
        num_states = len(states)
        plasma_cm = plt.get_cmap('plasma')
        for i, state in enumerate(states):
            color = plasma_cm(float(i) / num_states)
            ax.plot(state[0], -state[1],
                    marker='o', color=color, markersize=10,
                    )

        actions_x = actions[:, 0]
        actions_y = -actions[:, 1]

        ax.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1],
                  scale_units='xy', angles='xy', scale=1, width=0.005)
        ax.quiver(x[:-1], y[:-1], actions_x, actions_y, scale_units='xy',
                  angles='xy', scale=1, color='r',
                  width=0.0035, )
        ax.plot(
            [
                -self.boundary_dist,
                -self.boundary_dist,
            ],
            [
                self.boundary_dist,
                -self.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                self.boundary_dist,
                -self.boundary_dist,
            ],
            [
                self.boundary_dist,
                self.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                self.boundary_dist,
                self.boundary_dist,
            ],
            [
                self.boundary_dist,
                -self.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                self.boundary_dist,
                -self.boundary_dist,
            ],
            [
                -self.boundary_dist,
                -self.boundary_dist,
            ],
            color='k', linestyle='-',
        )

        if goal is not None:
            ax.plot(goal[0], -goal[1], marker='*', color='g', markersize=15)
        ax.set_ylim(
            -self.boundary_dist - 1,
            self.boundary_dist + 1,
        )
        ax.set_xlim(
            -self.boundary_dist - 1,
            self.boundary_dist + 1,
        )
