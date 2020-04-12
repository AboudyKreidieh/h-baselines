"""Script containing the Point2DEnv object."""
import logging
import numpy as np
from gym import spaces
from pygame import Color
import matplotlib.pyplot as plt
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from multiworld.envs.pygame.pygame_viewer import PygameViewer


class Point2DEnv(MultitaskEnv, Serializable):
    """A little 2D point whose life goal is to reach a target.

    States
        TODO

    TODO: everything...

    Attributes
    ----------
    render_dt_msec : float
        seconds before the next frame in the image is rendered
    action_l2norm_penalty : float
        penalty scale for the actions by the agent
    render_onscreen : bool
        whether to include the rendering visually (instead of simply using the
        image for the observation)
    render_size : int
        width/length number of pixels in the rendered image
    reward_type : str
        the reward type. Must be one of: "sparse", "dense", or
        "vectorized_dense"
    action_scale : float
        the multiple from action to velocity
    target_radius : float
        the radius of the targeted position when being rendered
    boundary_dist : float
        the distance from the center to the boundary
    ball_radius : float
        the radius of the agent when being rendered
    walls : TODO
        TODO
    fixed_goal : [float, float] or None
        the goal to use. If set to None, it is picked randomly.
    randomize_position_on_reset : bool
        whether to initialize the position of the agent randomly
    images_are_rgb : bool
        specifies whether the image is RGB. Otherwise, it's black and white
    show_goal : bool
        whether to render the goal(s)
    images_in_obs : bool
        whether to use the image in the observation
    image_size : int
        number of elements in the image. Set to 0 if images are not being used.
    max_target_distance : float
        TODO
    action_space : gym.spaces.*
        the action space of the environment
    obs_range : gym.spaces.*
        the range of the initial position of the agent
    context_space  : gym.spaces.*
        the context space of the environment
    observation_space : gym.spaces.*
        the observation space of the environment
    drawer : multiworld.envs.pygame.pygame_viewer.PygameViewer or None
        The drawer for the images of the environment. Set to None if images are
        not being used.
    render_drawer : multiworld.envs.pygame.pygame_viewer.PygameViewer or None
        The drawer for the images of the environment if the environment is
        being rendered. Set to None if images are not being used.
    horizon : int
        environment time horizon
    t : int
        number of steps since the start of the most recent episode
    """

    def __init__(self,
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
                 images_are_rgb=False,  # else black and white
                 show_goal=True,
                 images_in_obs=True,
                 **kwargs):
        """Instantiate the environment.

        Parameters
        ----------
        render_dt_msec : float
            seconds before the next frame in the image is rendered
        action_l2norm_penalty : float
            penalty scale for the actions by the agent
        render_onscreen : bool
            whether to include the rendering visually (instead of simply using
            the image for the observation)
        render_size : int
            width/length number of pixels in the rendered image
        reward_type : str
            the reward type. Must be one of: "sparse", "dense", or
            "vectorized_dense"
        action_scale : float
            the multiple from action to velocity
        target_radius : float
            the radius of the targeted position when being rendered
        boundary_dist : float
            the distance from the center to the boundary
        ball_radius : float
            the radius of the agent when being rendered
        walls : TODO
            TODO
        fixed_goal : [float, float] or None
            the goal to use. If set to None, it is picked randomly.
        randomize_position_on_reset : bool
            whether to initialize the position of the agent randomly
        images_are_rgb : bool
            specifies whether the image is RGB. Otherwise, it's black and white
        show_goal : bool
            whether to render the goal(s)
        images_in_obs : bool
            whether to use the image in the obsevation
        kwargs : dict
            additional arguments. Unused here.
        """
        if walls is None:
            walls = []
        if fixed_goal is not None:
            fixed_goal = np.array(fixed_goal)
        if len(kwargs) > 0:
            logger = logging.getLogger(__name__)
            logger.log(logging.WARNING, "WARNING, ignoring kwargs:", kwargs)

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
        """Return the current goal by the environment."""
        return self._target_position

    def step(self, velocities):
        """Advance the simulation by one step.

        Parameters
        ----------
        velocities : array_like
            the action by the agent, defined as its velocities in the x and y
            directions

        Returns
        -------
        array_like
            agent's observation of the current environment
        float
            amount of reward associated with the previous state/action pair
        bool
            indicates whether the episode has ended
        dict
            contains other diagnostic information from the previous action
        """
        assert self.action_scale <= 1.0
        velocities = np.clip(
            velocities, a_min=-1, a_max=1) * self.action_scale
        new_position = self._position + velocities
        orig_new_pos = new_position.copy()
        for wall in self.walls:
            new_position = wall.handle_collision(
                self._position, new_position
            )
        if sum(new_position != orig_new_pos) > 1:
            # Hack: sometimes you get caught on two walls at a time. If you
            # process the input in the other direction, you might only get
            # caught on one wall instead.
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
        """Reset the environment."""
        self.t = 0
        self._target_position = self.sample_goal()['goals']
        if self.randomize_position_on_reset:
            self._position = self._sample_position(
                self.obs_range.low,
                self.obs_range.high,
            )

        return self._get_obs()

    def _position_inside_wall(self, pos):
        """Return True if the agent is in a wall."""
        for wall in self.walls:
            if wall.contains_point(pos):
                return True
        return False

    def _sample_position(self, low, high):
        """Sample a starting position for the agent."""
        pos = np.random.uniform(low, high)
        while self._position_inside_wall(pos) is True:
            pos = np.random.uniform(low, high)
        return pos

    def _get_obs(self):
        """Return the observation of the agent.

        See States in the description of the environment for more.
        """
        if self.images_in_obs:
            img = self.get_image(
                32, 32).reshape([-1]).astype(np.float32) / 255.0
            return np.concatenate([img, self._position.copy()], 0)
        else:
            return self._position.copy()

    def compute_rewards(self, actions, obs):
        """See parent class.

        The rewards are described in the Rewards section of the description of
        the environment.
        """
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
        """See parent class."""
        return self._target_position.copy()

    def sample_goals(self, batch_size):
        """See parent class.

        The goal is the desired x,y coordinates.
        """
        if self.fixed_goal is not None:
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

    # ======================================================================= #
    #                     Functions for ImageEnv wrapper                      #
    # ======================================================================= #

    def get_image(self, width=None, height=None):
        """Return a black and white image."""
        if self.drawer is None:
            if width != height:
                raise NotImplementedError()
            self.drawer = PygameViewer(
                screen_width=width,
                screen_height=height,
                x_bounds=(-self.boundary_dist - self.ball_radius,
                          self.boundary_dist + self.ball_radius),
                y_bounds=(-self.boundary_dist - self.ball_radius,
                          self.boundary_dist + self.ball_radius),
                render_onscreen=self.render_onscreen,
            )
        self.draw(self.drawer)
        img = self.drawer.get_image()
        if self.images_are_rgb:
            return img.transpose((1, 0, 2))
        else:
            r, b = img[:, :, 0], img[:, :, 2]
            img = (-r + b).transpose().flatten()
            return img

    def draw(self, drawer):
        """Create the image corresponding to the current state."""
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
        """Render the environment state."""
        if mode == 'rgb_array':
            return self.get_image(self.render_size, self.render_size)

        if close:
            self.render_drawer = None
            return

        if self.render_drawer is None or self.render_drawer.terminated:
            self.render_drawer = PygameViewer(
                self.render_size,
                self.render_size,
                x_bounds=(-self.boundary_dist-self.ball_radius,
                          self.boundary_dist+self.ball_radius),
                y_bounds=(-self.boundary_dist-self.ball_radius,
                          self.boundary_dist+self.ball_radius),
                render_onscreen=True,
            )
        self.draw(self.render_drawer)
        self.render_drawer.tick(self.render_dt_msec)
        if mode != 'interactive':
            self.render_drawer.check_for_exit()

    # ======================================================================= #
    #                  Static visualization/utility methods                   #
    # ======================================================================= #

    @staticmethod
    def true_model(state, action):
        """Return the next position by the agent.

        Parameters
        ----------
        state : array_like
            the state by the agent
        action : array_like
            the action by the agent

        Returns
        -------
        array_like
            the next position.
        """
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
        """Return the next states given a set of states and actions.

        Parameters
        ----------
        state : array_like
            the states by the agent
        actions : array_like
            the actions by the agent

        Returns
        -------
        list of array_like
            the next states
        """
        real_states = [state]
        for action in actions:
            next_state = Point2DEnv.true_model(state, action)
            real_states.append(next_state)
            state = next_state
        return real_states

    def plot_trajectory(self, ax, states, actions, goal=None):
        """Plot the trajectory of an agent.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            the axis object to plot the figure on
        states : array_like
            the states by the agent
        actions : array_like
            the actions by the agent
        goal : [int, int]
            the x,y coordinates of the goal
        """
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
