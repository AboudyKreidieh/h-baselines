import numpy as np
import random
from gym.spaces import Box

from hbaselines.utils.reward_fns import negative_distance
from hbaselines.envs.efficient_hrl.ant_maze_env import AntMazeEnv


class UniversalAntMazeEnv(AntMazeEnv):
    """Universal environment variant of AntMazeEnv.

    TODO: description
    """

    def __init__(self,
                 maze_id,
                 contextual_reward,
                 use_contexts=False,
                 random_contexts=False,
                 context_range=None,
                 horizon=500):
        """Initialize the Universal environment.

        Parameters
        ----------
        maze_id : str
            the type of maze environment. One of "Maze", "Push", or "Fall"
        contextual_reward : function
            a reward function that takes as input (states, goals, next_states)
            and returns a float reward and whether the goal has been achieved
        use_contexts : bool, optional
            specifies whether to add contexts to the observations and add the
            contextual rewards
        random_contexts : bool
            specifies whether the context is a single value, or a random set of
            values between some range
        context_range : list of float or list of (float, float)
            TODO
        horizon : float, optional
            time horizon

        Raises
        ------
        AssertionError
            TODO
        """
        # Initialize the maze variant of the environment.
        super(UniversalAntMazeEnv, self).__init__(
            maze_id=maze_id,
            maze_height=0.5,
            maze_size_scaling=8,
            n_bins=0,
            sensor_range=3.,
            sensor_span=2 * np.pi,
            observe_blocks=False,
            put_spin_near_agent=False,
            top_down_view=False,
            manual_collision=False
        )

        self.horizon = horizon
        self.step_number = 0

        # contextual variables
        self.use_contexts = use_contexts
        self.random_contexts = random_contexts
        self.context_range = context_range or [16, 0]
        self.current_context = None
        self.contextual_reward = contextual_reward

        # a hack to deal with previous observations in the reward
        self.prev_obs = None

        # TODO: add assertions

    @property
    def observation_space(self):
        shape = self._get_obs().shape[0]
        if self.use_contexts:
            shape += len(self.context_range)
        return Box(low=-np.inf * np.ones(shape), high=np.inf * np.ones(shape))

    def step(self, action):
        self.step_number += 1
        obs, rew, done, info = super(UniversalAntMazeEnv, self).step(action)

        if self.use_contexts:
            # Add the contextual reward.
            new_rew, new_done = self.contextual_reward(
                states=self.prev_obs,
                next_states=obs,
                goals=self.current_context,
            )
            rew += new_rew

            # Add the context to the observation.
            obs = np.concatenate((obs, self.current_context), axis=0)

            # Compute the done in terms of the distance to the current context.
            done = done or new_done != 1  # FIXME

        # Check if the time horizon has been met.
        done = done or self.step_number == self.horizon

        # Update the previous observation
        self.prev_obs = np.copy(obs)

        return obs, rew, done, info

    def reset(self):
        self.step_number = 0
        self.prev_obs = super(UniversalAntMazeEnv, self).reset()

        if self.use_contexts:
            if not self.random_contexts:
                # In this case, the context range is just the context.
                self.current_context = self.context_range
            else:
                # TODO: check for if not an option
                # In this case, choose random values between the context range.
                self.current_context = []
                for range_i in self.context_range:
                    minval, maxval = range_i
                    self.current_context.append(random.uniform(minval, maxval))

            # Add the context to the observation.
            self.prev_obs = self.current_context + list(self.prev_obs)

        # Convert to numpy array.
        self.prev_obs = np.asarray(self.prev_obs)
        self.current_context = np.asarray(self.current_context)

        return self.prev_obs


class AntMaze(UniversalAntMazeEnv):
    """Ant Maze Environment.

    TODO: description
    """

    def __init__(self,
                 use_contexts=False,
                 random_contexts=False,
                 context_range=None):
        """Initialize the Ant Maze environment.

        use_contexts : bool, optional
            specifies whether to add contexts to the observations and add the
            contextual rewards
        random_contexts : bool
            specifies whether the context is a single value, or a random set of
            values between some range
        context_range : list of float or list of (float, float)
            TODO

        Raises
        ------
        AssertionError
            TODO
        """
        maze_id = "Maze"

        def contextual_reward(states, goals, next_states):
            return negative_distance(
                states=states,
                goals=goals,
                next_states=next_states,
                state_indices=[0, 1],
                relative_context=False,
                diff=False,
                offset=0.0
            )

        super(UniversalAntMazeEnv, self).__init__(
            maze_id=maze_id,
            contextual_reward=contextual_reward,
            use_contexts=use_contexts,
            random_contexts=random_contexts,
            context_range=context_range
        )


class AntPush(UniversalAntMazeEnv):
    """Ant Push Environment.

    TODO: description
    """

    def __init__(self,
                 use_contexts=False,
                 random_contexts=False,
                 context_range=None):
        """Initialize the Ant Push environment.

        use_contexts : bool, optional
            specifies whether to add contexts to the observations and add the
            contextual rewards
        random_contexts : bool
            specifies whether the context is a single value, or a random set of
            values between some range
        context_range : list of float or list of (float, float)
            TODO

        Raises
        ------
        AssertionError
            TODO
        """
        maze_id = "Push"

        def contextual_reward(states, goals, next_states):
            return negative_distance(
                states=states,
                goals=goals,
                next_states=next_states,
                state_indices=[0, 1],
                relative_context=False,
                diff=False,
                offset=0.0
            )

        super(UniversalAntMazeEnv, self).__init__(
            maze_id=maze_id,
            contextual_reward=contextual_reward,
            use_contexts=use_contexts,
            random_contexts=random_contexts,
            context_range=context_range
        )


class AntFall(UniversalAntMazeEnv):
    """Ant Fall Environment.

    TODO: description
    """

    def __init__(self,
                 use_contexts=False,
                 random_contexts=False,
                 context_range=None):
        """Initialize the Ant Fall environment.

        use_contexts : bool, optional
            specifies whether to add contexts to the observations and add the
            contextual rewards
        random_contexts : bool
            specifies whether the context is a single value, or a random set of
            values between some range
        context_range : list of float or list of (float, float)
            TODO

        Raises
        ------
        AssertionError
            TODO
        """
        maze_id = "Fall"

        def contextual_reward(states, goals, next_states):
            return negative_distance(
                states=states,
                goals=goals,
                next_states=next_states,
                state_indices=[0, 1, 2],
                relative_context=False,
                diff=False,
                offset=0.0
            )

        super(UniversalAntMazeEnv, self).__init__(
            maze_id=maze_id,
            contextual_reward=contextual_reward,
            use_contexts=use_contexts,
            random_contexts=random_contexts,
            context_range=context_range
        )
