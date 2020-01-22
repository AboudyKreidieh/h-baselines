"""Contextual representation of AntMaze, AntPush, and AntFall."""
import numpy as np
import random
from gym.spaces import Box

from hbaselines.utils.reward_fns import negative_distance
from hbaselines.envs.efficient_hrl.ant_maze_env import AntMazeEnv

# scale to the contextual reward. Does not affect the environmental reward.
REWARD_SCALE = 0.1
# threshold after which the agent is considered to have reached its target
DISTANCE_THRESHOLD = 5


class UniversalAntMazeEnv(AntMazeEnv):
    """Universal environment variant of AntMazeEnv.

    FIXME
    This environment extends the generic gym environment by including contexts,
    or goals. The goals are added to the observation, and an additional
    contextual reward is included to the generic rewards. If a certain goal is
    met, the environment registers a "done" flag and the environment is reset.
    """

    def __init__(self,
                 maze_id,
                 contextual_reward,
                 use_contexts=False,
                 random_contexts=False,
                 context_range=None,
                 maze_size_scaling=8,
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
        context_range : [float] or [(float, float)] or [[float]]
            one of the following three:

            1. the desired context / goal
            2. the (lower, upper) bound tuple for each dimension of the goal
            3. a list of desired contexts / goals. Goals are sampled from these
               list of possible goals
        horizon : float, optional
            time horizon

        Raises
        ------
        AssertionError
            If the context_range is not the right form based on whether
            contexts are a single value or random across a range.
        """
        # Initialize the maze variant of the environment.
        super(UniversalAntMazeEnv, self).__init__(
            maze_id=maze_id,
            maze_height=0.5,
            maze_size_scaling=maze_size_scaling,
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
        self.context_range = context_range
        self.contextual_reward = contextual_reward
        self.current_context = None

        # a hack to deal with previous observations in the reward
        self.prev_obs = None

        # Check that context_range is the right form based on whether contexts
        # are a single value or random across a range.
        if self.use_contexts:
            if self.random_contexts:
                assert all(isinstance(i, tuple) for i in self.context_range), \
                    "When using random contexts, every element in " \
                    "context_range, must be a tuple of (min,max) values."
            else:
                assert all(not isinstance(i, tuple) for i in
                           self.context_range), \
                    "When not using random contexts, every element in " \
                    "context_range, must be a single value or a list of " \
                    "values."

    @property
    def context_space(self):
        """Return the shape and bounds of the contextual term."""
        # Check if the environment is using contexts, and if not, return a None
        # value as the context space.
        if self.use_contexts:
            # If the context space is random, use the min and max values of
            # each context to specify the space range. Otherwise, the min and
            # max values are both the deterministic context value.
            if self.random_contexts:
                context_low = []
                context_high = []
                for context_i in self.context_range:
                    low, high = context_i
                    context_low.append(low)
                    context_high.append(high)
                return Box(low=np.asarray(context_low),
                           high=np.asarray(context_high))
            else:
                # If there are a list of possible goals, use the min and max
                # values of each index for the context space.
                if isinstance(self.context_range[0], list):
                    min_val = []
                    max_val = []
                    for i in range(len(self.context_range[0])):
                        min_val.append(min(v[i] for v in self.context_range))
                        max_val.append(max(v[i] for v in self.context_range))

                    return Box(low=np.array(min_val), high=np.array(max_val))
                else:
                    # Use the original context as the context space. It is a
                    # fixed value in this case.
                    return Box(low=np.asarray(self.context_range),
                               high=np.asarray(self.context_range))
        else:
            return None

    def step(self, action):
        """Advance the environment by one simulation step.

        If the environment is using the contextual setting, an "is_success"
        term is added to the info_dict to specify whether the objective has
        been met.

        Parameters
        ----------
        action : array_like
            actions to be performed by the agent

        Returns
        -------
        array_like
            next observation
        float
            environmental reward
        bool
            done mask
        dict
            extra information dictionary
        """
        # Run environment update.
        obs, rew, done, info = super(UniversalAntMazeEnv, self).step(action)

        if self.use_contexts:
            # Add success to the info dict
            dist = self.contextual_reward(
                states=self.prev_obs,
                next_states=obs,
                goals=self.current_context,
            )
            info["is_success"] = abs(dist) < DISTANCE_THRESHOLD * REWARD_SCALE

            # Replace the reward with the contextual reward.
            rew = dist

        # Check if the time horizon has been met.
        self.step_number += 1
        done = done or self.step_number == self.horizon

        return obs, rew, done, info

    def reset(self):
        """Reset the environment.

        If the environment is using the contextual setting, a new context is
        issued.

        Returns
        -------
        array_like
            initial observation
        """
        try:
            self.prev_obs = super(UniversalAntMazeEnv, self).reset()
        except NotImplementedError:
            # for testing purposes
            self.prev_obs = np.empty(1)

        # Reset the step counter.
        self.step_number = 0

        if self.use_contexts:
            if not self.random_contexts:
                if isinstance(self.context_range[0], list):
                    # In this case, sample on of the contexts as the next
                    # environmental context.
                    self.current_context = random.sample(self.context_range, 1)
                    self.current_context = self.current_context[0]
                else:
                    # In this case, the context range is just the context.
                    self.current_context = self.context_range
            else:
                # In this case, choose random values between the context range.
                self.current_context = []
                for range_i in self.context_range:
                    minval, maxval = range_i
                    self.current_context.append(random.uniform(minval, maxval))

            # Convert to numpy array.
            self.current_context = np.asarray(self.current_context)

        return self.prev_obs


class AntMaze(UniversalAntMazeEnv):
    """Ant Maze Environment.

    In this task, immovable blocks are placed to confine the agent to a
    U-shaped corridor. That is, blocks are placed everywhere except at (0,0),
    (8,0), (16,0), (16,8), (16,16), (8,16), and (0,16). The agent is
    initialized at position (0,0) and tasked at reaching a specific target
    position. "Success" in this environment is defined as being within an L2
    distance of 5 from the target.
    """

    def __init__(self,
                 use_contexts=False,
                 random_contexts=False,
                 context_range=None):
        """Initialize the Ant Maze environment.

        Parameters
        ----------
        use_contexts : bool, optional
            specifies whether to add contexts to the observations and add the
            contextual rewards
        random_contexts : bool
            specifies whether the context is a single value, or a random set of
            values between some range
        context_range : list of float or list of (float, float)
            the desired context / goal, or the (lower, upper) bound tuple for
            each dimension of the goal

        Raises
        ------
        AssertionError
            If the context_range is not the right form based on whether
            contexts are a single value or random across a range.
        """
        maze_id = "Maze"

        def contextual_reward(states, goals, next_states):
            return negative_distance(
                states=states,
                goals=goals,
                next_states=next_states,
                state_indices=[0, 1],
                relative_context=False,
                offset=0.0,
                reward_scales=REWARD_SCALE
            )

        super(AntMaze, self).__init__(
            maze_id=maze_id,
            contextual_reward=contextual_reward,
            use_contexts=use_contexts,
            random_contexts=random_contexts,
            context_range=context_range,
            maze_size_scaling=8,
        )


class AntPush(UniversalAntMazeEnv):
    """Ant Push Environment.

    In this task, immovable blocks are placed every where except at (0,0),
    (-8,0), (-8,8), (0,8), (8,8), (16,8), and (0,16), and a movable block is
    placed at (0,8). The agent is initialized at position (0,0), and is tasked
    with the objective of reaching position (0,19). Therefore, the agent must
    first move to the left, push the movable block to the right, and then
    finally navigate to the target. "Success" in this environment is defined as
    being within an L2 distance of 5 from the target.
    """

    def __init__(self,
                 use_contexts=False,
                 random_contexts=False,
                 context_range=None):
        """Initialize the Ant Push environment.

        Parameters
        ----------
        use_contexts : bool, optional
            specifies whether to add contexts to the observations and add the
            contextual rewards
        random_contexts : bool
            specifies whether the context is a single value, or a random set of
            values between some range
        context_range : list of float or list of (float, float)
            the desired context / goal, or the (lower, upper) bound tuple for
            each dimension of the goal

        Raises
        ------
        AssertionError
            If the context_range is not the right form based on whether
            contexts are a single value or random across a range.
        """
        maze_id = "Push"

        def contextual_reward(states, goals, next_states):
            return negative_distance(
                states=states,
                goals=goals,
                next_states=next_states,
                state_indices=[0, 1],
                relative_context=False,
                offset=0.0,
                reward_scales=REWARD_SCALE
            )

        super(AntPush, self).__init__(
            maze_id=maze_id,
            contextual_reward=contextual_reward,
            use_contexts=use_contexts,
            random_contexts=random_contexts,
            context_range=context_range,
            maze_size_scaling=8,
        )


class AntFall(UniversalAntMazeEnv):
    """Ant Fall Environment.

    In this task, the agent is initialized on a platform of height 4. Immovable
    blocks are placed everywhere except at (-8,0), (0,0), (-8,8), (0,8),
    (-8,16), (0,16), (-8,24), and (0,24). The raised platform is absent in the
    region [-4,12]x[12,20], and a movable block is placed at (8,8). The agent
    is initialized at position (0,0,4.5), and is with the objective of reaching
    position (0,27,4.5). Therefore, to achieve this, the agent must first push
    the movable block into the chasm and walk on top of it before navigating to
    the target. "Success" in this environment is defined as being within an L2
    distance of 5 from the target.
    """

    def __init__(self,
                 use_contexts=False,
                 random_contexts=False,
                 context_range=None):
        """Initialize the Ant Fall environment.

        Parameters
        ----------
        use_contexts : bool, optional
            specifies whether to add contexts to the observations and add the
            contextual rewards
        random_contexts : bool
            specifies whether the context is a single value, or a random set of
            values between some range
        context_range : list of float or list of (float, float)
            the desired context / goal, or the (lower, upper) bound tuple for
            each dimension of the goal

        Raises
        ------
        AssertionError
            If the context_range is not the right form based on whether
            contexts are a single value or random across a range.
        """
        maze_id = "Fall"

        def contextual_reward(states, goals, next_states):
            return negative_distance(
                states=states,
                goals=goals,
                next_states=next_states,
                state_indices=[0, 1, 2],
                relative_context=False,
                offset=0.0,
                reward_scales=REWARD_SCALE
            )

        super(AntFall, self).__init__(
            maze_id=maze_id,
            contextual_reward=contextual_reward,
            use_contexts=use_contexts,
            random_contexts=random_contexts,
            context_range=context_range,
            maze_size_scaling=8,
        )


class AntFourRooms(UniversalAntMazeEnv):
    """Ant Four Rooms Environment.

    Need to add description. TODO
    """

    def __init__(self,
                 use_contexts=False,
                 random_contexts=False,
                 context_range=None):
        """Initialize the Ant Four Rooms environment.

        Parameters
        ----------
        use_contexts : bool, optional
            specifies whether to add contexts to the observations and add the
            contextual rewards
        random_contexts : bool
            specifies whether the context is a single value, or a random set of
            values between some range
        context_range : list of float or list of (float, float)
            the desired context / goal, or the (lower, upper) bound tuple for
            each dimension of the goal

        Raises
        ------
        AssertionError
            If the context_range is not the right form based on whether
            contexts are a single value or random across a range.
        """
        maze_id = "FourRooms"

        def contextual_reward(states, goals, next_states):
            return negative_distance(
                states=states,
                goals=goals,
                next_states=next_states,
                state_indices=[0, 1],
                relative_context=False,
                offset=0.0,
                reward_scales=REWARD_SCALE
            )

        super(AntFourRooms, self).__init__(
            maze_id=maze_id,
            contextual_reward=contextual_reward,
            use_contexts=use_contexts,
            random_contexts=random_contexts,
            context_range=context_range,
            maze_size_scaling=3,
        )
