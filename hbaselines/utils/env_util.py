"""Utility methods when instantiating environments."""
import numpy as np
from gym.spaces import Box
import gym

from hbaselines.envs.deeploco.envs import BipedalSoccer
from hbaselines.envs.efficient_hrl.envs import AntMaze
from hbaselines.envs.efficient_hrl.envs import AntFall
from hbaselines.envs.efficient_hrl.envs import AntPush
from hbaselines.envs.efficient_hrl.envs import AntFourRooms
from hbaselines.envs.hac.envs import UR5, Pendulum

try:
    from hbaselines.envs.snn4hrl.envs import AntGatherEnv
except (ImportError, ModuleNotFoundError):
    pass

try:
    from flow.utils.registry import make_create_env
    from hbaselines.envs.mixed_autonomy import FlowEnv
    from hbaselines.envs.mixed_autonomy.merge \
        import get_flow_params as merge
    from hbaselines.envs.mixed_autonomy.ring \
        import get_flow_params as ring
    from hbaselines.envs.mixed_autonomy.figure_eight \
        import get_flow_params as figure_eight
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    pass  # pragma: no cover


# This dictionary element contains all relevant information when instantiating
# a single-agent, multi-agent, or hierarchical environment.
#
# The key in this dictionary in the name of the environment. The attributes for
# each element are:
#
# - manager_ac_space: a lambda function that takes an input whether the Manager
#   is assigning relative goals and returns the action space of the Manager
# - state_indices: a list that assigns the indices that correspond to goals in
#   the Worker's state space
# - env: a lambda term that takes an input (evaluate, render, multiagent,
#   shared, maddpg) and return an environment or list of environments
ENV_ATTRIBUTES = {
    # ======================================================================= #
    # Variants of the AntMaze environment.                                    #
    # ======================================================================= #
    "AntMaze": {
        "manager_ac_space": lambda relative_goals: Box(
            low=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3, -0.5,
                          -0.3, -0.5, -0.3, -0.5, -0.3]),
            high=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5,
                           0.3, 0.5, 0.3]),
            dtype=np.float32,
        ),
        "state_indices": [i for i in range(15)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: [
            AntMaze(
                use_contexts=True,
                context_range=[16, 0]
            ),
            AntMaze(
                use_contexts=True,
                context_range=[16, 16]
            ),
            AntMaze(
                use_contexts=True,
                context_range=[0, 16]
            )
        ] if evaluate else AntMaze(
            use_contexts=True,
            random_contexts=True,
            context_range=[(-4, 20), (-4, 20)]
        ),
    },

    "AntPush": {
        "manager_ac_space": lambda relative_goals: Box(
            low=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3, -0.5,
                          -0.3, -0.5, -0.3, -0.5, -0.3]),
            high=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5,
                           0.3, 0.5, 0.3]),
            dtype=np.float32,
        ),
        "state_indices": [i for i in range(15)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: AntPush(
            use_contexts=True,
            context_range=[0, 19]
        ) if evaluate else AntPush(
            use_contexts=True,
            context_range=[0, 19]
            # random_contexts=True,
            # context_range=[(-16, 16), (-4, 20)])
        ),
    },

    "AntFall": {
        "manager_ac_space": lambda relative_goals: Box(
            low=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3, -0.5,
                          -0.3, -0.5, -0.3, -0.5, -0.3]),
            high=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5,
                           0.3, 0.5, 0.3]),
            dtype=np.float32,
        ),
        "state_indices": [i for i in range(15)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: AntFall(
            use_contexts=True,
            context_range=[0, 27, 4.5]
        ) if evaluate else AntFall(
            use_contexts=True,
            context_range=[0, 27, 4.5]
            # random_contexts=True,
            # context_range=[(-4, 12), (-4, 28), (0, 5)])
        ),
    },

    "AntGather": {
        "manager_ac_space": lambda relative_goals: Box(
            low=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3, -0.5,
                          -0.3, -0.5, -0.3, -0.5, -0.3]),
            high=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5,
                           0.3, 0.5, 0.3]),
            dtype=np.float32,
        ),
        "state_indices": [i for i in range(15)],
        "env": lambda evaluate, render, multiagent, shared, maddpg:
        AntGatherEnv(),
    },

    "AntFourRooms": {
        "manager_ac_space": lambda relative_goals: Box(
            low=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3, -0.5,
                          -0.3, -0.5, -0.3, -0.5, -0.3]),
            high=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5,
                           0.3, 0.5, 0.3]),
            dtype=np.float32,
        ),
        "state_indices": [i for i in range(15)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: [
            AntFourRooms(
                use_contexts=True,
                context_range=[30, 0]
            ),
            AntFourRooms(
                use_contexts=True,
                context_range=[0, 30]
            ),
            AntFourRooms(
                use_contexts=True,
                context_range=[30, 30]
            )
        ] if evaluate else AntFourRooms(
            use_contexts=True,
            random_contexts=False,
            context_range=[[30, 0], [0, 30], [30, 30]]
        ),
    },

    # ======================================================================= #
    # UR5 and Pendulum environments.                                          #
    # ======================================================================= #
    "UR5": {
        "manager_ac_space": lambda relative_goals: Box(
            low=np.array([-2 * np.pi, -2 * np.pi, -2 * np.pi, -4, -4, -4]),
            high=np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 4, 4, 4]),
            dtype=np.float32,
        ),
        "state_indices": None,
        "env": lambda evaluate, render, multiagent, shared, maddpg: UR5(
            use_contexts=True,
            random_contexts=True,
            context_range=[(-np.pi, np.pi), (-np.pi / 4, 0),
                           (-np.pi / 4, np.pi / 4)],
            show=render
        ) if evaluate else UR5(
            use_contexts=True,
            random_contexts=True,
            context_range=[(-np.pi, np.pi), (-np.pi / 4, 0),
                           (-np.pi / 4, np.pi / 4)],
            show=render
        ),
    },

    "Pendulum": {
        "manager_ac_space": lambda relative_goals: Box(
            low=np.array([-np.pi, -15]),
            high=np.array([np.pi, 15]),
            dtype=np.float32
        ),
        "state_indices": [0, 2],
        "env": lambda evaluate, render, multiagent, shared, maddpg: Pendulum(
            use_contexts=True,
            context_range=[0, 0],
            show=render
        ) if evaluate else Pendulum(
            use_contexts=True,
            random_contexts=True,
            context_range=[(np.deg2rad(-16), np.deg2rad(16)), (-0.6, 0.6)],
            show=render
        ),
    },

    # ======================================================================= #
    # Mixed autonomy traffic flow environments.                               #
    # ======================================================================= #
    "ring0": {
        "manager_ac_space": lambda relative_goals: Box(
            low=-.5 if relative_goals else 0,
            high=.5 if relative_goals else 1,
            shape=(1,),
            dtype=np.float32
        ),
        "state_indices": [0],
        "env": lambda evaluate, render, multiagent, shared, maddpg: [
            FlowEnv(
                flow_params=ring(
                    ring_length=[230, 230],
                    evaluate=True,
                    multiagent=multiagent,
                ),
                render=render,
                multiagent=multiagent,
                shared=shared,
                maddpg=maddpg,
            ),
            FlowEnv(
                flow_params=ring(
                    ring_length=[260, 260],
                    evaluate=True,
                    multiagent=multiagent,
                ),
                render=render,
                multiagent=multiagent,
                shared=shared,
                maddpg=maddpg,
            ),
            FlowEnv(
                flow_params=ring(
                    ring_length=[290, 290],
                    evaluate=True,
                    multiagent=multiagent,
                ),
                render=render,
                multiagent=multiagent,
                shared=shared,
                maddpg=maddpg,
            )
        ] if evaluate else FlowEnv(
            flow_params=ring(
                evaluate=evaluate,
                multiagent=multiagent,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    "figureeight0": {
        "manager_ac_space": lambda relative_goals: Box(
            low=-.5 if relative_goals else 0,
            high=.5 if relative_goals else 1,
            shape=(1,),
            dtype=np.float32
        ),
        "state_indices": [13],
        "env": lambda evaluate, render, multiagent, shared, maddpg: FlowEnv(
            flow_params=figure_eight(
                num_automated=1,
                horizon=1500,
                simulator="traci",
                multiagent=multiagent,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    "figureeight1": {
        "manager_ac_space": lambda relative_goals: Box(
            low=-.5 if relative_goals else 0,
            high=.5 if relative_goals else 1,
            shape=(7,),
            dtype=np.float32
        ),
        "state_indices": [i for i in range(1, 14, 2)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: FlowEnv(
            flow_params=figure_eight(
                num_automated=7,
                horizon=1500,
                simulator="traci",
                multiagent=multiagent,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    "figureeight2": {
        "manager_ac_space": lambda relative_goals: Box(
            low=-.5 if relative_goals else 0,
            high=.5 if relative_goals else 1,
            shape=(14,),
            dtype=np.float32
        ),
        "state_indices": [i for i in range(14)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: FlowEnv(
            flow_params=figure_eight(
                num_automated=14,
                horizon=1500,
                simulator="traci",
                multiagent=multiagent,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    "merge0": {
        "manager_ac_space": lambda relative_goals: Box(
            low=-.5 if relative_goals else 0,
            high=.5 if relative_goals else 1,
            shape=(5,),
            dtype=np.float32
        ),
        "state_indices": [5 * i for i in range(5)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: FlowEnv(
            flow_params=merge(
                exp_num=0,
                horizon=6000,
                simulator="traci",
                multiagent=multiagent,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    "merge1": {
        "manager_ac_space": lambda relative_goals: Box(
            low=-.5 if relative_goals else 0,
            high=.5 if relative_goals else 1,
            shape=(13,),
            dtype=np.float32
        ),
        "state_indices": [5 * i for i in range(13)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: FlowEnv(
            flow_params=merge(
                exp_num=1,
                horizon=6000,
                simulator="traci",
                multiagent=multiagent,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    "merge2": {
        "manager_ac_space": lambda relative_goals: Box(
            low=-.5 if relative_goals else 0,
            high=.5 if relative_goals else 1,
            shape=(17,),
            dtype=np.float32
        ),
        "state_indices": [5 * i for i in range(17)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: FlowEnv(
            flow_params=merge(
                exp_num=2,
                horizon=6000,
                simulator="traci",
                multiagent=multiagent,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    # ======================================================================= #
    # Bipedal environments.                                                   #
    # ======================================================================= #
    "BipedalSoccer": {
        "manager_ac_space": lambda relative_goals: Box(
            low=np.array([0, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -1,
                          -2]),
            high=np.array([1.5, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2]),
            dtype=np.float32
        ),
        "state_indices": [0, 4, 5, 6, 7, 32, 33, 34, 50, 51, 52, 57, 58, 59],
        "env": lambda evaluate, render, multiagent, shared, maddpg:
        BipedalSoccer(render=render),
    },
}


def get_manager_ac_space(ob_space,
                         relative_goals,
                         env_name,
                         use_fingerprints,
                         fingerprint_dim):
    """Compute the action space for the Manager.

    If the fingerprint terms are being appended onto the observations, this
    should be removed from the action space.

    Parameters
    ----------
    ob_space : gym.spaces.*
        the observation space of the environment
    relative_goals : bool
        specifies whether the goal issued by the Manager is meant to be a
        relative or absolute goal, i.e. specific state or change in state
    env_name : str
        the name of the environment. Used for special cases to assign the
        Manager action space to only ego observations in the observation space.
    use_fingerprints : bool
        specifies whether to add a time-dependent fingerprint to the
        observations
    fingerprint_dim : tuple of int
        the shape of the fingerprint elements, if they are being used

    Returns
    -------
    gym.spaces.Box
        the action space of the Manager policy
    """
    if env_name in ENV_ATTRIBUTES.keys():
        manager_ac_space = ENV_ATTRIBUTES[env_name]["manager_ac_space"](
            relative_goals)
    else:
        if use_fingerprints:
            low = np.array(ob_space.low)[:-fingerprint_dim[0]]
            high = ob_space.high[:-fingerprint_dim[0]]
            manager_ac_space = Box(low=low, high=high, dtype=np.float32)
        else:
            manager_ac_space = ob_space

    return manager_ac_space


def get_state_indices(ob_space,
                      env_name,
                      use_fingerprints,
                      fingerprint_dim):
    """Return the state indices for the worker rewards.

    This assigns the indices of the state that are assigned goals, and
    subsequently rewarded for performing those goals.

    Parameters
    ----------
    ob_space : gym.spaces.*
        the observation space of the environment
    env_name : str
        the name of the environment. Used for special cases to assign the
        Manager action space to only ego observations in the observation space.
    use_fingerprints : bool
        specifies whether to add a time-dependent fingerprint to the
        observations
    fingerprint_dim : tuple of int
        the shape of the fingerprint elements, if they are being used

    Returns
    -------
    list of int
        the state indices that are assigned goals
    """
    if env_name in ENV_ATTRIBUTES.keys():
        state_indices = ENV_ATTRIBUTES[env_name]["state_indices"]
    elif use_fingerprints:
        # Remove the last element to compute the reward.
        state_indices = list(np.arange(
            0, ob_space.shape[0] - fingerprint_dim[0]))
    else:
        # All observations are presented in the goal.
        state_indices = list(np.arange(0, ob_space.shape[0]))

    return state_indices


def create_env(env, render=False, shared=False, maddpg=False, evaluate=False):
    """Return, and potentially create, the environment.

    Parameters
    ----------
    env : str or gym.Env
        the environment, or the name of a registered environment.
    render : bool
        whether to render the environment
    shared : bool
        specifies whether agents in an environment are meant to share policies.
        This is solely used by multi-agent Flow environments.
    maddpg : bool
        whether to use an environment variant that is compatible with the
        MADDPG algorithm
    evaluate : bool
        specifies whether this is a training or evaluation environment

    Returns
    -------
    gym.Env or list of gym.Env
        gym-compatible environment(s)
    """
    if env is None:
        # No environment (for evaluation environments).
        return None
    elif env in ENV_ATTRIBUTES.keys():
        env = ENV_ATTRIBUTES[env]["env"](
            evaluate, render, False, shared, maddpg)
    elif env.startswith("multiagent"):
        # multi-agent environments
        env_name = env[11:]
        env = ENV_ATTRIBUTES[env_name]["env"](
            evaluate, render, True, shared, maddpg)
    elif env in ["bottleneck0", "bottleneck1", "bottleneck2", "grid0",
                 "grid1"]:
        # Import the benchmark and fetch its flow_params
        benchmark = __import__("flow.benchmarks.{}".format(env),
                               fromlist=["flow_params"])
        flow_params = benchmark.flow_params

        # Get the env name and a creator for the environment.
        create_env, _ = make_create_env(flow_params, version=0, render=render)

        # Create the environment.
        env = create_env()
    elif isinstance(env, str):
        # This is assuming the environment is registered with OpenAI gym.
        env = gym.make(env)

    # Reset the environment.
    if env is not None:
        if isinstance(env, list):
            for next_env in env:
                next_env.reset()
        else:
            env.reset()

    return env