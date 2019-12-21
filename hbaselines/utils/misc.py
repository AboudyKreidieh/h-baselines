"""Miscellaneous utility methods for this repository."""
import os
import errno
import numpy as np
from gym.spaces import Box
import gym

try:
    from flow.utils.registry import make_create_env
    from hbaselines.envs.mixed_autonomy import FlowEnv
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    pass  # pragma: no cover
from hbaselines.envs.efficient_hrl.envs import AntMaze, AntFall, AntPush
from hbaselines.envs.hac.envs import UR5, Pendulum
try:
    from hbaselines.envs.snn4hrl.envs import AntGatherEnv
except (ImportError, ModuleNotFoundError):
    pass


def ensure_dir(path):
    """Ensure that the directory specified exists, and if not, create it."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise  # pragma: no cover
    return path


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
    if env_name in ["AntMaze", "AntPush", "AntFall", "AntGather"]:
        manager_ac_space = Box(
            low=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3, -0.5,
                          -0.3, -0.5, -0.3, -0.5, -0.3]),
            high=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3,
                           0.5, 0.3, 0.5, 0.3]),
            dtype=np.float32,
        )
    elif env_name == "UR5":
        manager_ac_space = Box(
            low=np.array([-2 * np.pi, -2 * np.pi, -2 * np.pi, -4, -4, -4]),
            high=np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 4, 4, 4]),
            dtype=np.float32,
        )
    elif env_name == "Pendulum":
        manager_ac_space = Box(
            low=np.array([-np.pi, -15]),
            high=np.array([np.pi, 15]),
            dtype=np.float32
        )
    elif env_name in ["ring0", "ring1"]:
        if relative_goals:
            manager_ac_space = Box(-.5, .5, shape=(1,), dtype=np.float32)
        else:
            manager_ac_space = Box(0, 1, shape=(1,), dtype=np.float32)
    elif env_name == "figureeight0":
        if relative_goals:
            manager_ac_space = Box(-.5, .5, shape=(1,), dtype=np.float32)
        else:
            manager_ac_space = Box(0, 1, shape=(1,), dtype=np.float32)
    elif env_name == "figureeight1":
        if relative_goals:
            manager_ac_space = Box(-.5, .5, shape=(7,), dtype=np.float32)
        else:
            manager_ac_space = Box(0, 1, shape=(7,), dtype=np.float32)
    elif env_name == "figureeight2":
        if relative_goals:
            manager_ac_space = Box(-.5, .5, shape=(14,), dtype=np.float32)
        else:
            manager_ac_space = Box(0, 1, shape=(14,), dtype=np.float32)
    elif env_name == "merge0":
        if relative_goals:
            manager_ac_space = Box(-.5, .5, shape=(5,), dtype=np.float32)
        else:
            manager_ac_space = Box(0, 1, shape=(5,), dtype=np.float32)
    elif env_name == "merge1":
        if relative_goals:
            manager_ac_space = Box(-.5, .5, shape=(13,), dtype=np.float32)
        else:
            manager_ac_space = Box(0, 1, shape=(13,), dtype=np.float32)
    elif env_name == "merge2":
        if relative_goals:
            manager_ac_space = Box(-.5, .5, shape=(17,), dtype=np.float32)
        else:
            manager_ac_space = Box(0, 1, shape=(17,), dtype=np.float32)
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
    # remove the last element to compute the reward FIXME
    if use_fingerprints:
        state_indices = list(np.arange(
            0, ob_space.shape[0] - fingerprint_dim[0]))
    else:
        state_indices = None

    if env_name in ["AntMaze", "AntPush", "AntFall", "AntGather"]:
        state_indices = list(np.arange(0, 15))
    elif env_name == "UR5":
        state_indices = None
    elif env_name == "Pendulum":
        state_indices = [0, 2]
    elif env_name in ["ring0", "ring1"]:
        state_indices = [0]
    elif env_name == "figureeight0":
        state_indices = [13]
    elif env_name == "figureeight1":
        state_indices = [i for i in range(1, 14, 2)]
    elif env_name == "figureeight2":
        state_indices = [i for i in range(14)]
    elif env_name == "merge0":
        state_indices = [5 * i for i in range(5)]
    elif env_name == "merge1":
        state_indices = [5 * i for i in range(13)]
    elif env_name == "merge2":
        state_indices = [5 * i for i in range(17)]

    return state_indices


def create_env(env, render=False, evaluate=False):
    """Return, and potentially create, the environment.

    Parameters
    ----------
    env : str or gym.Env
        the environment, or the name of a registered environment.
    render : bool
        whether to render the environment
    evaluate : bool
        specifies whether this is a training or evaluation environment

    Returns
    -------
    gym.Env or list of gym.Env
        gym-compatible environment(s)
    """
    if env == "AntGather":
        env = AntGatherEnv()

    if env == "AntMaze":
        if evaluate:
            env = [AntMaze(use_contexts=True, context_range=[16, 0]),
                   AntMaze(use_contexts=True, context_range=[16, 16]),
                   AntMaze(use_contexts=True, context_range=[0, 16])]
        else:
            env = AntMaze(use_contexts=True,
                          random_contexts=True,
                          context_range=[(-4, 20), (-4, 20)])

    elif env == "AntPush":
        if evaluate:
            env = AntPush(use_contexts=True, context_range=[0, 19])
        else:
            env = AntPush(use_contexts=True, context_range=[0, 19])
            # env = AntPush(use_contexts=True,
            #               random_contexts=True,
            #               context_range=[(-16, 16), (-4, 20)])

    elif env == "AntFall":
        if evaluate:
            env = AntFall(use_contexts=True, context_range=[0, 27, 4.5])
        else:
            env = AntFall(use_contexts=True, context_range=[0, 27, 4.5])
            # env = AntFall(use_contexts=True,
            #               random_contexts=True,
            #               context_range=[(-4, 12), (-4, 28), (0, 5)])

    elif env == "UR5":
        if evaluate:
            env = UR5(use_contexts=True,
                      random_contexts=True,
                      context_range=[(-np.pi, np.pi), (-np.pi / 4, 0),
                                     (-np.pi / 4, np.pi / 4)],
                      show=render)
        else:
            env = UR5(use_contexts=True,
                      random_contexts=True,
                      context_range=[(-np.pi, np.pi), (-np.pi / 4, 0),
                                     (-np.pi / 4, np.pi / 4)],
                      show=render)

    elif env == "Pendulum":
        if evaluate:
            env = Pendulum(use_contexts=True, context_range=[0, 0],
                           show=render)
        else:
            env = Pendulum(use_contexts=True,
                           random_contexts=True,
                           context_range=[(np.deg2rad(-16), np.deg2rad(16)),
                                          (-0.6, 0.6)],
                           show=render)

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

    elif env in ["ring0", "multi-ring0"]:
        env = FlowEnv("ring", render=render)  # FIXME

    elif env in ["merge0", "merge1", "merge2", "multi-merge0", "multi-merge1",
                 "multi-merge2"]:
        env_num = int(env[-1])
        env = FlowEnv(
            "merge",
            env_params={
                "exp_num": env_num,
                "horizon": 6000,
                "simulator": "traci",
                "multiagent": env[:5] == "multi"
            },
            render=render
        )

    elif env in ["figureeight0", "figureeight1", "figureeight02",
                 "multi-figureeight0", "multi-figureeight1",
                 "multi-figureeight02"]:
        env_num = int(env[-1])
        env = FlowEnv(
            "figure_eight",
            env_params={
                "num_automated": [1, 7, 14][env_num],
                "horizon": 750,
                "simulator": "traci",
                "multiagent": env[:5] == "multi"
            },
            render=render
        )

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
