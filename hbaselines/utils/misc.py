"""Miscellaneous utility methods for this repository."""
import os
import errno
import numpy as np
from gym.spaces import Box
import gym

from hbaselines.envs.deeploco.envs import BipedalSoccer, BipedalObstacles
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
    from hbaselines.envs.mixed_autonomy.merge import get_flow_params as merge
    from hbaselines.envs.mixed_autonomy.ring import get_flow_params as ring
    from hbaselines.envs.mixed_autonomy.figure_eight import get_flow_params \
        as figure_eight
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    pass  # pragma: no cover


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
    if env_name in ["AntMaze", "AntPush", "AntFall", "AntGather",
                    "AntFourRooms"]:
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
    elif env_name == "PD-Biped3D-HLC-Soccer-v1":
        manager_ac_space = Box(
            low=np.array([0, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -1,
                          -2]),
            high=np.array([1.5, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2]),
            dtype=np.float32
        )
    elif env_name == "BipedalObstacles":
        # TODO: Brandon
        #  there may eb a better way of encoding this
        #  but this is most convenient for now
        ind = [0, 4, 5, 6, 7, 32, 33, 34, 50, 51, 52]
        manager_ac_space = gym.spaces.Box(
            low=np.array([x for i, x in enumerate([0.06586086, -0.33134258, -0.22473772, -0.08842427, -0.24940665,
                          -0.15451589, -0.49955064, -0.44008026, -0.59096092, -0.27878672,
                          -0.14038287, -0.28852576, -0.20101279, -0.22234532, -0.22769515,
                          -0.54376805, -0.35379013, -0.3725186, -0.33276483, -0.67418987,
                          -0.35524186, -0.45274141, -0.25600547, -0.86293733, -0.60379982,
                          -1.3963486, -1.35225046, -1.56099963, -1.59434652, -2.93630743,
                          -3.02572751, -4.52309895, -5.14550066, -1.79466832, -1.95292163,
                          -2.29718137, -2.53373265, -2.79888201, -3.67393041, -1.96048367,
                          -2.22237873, -4.52637959, -5.36702728, -1.79870808, -1.6695528,
                          -2.83235455, -2.84780359, -1.73784041, -2.26103067, -0.062334,
                          -0.61482263, -0.61719877, -0.61664611, -0.61482263, -1.00000000,
                          -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
                          -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
                          -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
                          -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
                          -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
                          -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
                          -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
                          -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
                          -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
                          -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000,
                          -1.00000000, -1.00000000, -1.00000000, -1.00000000, -1.00000000]) if i in ind]),
            high=np.array([x for i, x in enumerate([0.61076754, 0.30874607, 0.16389988, 0.278528, 0.10735691,
                           0.55370122, 0.28979328, 0.88343042, 0.46615249, 0.24841864,
                           0.25305298, 0.20827545, 0.35527417, 0.10670558, 0.34333566,
                           0.46612564, 0.34286582, 0.24609406, 0.55321878, 0.50907576,
                           0.41017145, 0.19810088, 0.49811089, 0.83155686, 0.40484139,
                           1.4751488, 1.06637669, 1.60812414, 1.50176299, 3.01205444,
                           3.09199214, 4.45173025, 5.29966736, 1.6375221, 1.83521891,
                           2.14798474, 2.5548656, 2.72522235, 3.7703712, 2.17525077,
                           1.90829098, 4.67793322, 5.20727777, 1.98003554, 1.36583793,
                           2.76746488, 2.68556261, 2.02427745, 1.82794178, 1.07712889,
                           1.10115075, 1.13575351, 1.12543523, 1.10115075, 1.00000000,
                           1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
                           1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
                           1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
                           1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
                           1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
                           1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
                           1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
                           1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
                           1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
                           1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
                           1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000]) if i in ind]),
            dtype=np.float32)
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
    if env_name in ["AntMaze", "AntPush", "AntFall", "AntGather",
                    "AntFourRooms"]:
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
    elif env_name == "PD-Biped3D-HLC-Soccer-v1":
        state_indices = [0, 4, 5, 6, 7, 32, 33, 34, 50, 51, 52, 57, 58, 59]
    elif env_name == "BipedalObstacles":
        # TODO: Brandon remove the image from the goal
        # the first 1024 channels are the image
        # the next 55 channels are positions
        # the next 55 channels are velocities
        # the last 2 channels are goals

        state_indices = [i + 1024 for i in [0, 4, 5, 6, 7, 32, 33, 34, 50, 51, 52]]
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
    if env == "AntGather":
        env = AntGatherEnv()

    elif env == "AntMaze":
        if evaluate:
            env = [
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
            ]
        else:
            env = AntMaze(
                use_contexts=True,
                random_contexts=True,
                context_range=[(-4, 20), (-4, 20)]
            )

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

    elif env == "AntFourRooms":
        if evaluate:
            env = [
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
            ]
        else:
            env = AntFourRooms(
                use_contexts=True,
                random_contexts=False,
                context_range=[[30, 0], [0, 30], [30, 30]]
            )

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

    elif env in ["ring0", "multiagent-ring0"]:
        if evaluate:
            env = [
                FlowEnv(
                    flow_params=ring(
                        ring_length=[230, 230],
                        evaluate=True,
                        multiagent=(env[:10] == "multiagent"),
                    ),
                    render=render,
                    multiagent=(env[:10] == "multiagent"),
                    shared=shared,
                    maddpg=maddpg,
                ),
                FlowEnv(
                    flow_params=ring(
                        ring_length=[260, 260],
                        evaluate=True,
                        multiagent=(env[:10] == "multiagent"),
                    ),
                    render=render,
                    multiagent=(env[:10] == "multiagent"),
                    shared=shared,
                    maddpg=maddpg,
                ),
                FlowEnv(
                    flow_params=ring(
                        ring_length=[290, 290],
                        evaluate=True,
                        multiagent=(env[:10] == "multiagent"),
                    ),
                    render=render,
                    multiagent=(env[:10] == "multiagent"),
                    shared=shared,
                    maddpg=maddpg,
                )
            ]
        else:
            env = FlowEnv(
                flow_params=ring(
                    evaluate=evaluate,
                    multiagent=(env[:10] == "multiagent"),
                ),
                render=render,
                multiagent=(env[:10] == "multiagent"),
                shared=shared,
                maddpg=maddpg,
            )

    elif env in ["merge0", "merge1", "merge2", "multiagent-merge0",
                 "multiagent-merge1", "multiagent-merge2"]:
        env_num = int(env[-1])
        env = FlowEnv(
            flow_params=merge(
                exp_num=env_num,
                horizon=6000,
                simulator="traci",
                multiagent=(env[:10] == "multiagent"),
            ),
            render=render,
            multiagent=(env[:10] == "multiagent"),
            shared=shared,
            maddpg=maddpg,
        )

    elif env in ["figureeight0", "figureeight1", "figureeight02",
                 "multiagent-figureeight0", "multiagent-figureeight1",
                 "multiagent-figureeight02"]:
        env_num = int(env[-1])
        env = FlowEnv(
            flow_params=figure_eight(
                num_automated=[1, 7, 14][env_num],
                horizon=1500,
                simulator="traci",
                multiagent=(env[:10] == "multiagent"),
            ),
            render=render,
            multiagent=(env[:10] == "multiagent"),
            shared=shared,
            maddpg=maddpg,
        )

    elif env == "BipedalSoccer":
        env = BipedalSoccer(render=render)

    elif env == "BipedalObstacles":
        env = BipedalObstacles(render=render)

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
