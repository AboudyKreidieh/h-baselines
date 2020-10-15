"""Utility methods when instantiating environments."""
import numpy as np
import os
import sys
import gym
from copy import deepcopy
from gym.spaces import Box

from hbaselines.envs.deeploco.envs import BipedalSoccer
from hbaselines.envs.deeploco.envs import BipedalObstacles
from hbaselines.envs.efficient_hrl.envs import AntMaze
from hbaselines.envs.efficient_hrl.envs import HumanoidMaze
from hbaselines.envs.efficient_hrl.envs import ImageAntMaze
from hbaselines.envs.efficient_hrl.envs import AntFall
from hbaselines.envs.efficient_hrl.envs import AntPush
from hbaselines.envs.efficient_hrl.envs import AntFourRooms
from hbaselines.envs.hac.envs import UR5, Pendulum

try:
    from hbaselines.envs.snn4hrl.envs import AntGatherEnv
    from hbaselines.envs.snn4hrl.envs import SnakeGatherEnv
    from hbaselines.envs.snn4hrl.envs import SwimmerGatherEnv
except (ImportError, ModuleNotFoundError):
    pass

try:
    import flow.config as config
    from hbaselines.envs.mixed_autonomy import FlowEnv
    from hbaselines.envs.mixed_autonomy.params.merge \
        import get_flow_params as merge
    from hbaselines.envs.mixed_autonomy.params.ring \
        import get_flow_params as ring
    from hbaselines.envs.mixed_autonomy.params.ring_small \
        import get_flow_params as ring_small
    from hbaselines.envs.mixed_autonomy.params.highway \
        import get_flow_params as highway
    from hbaselines.envs.mixed_autonomy.params.i210 \
        import get_flow_params as i210
except (ImportError, ModuleNotFoundError) as e:  # pragma: no cover
    # ray seems to have a bug that requires you to install ray[tune] twice
    if "ray" in str(e):  # pragma: no cover
        raise e  # pragma: no cover
    pass  # pragma: no cover

try:
    from hbaselines.envs.point2d import Point2DEnv
except (ImportError, ModuleNotFoundError):
    pass


# This dictionary element contains all relevant information when instantiating
# a single-agent, multi-agent, or hierarchical environment.
#
# The key in this dictionary in the name of the environment. The attributes for
# each element are:
#
# - meta_ac_space: a lambda function that takes an input whether the higher
#   level policies are assigning relative goals and returns the action space of
#   the higher level policies
# - state_indices: a list that assigns the indices that correspond to goals in
#   the Worker's state space
# - env: a lambda term that takes an input (evaluate, render, multiagent,
#   shared, maddpg) and return an environment or list of environments
ENV_ATTRIBUTES = {

    # ======================================================================= #
    # Variants of the AntMaze environment.                                    #
    # ======================================================================= #

    "AntMaze": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3, -0.5,
                          -0.3, -0.5, -0.3, -0.5, -0.3]),
            high=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5,
                           0.3, 0.5, 0.3]),
            dtype=np.float32,
        ),
        "state_indices": lambda multiagent: [i for i in range(15)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: [
            AntMaze(
                use_contexts=True,
                context_range=[16, 0],
                evaluate=True,
            ),
            AntMaze(
                use_contexts=True,
                context_range=[16, 16],
                evaluate=True,
            ),
            AntMaze(
                use_contexts=True,
                context_range=[0, 16],
                evaluate=True,
            )
        ] if evaluate else AntMaze(
            use_contexts=True,
            random_contexts=True,
            context_range=[(-4, 20), (-4, 20)],
            evaluate=False,
        ),
    },

    "HumanoidMaze": {
        "meta_ac_space": lambda relative_goals, multiagent: gym.spaces.Box(
            low=np.array([-10.0, -10.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                          0.785398, -0.9162995, -0.610865,
                          -0.26179925, -0.8290325, -1.134463, -1.3788117,
                          -0.26179925, -0.8290325, -1.134463, -1.3788117,
                          -1.265365, -1.265365, -1.2217325,
                          -1.265365, -1.265365, -1.2217325]),
            high=np.array([10.0, 10.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 0.785398, 0.9162995, 0.610865,
                           0.26179925, 0.8290325, 1.134463, 1.3788117,
                           0.26179925, 0.8290325, 1.134463, 1.3788117,
                           1.265365, 1.265365, 1.2217325,
                           1.265365, 1.265365, 1.2217325]), dtype=np.float32)
        if relative_goals else gym.spaces.Box(
            low=np.array([-2.0, -2.0, 0.0, -1.0, -1.0, -1.0, -1.0,
                          -0.7853980, -1.309, -0.610865,
                          -0.436332, -1.0472, -1.91986, -2.79253,
                          -0.436332, -1.0472, -1.91986, -2.79253,
                          -1.48353, -1.48353, -1.5708,
                          -1.0472, -1.0472, -1.5708]),
            high=np.array([10.0, 10.0, 2.0, 1.0, 1.0, 1.0, 1.0,
                           0.785398, 0.523599, 0.610865,
                           0.0872665, 0.610865, 0.349066, -0.0349066,
                           0.0872665, 0.610865, 0.349066, -0.0349066,
                           1.0472, 1.0472, 0.872665,
                           1.48353, 1.48353, 0.872665]), dtype=np.float32),
        "state_indices": lambda multiagent: list(range(24)),
        "env": lambda evaluate, render, multiagent, shared, maddpg: [
            HumanoidMaze(
                use_contexts=True,
                context_range=[8, 0]
            ),
            HumanoidMaze(
                use_contexts=True,
                context_range=[8, 8]
            ),
            HumanoidMaze(
                use_contexts=True,
                context_range=[0, 8]
            )
        ] if evaluate else HumanoidMaze(
            use_contexts=True,
            random_contexts=True,
            context_range=[(-2, 10), (-2, 10)]
        ),
    },

    "HumanoidMazeXY": {
        "meta_ac_space": lambda relative_goals, multiagent: gym.spaces.Box(
            low=np.array([-3.0, -3.0]),
            high=np.array([3.0, 3.0]), dtype=np.float32)
        if relative_goals else gym.spaces.Box(
            low=np.array([-2.0, -2.0]),
            high=np.array([10.0, 10.0]), dtype=np.float32),
        "state_indices": lambda multiagent: list(range(2)),
        "env": lambda evaluate, render, multiagent, shared, maddpg: [
            HumanoidMaze(
                use_contexts=True,
                context_range=[8, 0]
            ),
            HumanoidMaze(
                use_contexts=True,
                context_range=[8, 8]
            ),
            HumanoidMaze(
                use_contexts=True,
                context_range=[0, 8]
            )
        ] if evaluate else HumanoidMaze(
            use_contexts=True,
            random_contexts=True,
            context_range=[(-2, 10), (-2, 10)]
        ),
    },

    "ImageAntMaze": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3, -0.5,
                          -0.3, -0.5, -0.3, -0.5, -0.3]),
            high=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5,
                           0.3, 0.5, 0.3]),
            dtype=np.float32,
        ),
        "state_indices": lambda multiagent: [32*32*3 + i for i in range(15)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: [
            ImageAntMaze(
                use_contexts=True,
                context_range=[16, 0],
                image_size=32,
                evaluate=True,
            ),
            ImageAntMaze(
                use_contexts=True,
                context_range=[16, 16],
                image_size=32,
                evaluate=True,
            ),
            ImageAntMaze(
                use_contexts=True,
                context_range=[0, 16],
                image_size=32,
                evaluate=True,
            )
        ] if evaluate else ImageAntMaze(
            use_contexts=True,
            random_contexts=True,
            context_range=[(-4, 20), (-4, 20)],
            image_size=32,
            evaluate=False,
        ),
    },

    "AntPush": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3, -0.5,
                          -0.3, -0.5, -0.3, -0.5, -0.3]),
            high=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5,
                           0.3, 0.5, 0.3]),
            dtype=np.float32,
        ),
        "state_indices": lambda multiagent: [i for i in range(15)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: AntPush(
            use_contexts=True,
            context_range=[0, 19],
            evaluate=True,
        ) if evaluate else AntPush(
            use_contexts=True,
            context_range=[0, 19],
            # random_contexts=True,
            # context_range=[(-16, 16), (-4, 20)])
            evaluate=False,
        ),
    },

    "AntFall": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3, -0.5,
                          -0.3, -0.5, -0.3, -0.5, -0.3]),
            high=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5,
                           0.3, 0.5, 0.3]),
            dtype=np.float32,
        ),
        "state_indices": lambda multiagent: [i for i in range(15)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: AntFall(
            use_contexts=True,
            context_range=[0, 27, 4.5],
            evaluate=True,
        ) if evaluate else AntFall(
            use_contexts=True,
            context_range=[0, 27, 4.5],
            # random_contexts=True,
            # context_range=[(-4, 12), (-4, 28), (0, 5)])
            evaluate=False,
        ),
    },

    "AntFourRooms": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3, -0.5,
                          -0.3, -0.5, -0.3, -0.5, -0.3]),
            high=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5,
                           0.3, 0.5, 0.3]),
            dtype=np.float32,
        ),
        "state_indices": lambda multiagent: [i for i in range(15)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: [
            AntFourRooms(
                use_contexts=True,
                context_range=[20, 0],
                evaluate=True,
            ),
            AntFourRooms(
                use_contexts=True,
                context_range=[0, 20],
                evaluate=True,
            ),
            AntFourRooms(
                use_contexts=True,
                context_range=[20, 20],
                evaluate=True,
            )
        ] if evaluate else AntFourRooms(
            use_contexts=True,
            random_contexts=False,
            context_range=[[20, 0], [0, 20], [20, 20]],
            evaluate=False,
        ),
    },

    # ======================================================================= #
    # Gather environments.                                                    #
    # ======================================================================= #

    "SwimmerGather": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=np.array([-10, -10, -np.pi/2, -np.pi/2, -np.pi/2]),
            high=np.array([10, 10, np.pi/2, np.pi/2, np.pi/2]),
            dtype=np.float32,
        ),
        "state_indices": lambda multiagent: [i for i in range(5)],
        "env": lambda evaluate, render, multiagent, shared, maddpg:
        SwimmerGatherEnv(),
    },

    "SnakeGather": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=np.array([
                -10, -10, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2]),
            high=np.array(
                [10, 10, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2]),
            dtype=np.float32,
        ),
        "state_indices": lambda multiagent: [i for i in range(7)],
        "env": lambda evaluate, render, multiagent, shared, maddpg:
        SnakeGatherEnv(),
    },

    "AntGather": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3, -0.5,
                          -0.3, -0.5, -0.3, -0.5, -0.3]),
            high=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5,
                           0.3, 0.5, 0.3]),
            dtype=np.float32,
        ),
        "state_indices": lambda multiagent: [i for i in range(15)],
        "env": lambda evaluate, render, multiagent, shared, maddpg:
        AntGatherEnv(),
    },

    # ======================================================================= #
    # UR5 and Pendulum environments.                                          #
    # ======================================================================= #

    "UR5": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=np.array([-2 * np.pi, -2 * np.pi, -2 * np.pi, -4, -4, -4]),
            high=np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 4, 4, 4]),
            dtype=np.float32,
        ),
        "state_indices": lambda multiagent: None,
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
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=np.array([-np.pi, -15]),
            high=np.array([np.pi, 15]),
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [0, 2],
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

    "ring_small": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=-.5 if relative_goals else 0,
            high=.5 if relative_goals else 1,
            shape=(1,),
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [0],
        "env": lambda evaluate, render, multiagent, shared, maddpg: [
            FlowEnv(
                flow_params=ring_small(
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
                flow_params=ring_small(
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
                flow_params=ring_small(
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
            flow_params=ring_small(
                evaluate=evaluate,
                multiagent=multiagent,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    "ring-v0": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=-5 if relative_goals else 0,
            high=5 if relative_goals else 10,
            shape=(5,),
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [0],
        "env": lambda evaluate, render, multiagent, shared, maddpg: FlowEnv(
            flow_params=ring(
                stopping_penalty=True,
                acceleration_penalty=True,
                evaluate=evaluate,
                multiagent=multiagent,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    "merge-v0": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=-.5 if relative_goals else 0,
            high=.5 if relative_goals else 1,
            shape=(1 if multiagent else 5,),
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [
            5 * i for i in range(1 if multiagent else 5)],
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

    "merge-v1": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=-.5 if relative_goals else 0,
            high=.5 if relative_goals else 1,
            shape=(1 if multiagent else 13,),
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [
            5 * i for i in range(1 if multiagent else 13)],
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

    "merge-v2": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=-.5 if relative_goals else 0,
            high=.5 if relative_goals else 1,
            shape=(1 if multiagent else 17,),
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [
            5 * i for i in range(1 if multiagent else 17)],
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

    "highway-v0": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=-5 if relative_goals else 0,
            high=5 if relative_goals else 20,
            shape=(1 if multiagent else 10,),
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [
            5 * i for i in range(1 if multiagent else 10)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: FlowEnv(
            flow_params=highway(
                fixed_boundary=True,
                stopping_penalty=True,
                acceleration_penalty=True,
                use_follower_stopper=False,
                multiagent=multiagent,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    "highway-v1": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=-5 if relative_goals else 0,
            high=5 if relative_goals else 20,
            shape=(1 if multiagent else 10,),
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [
            5 * i for i in range(1 if multiagent else 10)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: FlowEnv(
            flow_params=highway(
                fixed_boundary=True,
                stopping_penalty=False,
                acceleration_penalty=True,
                use_follower_stopper=False,
                multiagent=multiagent,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    "highway-v2": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=-5 if relative_goals else 0,
            high=5 if relative_goals else 20,
            shape=(1 if multiagent else 10,),
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [
            5 * i for i in range(1 if multiagent else 10)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: FlowEnv(
            flow_params=highway(
                fixed_boundary=True,
                stopping_penalty=False,
                acceleration_penalty=False,
                use_follower_stopper=False,
                multiagent=multiagent,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    "highway-v3": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=-5 if relative_goals else 0,
            high=5 if relative_goals else 20,
            shape=(1 if multiagent else 10,),
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [
            5 * i for i in range(1 if multiagent else 10)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: FlowEnv(
            flow_params=highway(
                fixed_boundary=True,
                stopping_penalty=True,
                acceleration_penalty=True,
                use_follower_stopper=True,
                multiagent=multiagent,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    "i210-v0": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=-5 if relative_goals else 0,
            high=5 if relative_goals else 20,
            shape=(10 if multiagent else 50,),
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [
            5 * i for i in range(10 if multiagent else 50)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: FlowEnv(
            flow_params=i210(
                fixed_boundary=True,
                stopping_penalty=True,
                acceleration_penalty=True,
                use_follower_stopper=False,
                multiagent=multiagent,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    "i210-v1": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=-5 if relative_goals else 0,
            high=5 if relative_goals else 20,
            shape=(10 if multiagent else 50,),
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [
            5 * i for i in range(10 if multiagent else 50)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: FlowEnv(
            flow_params=i210(
                fixed_boundary=True,
                stopping_penalty=False,
                acceleration_penalty=True,
                use_follower_stopper=False,
                multiagent=multiagent,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    "i210-v2": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=-5 if relative_goals else 0,
            high=5 if relative_goals else 20,
            shape=(10 if multiagent else 50,),
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [
            5 * i for i in range(10 if multiagent else 50)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: FlowEnv(
            flow_params=i210(
                fixed_boundary=True,
                stopping_penalty=False,
                acceleration_penalty=False,
                use_follower_stopper=False,
                multiagent=multiagent,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    "i210-v3": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=-5 if relative_goals else 0,
            high=5 if relative_goals else 20,
            shape=(10 if multiagent else 50,),
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [
            5 * i for i in range(10 if multiagent else 50)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: FlowEnv(
            flow_params=i210(
                fixed_boundary=True,
                stopping_penalty=True,
                acceleration_penalty=True,
                use_follower_stopper=True,
                multiagent=multiagent,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    # ======================================================================= #
    # Mixed autonomy traffic imitation environments.                          #
    # ======================================================================= #

    "ring-imitation": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=-1 if relative_goals else 0,
            high=1,
            shape=(5,),
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [5 * i for i in range(5)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: FlowEnv(
            flow_params=ring(
                stopping_penalty=False,
                acceleration_penalty=False,
                evaluate=evaluate,
                multiagent=multiagent,
                imitation=True,
            ),
            render=render,
            multiagent=multiagent,
            shared=shared,
            maddpg=maddpg,
        ),
    },

    "highway-imitation": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=-1 if relative_goals else 0,
            high=1,
            shape=(10,),
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [5 * i for i in range(10)],
        "env": lambda evaluate, render, multiagent, shared, maddpg: FlowEnv(
            flow_params=highway(
                fixed_boundary=True,
                stopping_penalty=False,
                acceleration_penalty=False,
                use_follower_stopper=False,
                evaluate=evaluate,
                multiagent=multiagent,
                imitation=True,
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
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            low=np.array([-0.5, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -1,
                          -2]),
            high=np.array([0.5, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2]),
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [
            0, 4, 5, 6, 7, 32, 33, 34, 50, 51, 52, 57, 58, 59],
        "env": lambda evaluate, render, multiagent, shared, maddpg:
        BipedalSoccer(render=render),
    },

    "BipedalObstacles": {
        "meta_ac_space": lambda relative_goals, multiagent: gym.spaces.Box(
            low=np.array([0, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2]),
            high=np.array([1.5, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]),
            dtype=np.float32),
        "state_indices": lambda multiagent: [i + 1024 for i in [
            0, 4, 5, 6, 7, 32, 33, 34, 50, 51, 52]],
        "env": lambda evaluate, render, multiagent, shared, maddpg:
        BipedalObstacles(render=render),
    },

    # ======================================================================= #
    # Point navigation environments.                                          #
    # ======================================================================= #

    "Point2DEnv": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            np.ones(2) * -4,
            np.ones(2) * 4,
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [0, 1],
        "env": lambda evaluate, render, multiagent, shared, maddpg: Point2DEnv(
            images_in_obs=False
        ),
    },

    "Point2DImageEnv": {
        "meta_ac_space": lambda relative_goals, multiagent: Box(
            np.ones(2) * -4,
            np.ones(2) * 4,
            dtype=np.float32
        ),
        "state_indices": lambda multiagent: [3072, 3073],
        "env": lambda evaluate, render, multiagent, shared, maddpg: Point2DEnv(
            images_in_obs=True
        ),
    },
}


def get_meta_ac_space(ob_space, relative_goals, env_name):
    """Compute the action space for the higher level policies.

    Parameters
    ----------
    ob_space : gym.spaces.*
        the observation space of the environment
    relative_goals : bool
        specifies whether the goal issued by the meta-level policy is meant to
        be a relative or absolute goal, i.e. specific state or change in state
    env_name : str
        the name of the environment. Used for special cases to assign the
        meta-level policies' action space to only ego observations in the
        observation space.

    Returns
    -------
    gym.spaces.Box
        the action space of the higher level policy
    """
    # Handle multi-agent environments.
    multiagent = env_name.startswith("multiagent")
    if multiagent:
        env_name = env_name[11:]

    if env_name in ENV_ATTRIBUTES.keys():
        meta_ac_space = ENV_ATTRIBUTES[env_name]["meta_ac_space"](
            relative_goals, multiagent)
    else:
        meta_ac_space = ob_space

    return meta_ac_space


def get_state_indices(ob_space, env_name):
    """Return the state indices for the intrinsic rewards.

    This assigns the indices of the state that are assigned goals, and
    subsequently rewarded for performing those goals.

    Parameters
    ----------
    ob_space : gym.spaces.*
        the observation space of the environment
    env_name : str
        the name of the environment. Used for special cases to assign the
        meta-level policies' action space to only ego observations in the
        observation space.

    Returns
    -------
    list of int
        the state indices that are assigned goals
    """
    # Handle multi-agent environments.
    multiagent = env_name.startswith("multiagent")
    if multiagent:
        env_name = env_name[11:]

    if env_name in ENV_ATTRIBUTES.keys():
        state_indices = ENV_ATTRIBUTES[env_name]["state_indices"](multiagent)
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
    gym.Env or list of gym.Env or None
        gym-compatible environment(s). Set to None if no environment is being
        returned.
    array_like or list of array_like or None
        the observation(s) from the environment(s) upon reset. Set to None if
        no environment is being returned.
    """
    if env is None:
        # No environment (for evaluation environments).
        return None, None

    elif isinstance(env, str):
        if env in ENV_ATTRIBUTES.keys() or env.startswith("multiagent"):
            # Handle multi-agent environments.
            multiagent = env.startswith("multiagent")
            if multiagent:
                env = env[11:]

            env = ENV_ATTRIBUTES[env]["env"](
                evaluate, render, multiagent, shared, maddpg)

        elif env.startswith("flow:"):
            # environments in flow/examples
            env = import_flow_env(env, render, shared, maddpg, evaluate)

        else:
            # This is assuming the environment is registered with OpenAI gym.
            env = gym.make(env)

    # Reset the environment.
    if isinstance(env, list):
        obs = [next_env.reset() for next_env in env]
    else:
        obs = env.reset()

    return env, obs


def import_flow_env(env_name, render, shared, maddpg, evaluate):
    """Import an environment from the flow/examples folder.

    This method imports the flow_params dict from the exp_configs folders in
    this directory and generates an appropriate FlowEnv object.

    Parameters
    ----------
    env_name : str
        the environment name. Starts with "flow:" to signify that it should be
        imported from the flow/experiments folder.
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
    hbaselines.envs.mixed_autonomy.FlowEnv
        the training/evaluation environment

    Raises
    ------
    ValueError
        if the environment is not abailable in flow/examples
    """
    # Parse the exp_config name from the environment name
    exp_config = env_name[5:]

    # Add flow/examples to your path to located the below modules.
    sys.path.append(os.path.join(config.PROJECT_PATH, "examples"))

    # Import relevant information from the exp_config script.
    module = __import__("exp_configs.rl.singleagent", fromlist=[exp_config])
    module_ma = __import__("exp_configs.rl.multiagent", fromlist=[exp_config])

    # Import the sub-module containing the specified exp_config and determine
    # whether the environment is single agent or multi-agent.
    if hasattr(module, exp_config):
        submodule = getattr(module, exp_config)
        multiagent = False
    elif hasattr(module_ma, exp_config):
        submodule = getattr(module_ma, exp_config)
        multiagent = True
    else:
        raise ValueError("Unable to find experiment config.")

    # Collect the flow_params object.
    flow_params = deepcopy(submodule.flow_params)

    # Update the evaluation flag to match what is requested.
    flow_params['env'].evaluate = evaluate

    # Return the environment.
    return FlowEnv(
        flow_params,
        multiagent=multiagent,
        shared=shared,
        maddpg=maddpg,
        render=render,
    )
